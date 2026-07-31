"""The tutor stage (T2.4) — Phase 1's logic, inside the pipeline.

This does **not** use Pipecat's LLM services or `LLMContext`. Phase 1 already
owns the context builder, the firewall, usage detection and FSRS scoring, and
history lives in SQLite. Adopting Pipecat's context aggregators would mean two
places deciding what the model sees, and the firewall would have to be
reimplemented as a text filter on the way out. Instead this stage calls
`run_turn`, so there is exactly one implementation of the turn.

## Streaming, and how the firewall survives it

The firewall is inviolable (standing constraint 2): when the model emits
`[GRAMMAR_QUERY]` its own text must never reach the user. Forwarding tokens as
they arrive would break that -- by the time the sentinel is recognised, its
neighbours have been spoken.

So generation streams, but **emission is gated on the sentence**. Tokens
accumulate until the first 。！？ (or the end of generation); the firewall runs on
that prefix; only then is anything pushed. After the gate opens, each further
completed sentence is pushed as it finishes.

This is not a compromise of §5.2 rule 1 -- rule 2 wants synthesis to start at the
first *sentence*, and the sentinel is short and contains no sentence punctuation,
so the gate costs nothing correctness was not already charging for.

**This started out as "generate the whole reply first", which measurement
rejected.** Holding until generation finished put the LLM stage at 1.99 s against
T0.7's 0.75 s first-sentence figure, and pushed voice-to-first-audio to 3.67 s --
outside PRD G1b's 3.2 s p50 bound. Streaming alone then bought nothing, because
generation was blocking the event loop and a pushed frame is not a delivered one.
Both facts are why the `InferenceWorker` exists. See benchmarks/voice-loop.md.

## Threading

Split, and the split is load-bearing:

- **Generation runs on the `InferenceWorker` thread**, which owns both models for
  the process lifetime (constraint 6). Chunks arrive here through an asyncio queue,
  so the event loop stays free while the model works -- that is what lets VOICEVOX
  synthesise sentence one while sentence two is still being generated.
- **`run_turn` runs on the event loop**, because the SQLite connection is bound to
  the thread that opened it. Sending DB work to the worker reproduces the T1.8
  thread-affinity bug from the other direction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.scheduling.scheduler import ItemScheduler
from ocha.tutor.context import MAX_REPLY_TOKENS, build_context
from ocha.tutor.firewall import apply_firewall
from ocha.tutor.grammar import GrammarReference
from ocha.tutor.llm import LlmService
from ocha.tutor.turn import TurnResult, run_turn

SENTENCE_ENDINGS = "。！？"


class TutorStage(FrameProcessor):
    """Transcript in, tutor text out. Owns the turn, not just the generation."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        scheduler: ItemScheduler,
        reference: GrammarReference,
        llm: LlmService,
    ) -> None:
        super().__init__()
        self._conn = conn
        self._scheduler = scheduler
        self._reference = reference
        self._llm = llm
        self._session_id: int | None = None
        self.last_result: TurnResult | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self.push_frame(frame, direction)  # the client shows what it heard
            await self._handle(frame.text.strip())
            return
        await self.push_frame(frame, direction)

    async def _chunks(self, system_prompt: str, user_text: str) -> AsyncIterator[str]:
        """Token chunks off the worker thread, or off a stub in tests.

        `astream` is the worker-backed path. A plain `LlmService` (the stubs, and
        anything not backed by MLX) has only the synchronous `stream`, which is
        fine to iterate inline precisely because it is not doing GPU work.
        """
        astream = getattr(self._llm, "astream", None)
        if astream is not None:
            async for chunk in astream(system_prompt, user_text, max_tokens=MAX_REPLY_TOKENS):
                yield str(chunk)
            return
        for chunk in self._llm.stream(system_prompt, user_text, max_tokens=MAX_REPLY_TOKENS):
            yield chunk

    async def _handle(self, user_text: str) -> None:
        await self.push_frame(LLMFullResponseStartFrame())

        raw, emitted = await self._stream_gated(user_text)

        # The turn is completed by Phase 1's own code path: firewall (again, on the
        # full text), usage detection, FSRS scoring, persistence. `raw` is passed so
        # this does not generate a second, different reply.
        result = run_turn(
            self._conn,
            self._scheduler,
            self._reference,
            self._llm,
            user_text,
            session_id=self._session_id,
            raw=raw,
        )
        self._session_id = result.session_id
        self.last_result = result

        if result.grammar is not None:
            # A grammar answer is text, never speech: it is reference material with
            # examples and a Hindi contrast note, and synthesising it would both
            # mangle the romaji and imply the tutor authored it.
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={
                        "type": "grammar",
                        "kind": result.grammar.kind,
                        "text": result.grammar.text,
                        "entry_id": result.grammar.entry_id,
                        "examples": list(result.grammar.examples),
                        "hindi_contrast": result.grammar.hindi_contrast,
                        "interference_warning": result.grammar.interference_warning,
                    }
                )
            )
        elif result.reply:
            # Whatever the gate has not already sent. Normally nothing: the streamed
            # sentences and the firewalled reply are the same text. They differ only
            # if the firewall rewrote it, and then the rewrite is what ships.
            remainder = (
                result.reply[len(emitted) :] if result.reply.startswith(emitted) else result.reply
            )
            for sentence in _split_sentences(remainder):
                await self.push_frame(LLMTextFrame(sentence))

        await self.push_frame(LLMFullResponseEndFrame())

    async def _stream_gated(self, user_text: str) -> tuple[str, str]:
        """Stream the reply, emitting complete sentences once the firewall allows.

        Returns `(everything generated, everything emitted)`. Nothing is emitted
        before `apply_firewall` has seen the first sentence, and if the sentinel
        fires, generation stops and not one token is pushed.

        Generation happens on the worker thread; this coroutine only consumes
        chunks, so the event loop stays free to deliver what it pushes.
        """
        ctx = build_context(self._scheduler)
        raw = ""
        emitted = ""
        gate_open = False
        pending = ""

        async for chunk in self._chunks(ctx.system_prompt, user_text):
            raw += chunk
            pending += chunk
            if not any(ch in pending for ch in SENTENCE_ENDINGS):
                continue

            if not gate_open:
                # The one check that matters. If the sentinel is anywhere in the
                # prefix, stop: the grammar path owns the response from here and the
                # model's own words are discarded, unsent.
                if apply_firewall(raw, user_text, self._reference, self._conn).fired:
                    return raw, ""
                gate_open = True

            for sentence in _split_sentences(pending):
                await self.push_frame(LLMTextFrame(sentence))
                emitted += sentence
            pending = ""

        # A reply that never reached a sentence boundary -- or a sentinel-only one.
        if pending.strip():
            if not gate_open and apply_firewall(raw, user_text, self._reference, self._conn).fired:
                return raw, ""
            await self.push_frame(LLMTextFrame(pending))
            emitted += pending
        return raw, emitted


def _split_sentences(text: str) -> list[str]:
    """Split after 。！？, keeping the punctuation. Never returns an empty string."""
    out: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in SENTENCE_ENDINGS:
            out.append(text[start : i + 1])
            start = i + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out or [text]
