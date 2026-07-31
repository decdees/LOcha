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
outside PRD G1b's 3.2 s p50 bound. See benchmarks/voice-loop.md. The laziest
version of this stage was not viable, and the measurement is what said so rather
than an argument about streaming being good practice.

## Threading

Everything here runs on the event loop, deliberately, and must keep doing so.
MLX GPU streams are thread-local to the thread that ran `load()` (constraint 6),
and the SQLite connection is bound to the thread that opened it -- both are the
lifespan thread. `asyncio.to_thread` around either one fails at runtime and no
stub-based test would catch it.
"""

from __future__ import annotations

import asyncio
import sqlite3

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

# Long enough for a pushed frame to reach the TTS service and for its synthesis
# thread to be dispatched; short enough to be noise against a ~50 ms token. Paid
# once per sentence, not per token.
DELIVERY_YIELD_S = 0.001


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

        Blocking on the event loop between tokens, deliberately -- MLX streams are
        thread-local to the loading thread (constraint 6) and the SQLite connection
        is bound to the same thread. See "Threading" below.
        """
        ctx = build_context(self._scheduler)
        raw = ""
        emitted = ""
        gate_open = False
        pending = ""

        for chunk in self._llm.stream(ctx.system_prompt, user_text, max_tokens=MAX_REPLY_TOKENS):
            raw += chunk
            pending += chunk
            # Yield on every token, not only at sentence boundaries. Sentence-only
            # yielding got the first sentence into TTS early but the *audio* still
            # could not come back: VOICEVOX synthesises in a thread and finishes
            # while this loop is generating, and the frames it produced needed the
            # event loop to be delivered. Measured: tts_s went from 0.69 s to 1.20 s
            # -- the wait moved rather than disappearing. ~1 ms per token against a
            # ~50 ms token is noise; a blocked loop is not.
            await asyncio.sleep(DELIVERY_YIELD_S)
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
            # Yield the loop, or the sentence just pushed is not *delivered* until
            # generation ends. push_frame only queues; the downstream processor
            # drains that queue in another task, and this one owns the event loop
            # for the whole generation (constraint 6 forbids moving it off).
            # Without this the gate opened on time and bought nothing: measured
            # 2.07 s to first text either way. `sleep(0)` alone was also not enough --
            # it yields exactly one pass, and this task becomes ready again before the
            # frame has crossed the remaining processors, so it blocks for the next
            # token instead. A small real delay lets the chain drain and VOICEVOX start
            # on sentence one, overlapping the rest of generation.
            await asyncio.sleep(DELIVERY_YIELD_S)

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
