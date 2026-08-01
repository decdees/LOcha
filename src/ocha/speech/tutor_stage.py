"""The tutor stage (T2.4) — Phase 1's logic, inside the pipeline.

This does **not** use Pipecat's LLM services or `LLMContext`. Phase 1 already
owns the context builder, the firewall, conservative observations and persistence,
and explicit history is loaded from SQLite. Adopting Pipecat's context aggregators
would mean two places deciding what the model sees, and the firewall would have to
be reimplemented as a text filter on the way out. Instead this stage calls the
same ``finalize_turn`` boundary as HTTP, so there is exactly one implementation.

## Quarantine and the firewall

The firewall is inviolable (standing constraint 2): when the model emits
`[GRAMMAR_QUERY]` its own text must never reach the user. Forwarding tokens as
they arrive would break that -- by the time the sentinel is recognised, its
neighbours have been spoken.

The complete generation is collected on the inference worker. Nothing derived
from model output is emitted until the complete reply has passed through the
shared ``finalize_turn`` boundary. A sentinel at any position therefore suppresses
all model text and audio. This deliberately pays the latency cost of quarantine.

## Threading

Split, and the split is load-bearing:

- **Complete generation runs on the `InferenceWorker` thread**, which owns both
  models for the process lifetime (constraint 6). Only the completed string is
  returned to the event loop.
- **`finalize_turn` runs on the event loop**, because the SQLite connection is
  bound to the thread that opened it. Sending DB work to the worker reproduces
  the T1.8 thread-affinity bug from the other direction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

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
from ocha.tutor.grammar import GrammarReference
from ocha.tutor.llm import ChatMessage, LlmService
from ocha.tutor.turn import TurnResult, conversation_history, ensure_session, finalize_turn

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

    async def _handle(self, user_text: str) -> None:
        await self.push_frame(LLMFullResponseStartFrame())

        self._session_id = ensure_session(self._conn, self._session_id)
        ctx = build_context(self._scheduler)
        history = conversation_history(self._conn, self._session_id)
        raw = await self._collect(ctx.system_prompt, user_text, history)

        # The turn is completed by Phase 1's own code path: firewall on the full
        # text, conservative observations and persistence. Free conversation
        # never creates an FSRS rating.
        result = finalize_turn(
            self._conn,
            self._scheduler,
            self._reference,
            user_text,
            context=ctx,
            raw=raw,
            session_id=self._session_id,
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
            for sentence in _split_sentences(result.reply):
                await self.push_frame(LLMTextFrame(sentence))

        await self.push_frame(LLMFullResponseEndFrame())

    async def _collect(
        self,
        system_prompt: str,
        user_text: str,
        history: Sequence[ChatMessage],
    ) -> str:
        """Collect the complete generation without emitting model-derived frames."""
        agenerate = getattr(self._llm, "agenerate", None)
        if agenerate is not None:
            value = await agenerate(
                system_prompt, user_text, history=history, max_tokens=MAX_REPLY_TOKENS
            )
            return str(value)
        return self._llm.generate(
            system_prompt, user_text, history=history, max_tokens=MAX_REPLY_TOKENS
        )


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
