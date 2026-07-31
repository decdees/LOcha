"""Pipeline instrumentation (T2.1, built before the first real FrameProcessor).

Two jobs, both of which get harder to add later:

1. Map Pipecat frames onto `TurnState` so PRD G1a is assertable from the first
   commit rather than retrofitted through five components.
2. Time each stage, so ARCHITECTURE §5.2's named worst failure -- a processor
   that buffers and breaks the streaming chain -- shows up when it is introduced
   instead of at the end of Phase 2.

**This processor must never buffer, reorder, or drop a frame.** It is a tap, not
a stage. Every frame is forwarded immediately and unchanged; a test asserts
identity and order, because an instrument that perturbs the thing it measures is
worse than no instrument.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.speech.wire import state_message
from ocha.turnstate import TurnState, TurnTimeline

# Frame -> the state the user is shown. Deliberately explicit: a frame absent
# here emits nothing, which is safer than guessing a state nobody can render.
_STATE_FOR: dict[type[Frame], TurnState] = {
    UserStartedSpeakingFrame: TurnState.LISTENING,
    UserStoppedSpeakingFrame: TurnState.TRANSCRIBING,
    # Interim transcripts are the G1a workhorse: they break the long silent
    # stretch between endpoint and first audio into visible updates.
    InterimTranscriptionFrame: TurnState.TRANSCRIBING,
    TranscriptionFrame: TurnState.THINKING,
    LLMFullResponseStartFrame: TurnState.THINKING,
    TTSStartedFrame: TurnState.SPEAKING,
    BotStartedSpeakingFrame: TurnState.SPEAKING,
    BotStoppedSpeakingFrame: TurnState.IDLE,
}


@dataclass(slots=True)
class Spans:
    """Wall-clock marks for the stages §5.1 budgets."""

    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, at: float) -> None:
        self.marks.setdefault(name, at)  # first occurrence wins

    def span(self, a: str, b: str) -> float | None:
        if a in self.marks and b in self.marks:
            return self.marks[b] - self.marks[a]
        return None

    def voice_to_first_audio(self) -> float | None:
        """PRD G1b's metric, measured on the live pipeline rather than summed."""
        return self.span("user_stopped", "first_audio")


class TurnStateProbe(FrameProcessor):
    """A pass-through tap that records state and timing."""

    def __init__(  # type: ignore[no-untyped-def]
        self,
        timeline: TurnTimeline | None = None,
        clock=time.monotonic,
        emit_state: bool = False,
    ) -> None:
        super().__init__()
        self.timeline = timeline if timeline is not None else TurnTimeline(_clock=clock)
        self.spans = Spans()
        self.seen: list[type[Frame]] = []
        # When true, every state change is also pushed to the client as an urgent
        # transport message. G1a is a client-visible criterion, so the component
        # that decides the state is the one that should announce it -- deriving it
        # a second time in the app would be two implementations of one rule.
        self._emit_state = emit_state
        # NOT self._clock: FrameProcessor already owns that name as a BaseClock,
        # and shadowing it breaks Pipecat's internal timing. mypy caught this.
        self._now = clock

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        now = self._now()
        self.seen.append(type(frame))

        if isinstance(frame, UserStoppedSpeakingFrame):
            self.spans.mark("user_stopped", now)
        elif isinstance(frame, TranscriptionFrame):
            self.spans.mark("transcript", now)
        elif isinstance(frame, LLMTextFrame):
            self.spans.mark("first_token", now)
        elif isinstance(frame, TTSStartedFrame | BotStartedSpeakingFrame):
            self.spans.mark("first_audio", now)

        state = _STATE_FOR.get(type(frame))
        if state is not None:
            last = self.timeline.changes[-1].state if self.timeline.changes else None
            # Interim transcripts repeat; re-emitting the same state is exactly
            # what keeps the G1a gap short, so repeats are allowed for
            # TRANSCRIBING and suppressed elsewhere to avoid noise.
            if state is not last or state is TurnState.TRANSCRIBING:
                self.timeline.emit(state, type(frame).__name__)
                if self._emit_state:
                    # Urgent: it must overtake queued audio. A "listening" badge
                    # that arrives behind two seconds of buffered playback is not
                    # feedback, it is a stale label.
                    await self.push_frame(
                        OutputTransportMessageUrgentFrame(message=state_message(state.value)),
                        FrameDirection.DOWNSTREAM,
                    )

        # Forward immediately. Never buffer -- see the module docstring.
        await self.push_frame(frame, direction)

    def report(self) -> dict[str, float | None | bool]:
        return {
            "asr_s": self.spans.span("user_stopped", "transcript"),
            "llm_ttft_s": self.spans.span("transcript", "first_token"),
            "tts_s": self.spans.span("first_token", "first_audio"),
            "voice_to_first_audio_s": self.spans.voice_to_first_audio(),
            "satisfies_g1a": self.timeline.satisfies_g1a(),
        }
