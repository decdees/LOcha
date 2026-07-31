"""Pipeline instrumentation (T2.1, built before the first real FrameProcessor).

Two jobs, both of which get harder to add later:

1. Map Pipecat frames onto `TurnState` so PRD G1a is assertable from the first
   commit rather than retrofitted through five components.
2. Time each stage, so ARCHITECTURE §5.2's named worst failure -- a processor
   that buffers and breaks the streaming chain -- shows up when it is introduced
   instead of at the end of Phase 2.

Several instances can share one `Spans` and one `TurnTimeline`, which is how the
pipeline taps more than one point with a single instrument -- see
`speech/pipeline.py`.

**This processor must never buffer, reorder, or drop a frame.** It is a tap, not
a stage. Every frame is forwarded immediately and unchanged; a test asserts
identity and order, because an instrument that perturbs the thing it measures is
worse than no instrument.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.speech.filler import FillerAudioFrame
from ocha.speech.wire import state_message
from ocha.turnstate import TurnState, TurnTimeline

# Frame -> the state the user is shown. Deliberately explicit: a frame absent
# here emits nothing, which is safer than guessing a state nobody can render.
#
# VADUserStartedSpeakingFrame is NOT a subclass of UserStartedSpeakingFrame -- both
# are bare SystemFrames in Pipecat 1.6. `VADProcessor` emits only the VAD pair, and
# the User pair comes from the LLM context aggregators, which this pipeline does not
# use (see speech/tutor_stage.py). Mapping only the User pair would leave G1a
# reporting nothing for the listening and transcribing phases -- the instrument
# would pass while measuring an empty timeline.
_STATE_FOR: dict[type[Frame], TurnState] = {
    UserStartedSpeakingFrame: TurnState.LISTENING,
    VADUserStartedSpeakingFrame: TurnState.LISTENING,
    UserStoppedSpeakingFrame: TurnState.TRANSCRIBING,
    VADUserStoppedSpeakingFrame: TurnState.TRANSCRIBING,
    # Interim transcripts are the G1a workhorse: they break the long silent
    # stretch between endpoint and first audio into visible updates.
    InterimTranscriptionFrame: TurnState.TRANSCRIBING,
    TranscriptionFrame: TurnState.THINKING,
    LLMFullResponseStartFrame: TurnState.THINKING,
    TTSStartedFrame: TurnState.SPEAKING,
    TTSAudioRawFrame: TurnState.SPEAKING,
    FillerAudioFrame: TurnState.SPEAKING,
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
        spans: Spans | None = None,
    ) -> None:
        super().__init__()
        self.timeline = timeline if timeline is not None else TurnTimeline(_clock=clock)
        # Shared `spans` and `timeline` let several probes act as one instrument at
        # different points in the chain. That is not an optimisation: a single tap at
        # the end of the pipeline sees a TranscriptionFrame only after it has passed
        # through the stage that blocks on generation, so ASR is charged for the
        # LLM's time and every later stage measures as instantaneous. Stage
        # attribution requires a tap between the stages being attributed.
        self.spans = spans if spans is not None else Spans()
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

        if isinstance(frame, UserStoppedSpeakingFrame | VADUserStoppedSpeakingFrame):
            self.spans.mark("user_stopped", now)
        elif isinstance(frame, TranscriptionFrame):
            self.spans.mark("transcript", now)
        # Three frame types, because the probe sits after TTS and TTSService
        # consumes LLMTextFrame. It emits AggregatedTextFrame when a sentence is
        # ready to synthesise and TTSTextFrame alongside the audio. Marking only
        # LLMTextFrame left llm_ttft_s permanently None in the real pipeline while
        # every unit test passed; marking only TTSTextFrame made tts_s negative,
        # because the audio for a sentence is pushed before its text frame.
        elif isinstance(frame, LLMTextFrame | AggregatedTextFrame | TTSTextFrame):
            self.spans.mark("first_token", now)
        # Audio frames only. TTSStartedFrame is emitted *before* synthesis begins --
        # counting it here made tts_s read 15 ms against VOICEVOX's real ~0.5 s, and
        # so under-reported voice-to-first-audio by half a second. It still maps to
        # the SPEAKING state below, which is a claim about feedback, not about audio.
        #
        # FillerAudioFrame is excluded. A filled pause is real feedback and counts
        # for G1a via the state below, but counting it here would make G1b measure
        # how fast the tutor can say 「ええと」 -- a number that improves while the
        # product does not.
        elif isinstance(frame, FillerAudioFrame):
            pass  # feedback for G1a via the state below, never a G1b measurement
        elif isinstance(frame, TTSAudioRawFrame | BotStartedSpeakingFrame):
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

        # Credit audio AFTER the state block, not before: the first audio frame of a
        # turn is also what emits SPEAKING, and crediting first would attribute its
        # duration to whatever state preceded it -- or to nothing at all on an empty
        # timeline, which is how this was first written and why audio_s read 0.
        if isinstance(frame, TTSAudioRawFrame):
            self.timeline.add_audio(len(frame.audio) / 2 / (frame.sample_rate or 16_000))

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
