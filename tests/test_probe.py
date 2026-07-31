"""T2.1 — pipeline instrumentation.

Built before the first real FrameProcessor so the streaming chain is assertable
from the first commit. ARCHITECTURE §5.2 names a buffering FrameProcessor as the
single biggest failure mode in the voice loop, and it is invisible until the end
unless something checks for it continuously.

Uses Pipecat's own `run_test` harness. Hand-wiring `link()` + `process_frame()`
does not work: `push_frame` short-circuits on `_check_started`, and the processor
needs `setup()` with an initialised TaskManager before a StartFrame. A hand-rolled
rig silently forwarded nothing, which looked exactly like the buffering bug this
file exists to catch.

**Pipecat SYSTEM frames bypass the queue and overtake DATA/CONTROL frames.**
UserStarted/StoppedSpeaking and BotStarted/StoppedSpeaking are SYSTEM;
Transcription, LLMText are DATA; TTSStarted, LLMFullResponseStart are CONTROL.
So global arrival order across families is NOT guaranteed -- an early version of
these tests asserted it and failed for that reason, not because the probe was
wrong. Ordering is therefore asserted within a family only.
"""

from __future__ import annotations

import pytest
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
from pipecat.tests.utils import run_test

from ocha.speech.probe import TurnStateProbe
from ocha.turnstate import TurnState


class Clock:
    """Deterministic time. Real sleeps make the suite slow and flaky."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, s: float) -> None:
        self.t += s


def transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="u", timestamp="t")


def interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text=text, user_id="u", timestamp="t")


class TickingProbe(TurnStateProbe):
    """Advances a fake clock by a per-frame-type amount as frames arrive.

    Lets a whole turn be timed deterministically inside run_test, which drives
    the frames itself and gives no hook between them.
    """

    def __init__(self, clock: Clock, delays: dict[type[Frame], float]) -> None:
        super().__init__(clock=clock)
        self._clk = clock
        self._delays = delays

    async def process_frame(self, frame: Frame, direction) -> None:  # type: ignore[no-untyped-def]
        self._clk.tick(self._delays.get(type(frame), 0.0))
        await super().process_frame(frame, direction)


# ---- the §5.2 guard: never buffer -----------------------------------------


async def test_probe_forwards_data_frames_in_order() -> None:
    """§5.2's named worst failure is a processor that buffers and breaks the
    streaming chain. DATA frames share one queue, so their relative order IS
    guaranteed -- that makes them the right family to assert a strict sequence on.
    """
    probe = TurnStateProbe(clock=Clock())
    frames: list[Frame] = [
        interim("こ"),
        interim("こん"),
        transcription("こんにちは。"),
        LLMTextFrame(text="いい"),
        LLMTextFrame(text="ですね"),
    ]
    await run_test(probe, frames_to_send=frames, expected_down_frames=[type(f) for f in frames])


async def test_probe_forwards_system_frames_too() -> None:
    """System frames overtake the queue, so only membership can be asserted --
    but they must still all arrive."""
    probe = TurnStateProbe(clock=Clock())
    frames: list[Frame] = [
        UserStartedSpeakingFrame(),
        UserStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
    ]
    down, _ = await run_test(probe, frames_to_send=frames)
    assert {type(f) for f in down} >= {type(f) for f in frames}


async def test_unknown_frame_passes_through_without_emitting_state() -> None:
    """A frame with no mapping must still be forwarded. Emitting a guessed state
    would put something on screen nobody can render."""
    probe = TurnStateProbe(clock=Clock())
    await run_test(
        probe, frames_to_send=[LLMTextFrame(text="x")], expected_down_frames=[LLMTextFrame]
    )
    assert probe.timeline.changes == []


# ---- state mapping --------------------------------------------------------


async def test_frames_map_to_the_expected_states() -> None:
    probe = TurnStateProbe(clock=Clock())
    await run_test(
        probe,
        frames_to_send=[
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            transcription("こんにちは。"),
            TTSStartedFrame(),
        ],
    )
    # Membership, not order: the two SYSTEM frames overtake the queued ones.
    assert {c.state for c in probe.timeline.changes} == {
        TurnState.LISTENING,
        TurnState.TRANSCRIBING,
        TurnState.THINKING,
        TurnState.SPEAKING,
    }


async def test_interim_transcripts_repeat_the_state_on_purpose() -> None:
    """Interim transcripts are the G1a workhorse: they break the silent stretch
    between endpoint and first audio into visible updates. Suppressing the repeat
    would defeat the mechanism."""
    probe = TurnStateProbe(clock=Clock())
    await run_test(
        probe,
        frames_to_send=[interim("こ"), interim("こん"), interim("こんに")],
        expected_down_frames=[InterimTranscriptionFrame] * 3,
    )
    assert len(probe.timeline.changes) == 3


async def test_repeated_non_interim_states_are_not_duplicated() -> None:
    """Both frames map to SPEAKING; only one transition should be recorded,
    however the two families happen to interleave."""
    probe = TurnStateProbe(clock=Clock())
    await run_test(probe, frames_to_send=[TTSStartedFrame(), BotStartedSpeakingFrame()])
    assert [c.state for c in probe.timeline.changes] == [TurnState.SPEAKING]


# ---- timing ---------------------------------------------------------------


async def test_spans_measure_the_stages_5_1_budgets() -> None:
    """Timings are T0.9's measured values: ASR 1.25 s, LLM TTFT 0.50 s, TTS 0.63 s."""
    clock = Clock()
    probe = TickingProbe(
        clock,
        {TranscriptionFrame: 1.25, LLMTextFrame: 0.50, TTSStartedFrame: 0.63},
    )
    await run_test(
        probe,
        frames_to_send=[
            UserStoppedSpeakingFrame(),
            transcription("こんにちは。"),
            LLMTextFrame(text="い"),
            TTSStartedFrame(),
        ],
        expected_down_frames=[
            UserStoppedSpeakingFrame,
            TranscriptionFrame,
            LLMTextFrame,
            TTSStartedFrame,
        ],
    )
    r = probe.report()
    assert r["asr_s"] == pytest.approx(1.25)
    assert r["llm_ttft_s"] == pytest.approx(0.50)
    assert r["tts_s"] == pytest.approx(0.63)
    assert r["voice_to_first_audio_s"] == pytest.approx(2.38)


async def test_first_occurrence_wins_for_a_repeated_mark() -> None:
    """TTS emits a started frame per sentence; first audio is the first of them.
    Uses two CONTROL frames so queue ordering is well-defined."""
    clock = Clock()
    probe = TickingProbe(clock, {TTSStartedFrame: 1.0})
    await run_test(
        probe,
        frames_to_send=[UserStoppedSpeakingFrame(), TTSStartedFrame(), TTSStartedFrame()],
    )
    # two ticks of 1.0 elapsed, but the mark is taken at the first
    assert probe.report()["voice_to_first_audio_s"] == pytest.approx(1.0)


async def test_incomplete_turn_reports_none_not_a_wrong_number() -> None:
    probe = TurnStateProbe(clock=Clock())
    await run_test(
        probe,
        frames_to_send=[UserStoppedSpeakingFrame()],
        expected_down_frames=[UserStoppedSpeakingFrame],
    )
    assert probe.report()["voice_to_first_audio_s"] is None


# ---- G1a on realistic frame sequences -------------------------------------


async def test_measured_latency_without_interim_transcripts_violates_g1a() -> None:
    """Why T2.8's feedback states are load-bearing. At the measured 1.25 s ASR
    stage with no interim transcripts, the user sees nothing at all."""
    clock = Clock()
    probe = TickingProbe(clock, {TranscriptionFrame: 1.25, TTSStartedFrame: 0.50})
    await run_test(
        probe,
        frames_to_send=[
            UserStoppedSpeakingFrame(),
            transcription("こんにちは。"),
            TTSStartedFrame(),
        ],
        expected_down_frames=[UserStoppedSpeakingFrame, TranscriptionFrame, TTSStartedFrame],
    )
    assert probe.report()["satisfies_g1a"] is False


async def test_interim_transcripts_rescue_g1a_at_the_same_total_latency() -> None:
    """Same 1.75 s total, broken up. G1a is a feedback criterion, not a speed one
    -- this is the evidence for that claim, and it is why the measured 2.53 s
    chain is acceptable."""
    clock = Clock()
    probe = TickingProbe(
        clock,
        {InterimTranscriptionFrame: 0.25, LLMFullResponseStartFrame: 0.25},
    )
    frames: list[Frame] = [UserStoppedSpeakingFrame()]
    frames += [interim("こん") for _ in range(5)]
    frames += [transcription("こんにちは。")]
    frames += [LLMFullResponseStartFrame() for _ in range(2)]
    frames += [TTSStartedFrame()]

    await run_test(probe, frames_to_send=frames, expected_down_frames=[type(f) for f in frames])

    r = probe.report()
    assert r["voice_to_first_audio_s"] == pytest.approx(1.75)
    assert r["satisfies_g1a"] is True, "feedback, not speed, is what G1a measures"


async def test_state_changes_are_pushed_to_the_client_urgently() -> None:
    """G1a is client-visible, so the probe announces the state it decides.

    Urgent, because a state message queued behind buffered audio arrives after
    the thing it describes is over. The client would show "listening" while the
    tutor is already speaking -- worse than no indicator.

    The state message precedes the frame that triggered it. That is the intended
    order and the reason the announcement happens before the forward: the label
    should be on screen when the thing it labels starts, not one frame late.
    """
    probe = TurnStateProbe(emit_state=True)
    await run_test(
        probe,
        frames_to_send=[UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()],
        expected_down_frames=[
            OutputTransportMessageUrgentFrame,
            UserStartedSpeakingFrame,
            OutputTransportMessageUrgentFrame,
            UserStoppedSpeakingFrame,
        ],
    )


async def test_no_client_messages_unless_asked() -> None:
    """The default stays a pure tap -- tests and offline runs get no extra frames."""
    probe = TurnStateProbe()
    await run_test(
        probe,
        frames_to_send=[UserStartedSpeakingFrame()],
        expected_down_frames=[UserStartedSpeakingFrame],
    )
