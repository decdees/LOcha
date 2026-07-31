"""T2.8 — filled pauses.

The behaviour that matters is not "audio comes out". It is that the audio counts
as feedback for G1a and **does not** count toward G1b, that it stops when the user
speaks, and that it does not become a tic. Each of those is a way this feature
could quietly turn into latency papering, which is the thing it is not.
"""

from __future__ import annotations

import asyncio

import pytest
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test

from ocha.speech.filler import (
    FILLERS,
    FillerAudioFrame,
    FillerBank,
    FillerProcessor,
    FillerState,
)
from ocha.speech.probe import TurnStateProbe
from ocha.turnstate import TurnState

RATE = 16_000


def bank(seconds: float = 0.2) -> FillerBank:
    """A bank of short clips. Real ones are 0.6-1.0 s; tests do not wait that long."""
    clip = b"\x01\x02" * int(RATE * seconds)
    return FillerBank({text: clip for text in FILLERS}, RATE)


def test_the_bank_refuses_to_be_empty() -> None:
    """A silently empty bank would look like a working feature that never fires."""
    with pytest.raises(ValueError, match="empty filler bank"):
        FillerBank({}, RATE)


def test_yes_is_not_a_filler() -> None:
    """はい means "yes". Using it as a thinking noise would teach the learner to
    agree with things they have not understood."""
    assert "はい。" not in FILLERS
    assert len(FILLERS) >= 5, "too few to rotate without becoming a tic"


async def test_a_filler_fires_at_the_vad_endpoint() -> None:
    """At the endpoint, not at generation start -- the endpoint is where the silence
    begins, and it has to cover transcription as well as thinking.

    Both halves are wired, in the order the pipeline wires them: the emitter must
    exist before the trigger can fire through it.
    """
    state = FillerState()
    b = bank()
    emitter = FillerProcessor(b, state, emit=True)
    trigger = FillerProcessor(b, state)
    down, _ = await run_test(
        Pipeline([trigger, emitter]), frames_to_send=[VADUserStoppedSpeakingFrame()]
    )
    assert any(isinstance(f, FillerAudioFrame) for f in down), "nothing was said"
    state.cancel()


async def test_the_trigger_alone_says_nothing() -> None:
    """A trigger with no emitter registered must fail quietly, not crash a turn."""
    state = FillerState()
    trigger = FillerProcessor(bank(), state)
    down, _ = await run_test(trigger, frames_to_send=[VADUserStoppedSpeakingFrame()])
    assert not any(isinstance(f, FillerAudioFrame) for f in down)


async def test_barge_in_stops_it() -> None:
    """The tutor must stop thinking out loud when the user starts talking."""
    state = FillerState()
    b = bank(seconds=2.0)
    emitter = FillerProcessor(b, state, emit=True)
    trigger = FillerProcessor(b, state)
    await run_test(
        Pipeline([trigger, emitter]),
        frames_to_send=[VADUserStoppedSpeakingFrame(), VADUserStartedSpeakingFrame()],
    )
    assert state.task is None, "the filler task outlived the barge-in"


async def test_real_audio_stops_the_follow_up() -> None:
    """The observing tap cancels the second filler once the tutor really speaks."""
    state = FillerState()
    observer = FillerProcessor(bank(), state, emit=True)
    state.task = asyncio.create_task(asyncio.sleep(10))
    await run_test(
        observer,
        frames_to_send=[
            TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=RATE, num_channels=1)
        ],
    )
    assert state.real_audio is True
    assert state.task is None


async def test_filler_audio_does_not_stop_the_follow_up() -> None:
    """Its own audio must not be mistaken for the tutor having started talking."""
    state = FillerState()
    observer = FillerProcessor(bank(), state, emit=True)
    await run_test(
        observer,
        frames_to_send=[
            FillerAudioFrame(audio=b"\x00\x00" * 160, sample_rate=RATE, num_channels=1)
        ],
    )
    assert state.real_audio is False


def test_rotation_never_repeats_back_to_back() -> None:
    """A tic is a tic whether it is one phrase or a predictable cycle."""
    b = bank()
    state = FillerState()
    picks = [b.pick(state)[0] for _ in range(40)]
    assert not any(a == b_ for a, b_ in zip(picks, picks[1:], strict=False)), picks
    assert len(set(picks)) == len(FILLERS), "some filler is never used"


# ---- the part that keeps this honest -------------------------------------


async def test_filler_audio_is_feedback_for_g1a() -> None:
    """It genuinely is heard, so it genuinely counts as feedback."""
    probe = TurnStateProbe()
    await run_test(
        probe,
        frames_to_send=[
            FillerAudioFrame(audio=b"\x01\x02" * 1600, sample_rate=RATE, num_channels=1)
        ],
    )
    assert [c.state for c in probe.timeline.changes] == [TurnState.SPEAKING]
    assert probe.timeline.changes[0].audio_s == pytest.approx(0.1)


async def test_filler_audio_is_not_counted_toward_g1b() -> None:
    """The one that stops this becoming latency papering.

    If a filled pause counted as first audio, voice-to-first-audio would measure
    how fast the tutor can say 「ええと」 -- a number that improves while the product
    does not move.
    """
    probe = TurnStateProbe()
    await run_test(
        probe,
        frames_to_send=[
            VADUserStoppedSpeakingFrame(),
            FillerAudioFrame(audio=b"\x01\x02" * 1600, sample_rate=RATE, num_channels=1),
        ],
    )
    assert probe.report()["voice_to_first_audio_s"] is None, "a filler was counted as the reply"


async def test_real_audio_after_a_filler_is_counted() -> None:
    """...and real audio still is, so the metric is not simply broken."""
    probe = TurnStateProbe()
    await run_test(
        probe,
        frames_to_send=[
            VADUserStoppedSpeakingFrame(),
            FillerAudioFrame(audio=b"\x01\x02" * 1600, sample_rate=RATE, num_channels=1),
            TTSAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=RATE, num_channels=1),
        ],
    )
    assert probe.report()["voice_to_first_audio_s"] is not None
