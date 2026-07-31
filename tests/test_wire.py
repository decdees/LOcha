"""T2.1 — the client wire format.

These assertions are the client's contract. The iOS app is a separate codebase
that cannot be typechecked against this one, so a silent change here shows up on
the phone as audio that is silent, sped up, or state that never updates. Every
field the app reads is pinned by a test.
"""

from __future__ import annotations

import json

import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.tests.utils import run_test

from ocha.speech.wire import (
    CHANNELS,
    SAMPLE_RATE,
    ClientText,
    OchaSerializer,
    _client_message,
    state_message,
)


@pytest.fixture
def ser() -> OchaSerializer:
    return OchaSerializer()


async def test_binary_in_is_pcm_at_the_pinned_rate(ser: OchaSerializer) -> None:
    frame = await ser.deserialize(b"\x01\x02" * 160)
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == SAMPLE_RATE == 16_000
    assert frame.num_channels == CHANNELS == 1
    assert frame.audio == b"\x01\x02" * 160


async def test_audio_out_is_bare_pcm(ser: OchaSerializer) -> None:
    """No header, no envelope. The client reads bytes straight into a buffer."""
    pcm = OutputAudioRawFrame(audio=b"\xff\x00", sample_rate=16_000, num_channels=1)
    out = await ser.serialize(pcm)
    assert out == b"\xff\x00"


def test_transcript_carries_the_final_flag() -> None:
    """Built by `ClientText`, not the serializer -- see the next test for why."""
    assert _client_message(InterimTranscriptionFrame("こんに", "u", "t")) == {
        "type": "transcript",
        "text": "こんに",
        "final": False,
    }
    assert _client_message(TranscriptionFrame("こんにちは", "u", "t")) == {
        "type": "transcript",
        "text": "こんにちは",
        "final": True,
    }
    assert _client_message(LLMTextFrame("今日は")) == {"type": "reply", "text": "今日は"}


async def test_text_frames_reach_the_client_as_transport_messages() -> None:
    """The regression test for a bug only an end-to-end run could find.

    Pipecat's output transport serialises audio and transport messages and nothing
    else, so a TranscriptionFrame arriving at `transport.output()` is silently not
    sent. The pipeline worked, the tutor spoke, and the client received no
    transcript at all. `ClientText` converts them; the serializer no longer has a
    branch for text frames, because that branch was dead code that looked alive.
    """
    down, _ = await run_test(
        ClientText(),
        frames_to_send=[TranscriptionFrame("こんにちは", "u", "t"), LLMTextFrame("はい。")],
    )
    messages = [f.message for f in down if isinstance(f, OutputTransportMessageUrgentFrame)]
    assert {m["type"] for m in messages} == {"transcript", "reply"}


async def test_japanese_is_not_escaped(ser: OchaSerializer) -> None:
    """`ensure_ascii=False`, so the payload is readable in a packet dump.

    Not cosmetic: every string on this wire is Japanese, and \\u-escaping
    triples the size of the text channel for no benefit.
    """
    raw = await ser.serialize(
        OutputTransportMessageUrgentFrame(message={"type": "reply", "text": "今日は"})
    )
    assert isinstance(raw, str) and "今日は" in raw


async def test_interruption_reaches_the_client(ser: OchaSerializer) -> None:
    """Barge-in is useless if the client keeps playing its queue."""
    assert json.loads(await ser.serialize(InterruptionFrame()) or "") == {"type": "interrupt"}


async def test_transport_messages_pass_through_verbatim(ser: OchaSerializer) -> None:
    msg = state_message("thinking")
    out = await ser.serialize(OutputTransportMessageUrgentFrame(message=msg))
    assert json.loads(out or "") == {"type": "state", "state": "thinking"}


async def test_client_text_is_dropped_not_guessed(ser: OchaSerializer) -> None:
    assert await ser.deserialize('{"type": "hello"}') is None


async def test_setup_rejects_a_rate_mismatch(ser: OchaSerializer) -> None:
    """A resample would be inaudible here and show up as an ASR accuracy bug."""
    with pytest.raises(ValueError, match="16000"):
        await ser.setup(_start(48_000))
    await ser.setup(_start(SAMPLE_RATE))  # the matching case must not raise


def _start(rate: int) -> StartFrame:
    return StartFrame(audio_in_sample_rate=rate, audio_out_sample_rate=rate)
