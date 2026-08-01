"""T2.1 — the client wire format.

These assertions are the client's contract. The PWA is a separate codebase
that cannot be typechecked against this one, so a silent change here shows up on
the phone as audio that is silent, sped up, or state that never updates. Every
field the app reads is pinned by a test.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import cast

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

from ocha.speech.attribution import AttributionState
from ocha.speech.wire import (
    AUDIO_HEADER,
    CHANNELS,
    SAMPLE_RATE,
    AudioKind,
    ClientText,
    LessonActionFrame,
    OchaSerializer,
    _client_message,
    state_message,
    unpack_audio,
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


async def test_audio_out_has_a_fixed_attribution_header(ser: OchaSerializer) -> None:
    pcm = OutputAudioRawFrame(audio=b"\xff\x00", sample_rate=16_000, num_channels=1)
    exchange_id = uuid.uuid4()
    pcm.metadata.update({"ocha.exchange_id": str(exchange_id), "ocha.seq": 7, "ocha.audio_kind": 2})
    out = await ser.serialize(pcm)
    assert isinstance(out, bytes)
    assert len(out) == AUDIO_HEADER.size + 2
    assert unpack_audio(out) == (exchange_id, 7, AudioKind.TUTOR, b"\xff\x00")


async def test_unattributed_audio_is_an_instrument_failure(ser: OchaSerializer) -> None:
    pcm = OutputAudioRawFrame(audio=b"\xff\x00", sample_rate=16_000, num_channels=1)
    with pytest.raises(RuntimeError, match="attribution"):
        await ser.serialize(pcm)


async def test_audio_attribution_survives_transport_frame_reconstruction() -> None:
    state = AttributionState()
    exchange_id = state.start().exchange_id
    state.enqueue_audio(exchange_id, sequence=7, kind=AudioKind.REPAIR)
    serializer = OchaSerializer(state)

    # FastAPIWebsocketOutputTransport constructs a fresh OutputAudioRawFrame,
    # discarding the source frame's metadata immediately before serialization.
    reconstructed = OutputAudioRawFrame(audio=b"\x01\x02", sample_rate=SAMPLE_RATE, num_channels=1)
    encoded = await serializer.serialize(reconstructed)

    assert isinstance(encoded, bytes)
    assert unpack_audio(encoded) == (exchange_id, 7, AudioKind.REPAIR, b"\x01\x02")


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
    frame = InterruptionFrame()
    exchange_id = uuid.uuid4()
    frame.metadata.update({"ocha.exchange_id": str(exchange_id), "ocha.seq": 9})
    assert json.loads(await ser.serialize(frame) or "") == {
        "type": "interrupt",
        "exchange_id": str(exchange_id),
        "seq": 9,
    }


async def test_transport_messages_pass_through_verbatim(ser: OchaSerializer) -> None:
    msg = state_message("thinking")
    out = await ser.serialize(OutputTransportMessageUrgentFrame(message=msg))
    assert json.loads(out or "") == {"type": "state", "state": "thinking"}


async def test_client_text_is_dropped_not_guessed(ser: OchaSerializer) -> None:
    assert await ser.deserialize('{"type": "hello"}') is None


async def test_client_playback_metric_is_typed(ser: OchaSerializer) -> None:
    from ocha.speech.attribution import ClientMetricFrame

    exchange_id = uuid.uuid4()
    frame = await ser.deserialize(
        json.dumps(
            {
                "type": "client_metric",
                "exchange_id": str(exchange_id),
                "event": "playback_duration",
                "seq": 4,
                "client_time_ms": 1200.5,
                "duration_ms": 320.0,
            }
        )
    )
    assert isinstance(frame, ClientMetricFrame)
    assert frame.exchange_id == exchange_id
    assert frame.duration_ms == 320.0


async def test_lesson_action_is_typed_and_bounded(ser: OchaSerializer) -> None:
    frame = await ser.deserialize(
        json.dumps(
            {
                "type": "lesson_action",
                "action": "replay",
                "lesson_id": "greetings",
                "step_id": "greeting-hello",
            }
        )
    )
    assert frame is not None and frame.__class__.__name__ == LessonActionFrame.__name__
    assert cast(LessonActionFrame, frame).action == "replay"
    assert (
        await ser.deserialize(
            '{"type":"lesson_action","action":"delete","lesson_id":"x","step_id":"y"}'
        )
        is None
    )


def test_pwa_buffers_playback_by_150ms() -> None:
    client = (Path(__file__).parents[1] / "web" / "index.html").read_text()
    assert "const PLAYBACK_BUFFER_SECONDS = 0.150;" in client
    assert "if (playAt <= now) playAt = now + PLAYBACK_BUFFER_SECONDS;" in client


def test_pwa_defaults_to_guided_and_never_selects_remote_english_voice() -> None:
    client = (Path(__file__).parents[1] / "web" / "index.html").read_text()
    assert "currentMode = 'guided'" in client
    assert "voice.localService &&" in client
    assert "recording && !appIsSpeaking" in client
    assert "track.enabled = false" in client
    assert "new Int16Array(RATE).buffer" in client
    assert "pendingGuidedInstruction" in client
    assert "/ws?mode=${currentMode}" in client


async def test_setup_rejects_a_rate_mismatch(ser: OchaSerializer) -> None:
    """A resample would be inaudible here and show up as an ASR accuracy bug."""
    with pytest.raises(ValueError, match="16000"):
        await ser.setup(_start(48_000))
    await ser.setup(_start(SAMPLE_RATE))  # the matching case must not raise


def _start(rate: int) -> StartFrame:
    return StartFrame(audio_in_sample_rate=rate, audio_out_sample_rate=rate)
