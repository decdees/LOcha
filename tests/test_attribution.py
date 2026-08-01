"""Every outbound event belongs unambiguously to one exchange."""

from __future__ import annotations

from pipecat.frames.frames import LLMTextFrame, OutputTransportMessageUrgentFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.tests.utils import run_test

from ocha.speech.attribution import AttributionState, OutputAttributionProcessor
from ocha.speech.repair import RepairAudioFrame
from ocha.speech.wire import ClientText


async def test_json_and_audio_share_a_monotonic_exchange_sequence() -> None:
    state = AttributionState()
    processor = OutputAttributionProcessor(state)
    down, _ = await run_test(
        Pipeline([ClientText(), processor]),
        frames_to_send=[
            LLMTextFrame("はい。"),
            TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16_000, num_channels=1),
            RepairAudioFrame(audio=b"\x00\x00", sample_rate=16_000, num_channels=1),
        ],
    )
    message = next(f for f in down if isinstance(f, OutputTransportMessageUrgentFrame))
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    exchange_id = message.message["exchange_id"]
    assert message.message["seq"] == 0
    assert [frame.metadata["ocha.seq"] for frame in audio] == [1, 2]
    assert all(frame.metadata["ocha.exchange_id"] == exchange_id for frame in audio)
    assert [frame.metadata["ocha.audio_kind"] for frame in audio] == [2, 3]


def test_finalized_telemetry_is_one_record_per_exchange() -> None:
    state = AttributionState()
    first = state.start().exchange_id
    second = state.start().exchange_id
    state.finalize()
    assert [record.exchange_id for record in state.records] == [first, second]
    assert all(record.finalized for record in state.records)
