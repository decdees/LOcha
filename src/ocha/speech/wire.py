"""The client wire format (T2.1).

ARCHITECTURE §2.3 chose a WebSocket transport because the PWA captures and plays
PCM directly. That choice makes the serializer ours, and the lazy option is
the WebSocket protocol's own frame types:

- **binary frame = raw PCM**, 16 kHz mono signed 16-bit little-endian
- **text frame = one JSON object**, `{"type": ..., ...}`

Pipecat ships `ProtobufFrameSerializer`, which was the obvious reuse. It is not
taken here: it would put SwiftProtobuf plus a generated mirror of `frames_pb2`
on the client to encode a format both ends of which we own. The transport
already tells us whether a binary or a text frame arrived, so the discriminator
protobuf would carry is free.

Sample rate is NOT on the wire. Both ends are pinned to `SAMPLE_RATE` and a
mismatch is a bug to fix, not a case to negotiate -- whisper wants 16 kHz and
nothing here has a reason to send anything else.

Outbound JSON types, which are the client's entire non-audio contract:

| `type` | payload | why the client cares |
|---|---|---|
| `state` | `state`: a `TurnState` value | G1a -- what to show during the silence |
| `transcript` | `text`, `final` | the user's own words, appearing as they resolve |
| `reply` | `text` | tutor text, accumulated alongside the audio |
| `grammar` | the FR-5 payload | rendered in the grammar panel, never spoken |
"""

from __future__ import annotations

import json
import struct
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer

if TYPE_CHECKING:
    from ocha.speech.attribution import AudioAttribution


class AudioAttributionSource(Protocol):
    def pop_audio(self) -> AudioAttribution | None: ...


SAMPLE_RATE = 16_000
CHANNELS = 1
AUDIO_MAGIC = b"OCH1"
AUDIO_VERSION = 1
AUDIO_HEADER = struct.Struct(">4sBBI16s")


class AudioKind(IntEnum):
    FILLER = 1
    TUTOR = 2
    REPAIR = 3


def pack_audio(exchange_id: uuid.UUID, sequence: int, kind: AudioKind, pcm: bytes) -> bytes:
    return AUDIO_HEADER.pack(AUDIO_MAGIC, AUDIO_VERSION, kind, sequence, exchange_id.bytes) + pcm


def unpack_audio(data: bytes) -> tuple[uuid.UUID, int, AudioKind, bytes]:
    if len(data) < AUDIO_HEADER.size:
        raise ValueError("audio frame is shorter than the OCH1 header")
    magic, version, kind, sequence, raw_uuid = AUDIO_HEADER.unpack_from(data)
    if magic != AUDIO_MAGIC or version != AUDIO_VERSION:
        raise ValueError("unsupported audio frame header")
    return uuid.UUID(bytes=raw_uuid), sequence, AudioKind(kind), data[AUDIO_HEADER.size :]


class OchaSerializer(FrameSerializer):
    """PCM on binary frames, JSON on text frames. See the module docstring."""

    def __init__(self, attribution: AudioAttributionSource | None = None) -> None:
        super().__init__()
        self._attribution = attribution

    # The override ignore below is upstream's problem, not ours: BaseObject.setup takes
    # a BaseTaskManager and FrameSerializer.setup narrows it to a StartFrame, so any
    # serializer with a correctly-typed setup violates Liskov against BaseObject.
    # Matching FrameSerializer is the only useful choice; the alternative is not
    # validating the sample rate at all.
    async def setup(self, frame: StartFrame) -> None:  # type: ignore[override]
        # The transport's negotiated rate must match what the client sends. If it
        # ever does not, fail here rather than shipping resampled-sounding audio
        # and debugging it as an ASR accuracy problem three tasks later.
        if frame.audio_in_sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"transport audio_in_sample_rate={frame.audio_in_sample_rate}, "
                f"wire format is {SAMPLE_RATE}"
            )

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            pending = self._attribution.pop_audio() if self._attribution is not None else None
            if pending is not None:
                exchange_id = pending.exchange_id
                sequence = pending.sequence
                kind = AudioKind(pending.kind)
            else:
                try:
                    exchange_id = uuid.UUID(str(frame.metadata["ocha.exchange_id"]))
                    sequence = int(frame.metadata["ocha.seq"])
                    kind = AudioKind(int(frame.metadata["ocha.audio_kind"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("outbound audio has no valid exchange attribution") from exc
            return pack_audio(exchange_id, sequence, kind, frame.audio)
        if isinstance(frame, InterruptionFrame):
            # Barge-in: the client must drop whatever it has queued for playback,
            # otherwise the tutor keeps talking over the user for the length of
            # its buffer.
            try:
                interrupt_exchange_id = str(uuid.UUID(str(frame.metadata["ocha.exchange_id"])))
                sequence = int(frame.metadata["ocha.seq"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("interrupt has no valid exchange attribution") from exc
            return json.dumps(
                {"type": "interrupt", "exchange_id": interrupt_exchange_id, "seq": sequence},
                ensure_ascii=False,
            )
        if isinstance(frame, OutputTransportMessageFrame | OutputTransportMessageUrgentFrame):
            # Everything non-audio arrives here. See `ClientText` below for why
            # transcripts and replies are transport messages rather than the text
            # frames they start life as.
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
        try:
            message = json.loads(data)
            if message.get("type") != "client_metric":
                return None
            from ocha.speech.attribution import ClientMetricFrame

            return ClientMetricFrame(
                exchange_id=uuid.UUID(str(message["exchange_id"])),
                event=message["event"],
                sequence=int(message["seq"]),
                client_time_ms=float(message["client_time_ms"]),
                duration_ms=(
                    float(message["duration_ms"])
                    if message.get("duration_ms") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None


def state_message(state: str) -> dict[str, Any]:
    return {"type": "state", "state": state}


class ClientText(FrameProcessor):
    """Turns transcripts and tutor text into transport messages.

    **Pipecat's output transport only serialises audio and transport messages.**
    A `TranscriptionFrame` or `LLMTextFrame` reaching `transport.output()` is
    simply not sent, so a serializer branch for them is dead code that looks
    alive. Found end to end: the pipeline was working, the tutor was speaking, and
    the client never received a single transcript.

    Place immediately before `transport.output()` -- after TTS, so the text has
    already been synthesised, and after anything that reads it.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        message = _client_message(frame)
        if message is not None:
            await self.push_frame(
                OutputTransportMessageUrgentFrame(message=message), FrameDirection.DOWNSTREAM
            )
        await self.push_frame(frame, direction)


def _client_message(frame: Frame) -> dict[str, Any] | None:
    """The client-visible payload for a frame, or None if it has none."""
    if isinstance(frame, InterimTranscriptionFrame):
        return {"type": "transcript", "text": frame.text, "final": False}
    if isinstance(frame, TranscriptionFrame):
        return {"type": "transcript", "text": frame.text, "final": True}
    if isinstance(frame, LLMTextFrame):
        # Generation ends with an empty chunk; forwarding it sends the client a
        # `reply` with nothing in it, which is a no-op it has to know to ignore.
        return {"type": "reply", "text": frame.text} if frame.text else None
    return None
