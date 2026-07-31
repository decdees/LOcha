"""The client wire format (T2.1).

ARCHITECTURE §2.3 chose a WebSocket transport because the client is native and
already holds PCM. That choice makes the serializer ours, and the lazy option is
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
from typing import Any

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

SAMPLE_RATE = 16_000
CHANNELS = 1


class OchaSerializer(FrameSerializer):
    """PCM on binary frames, JSON on text frames. See the module docstring."""

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
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            # Barge-in: the client must drop whatever it has queued for playback,
            # otherwise the tutor keeps talking over the user for the length of
            # its buffer.
            return json.dumps({"type": "interrupt"}, ensure_ascii=False)
        if isinstance(frame, OutputTransportMessageFrame | OutputTransportMessageUrgentFrame):
            # Everything non-audio arrives here. See `ClientText` below for why
            # transcripts and replies are transport messages rather than the text
            # frames they start life as.
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
        # The client sends no control messages yet. Text arriving here is a
        # protocol change that has not been designed, so it is dropped rather
        # than guessed at.
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
