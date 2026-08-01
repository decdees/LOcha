"""Exchange identity, sequencing, and client-reported playback telemetry."""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ocha.speech.filler import FillerAudioFrame
from ocha.speech.repair import RepairAudioFrame

ClientMetricEvent = Literal["playback_start", "playback_duration", "cancellation"]


@dataclass
class ClientMetricFrame(Frame):
    exchange_id: uuid.UUID
    event: ClientMetricEvent
    sequence: int
    client_time_ms: float
    duration_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ClientMetric:
    event: ClientMetricEvent
    sequence: int
    client_time_ms: float
    duration_ms: float | None


@dataclass(frozen=True, slots=True)
class AudioAttribution:
    exchange_id: uuid.UUID
    sequence: int
    kind: int


@dataclass(slots=True)
class ExchangeTelemetry:
    exchange_id: uuid.UUID
    endpoint_server_ns: int | None = None
    events: int = 0
    client_metrics: list[ClientMetric] = field(default_factory=list)
    cancelled: bool = False
    finalized: bool = False


class AttributionState:
    """One mutable coordinator; telemetry records themselves are per exchange."""

    def __init__(self) -> None:
        self.current: ExchangeTelemetry | None = None
        self.records: list[ExchangeTelemetry] = []
        self._sequence = 0
        self.cancelled_exchange: uuid.UUID | None = None
        # Pipecat's WebSocket output transport recreates audio frames to apply
        # its configured sample rate. That necessarily drops frame metadata, so
        # the final processor hands attribution to the serializer in wire order.
        self._pending_audio: deque[AudioAttribution] = deque()

    def start(self) -> ExchangeTelemetry:
        self.finalize()
        self.current = ExchangeTelemetry(exchange_id=uuid.uuid4())
        self._sequence = 0
        return self.current

    def ensure(self) -> ExchangeTelemetry:
        return self.current if self.current is not None else self.start()

    def endpoint(self) -> None:
        self.ensure().endpoint_server_ns = time.monotonic_ns()

    def next_sequence(self) -> int:
        exchange = self.ensure()
        value = self._sequence
        self._sequence += 1
        exchange.events += 1
        return value

    def enqueue_audio(self, exchange_id: uuid.UUID, sequence: int, kind: int) -> None:
        self._pending_audio.append(AudioAttribution(exchange_id, sequence, kind))

    def pop_audio(self) -> AudioAttribution | None:
        return self._pending_audio.popleft() if self._pending_audio else None

    def cancel(self) -> None:
        if self.current is not None:
            self.current.cancelled = True
            self.cancelled_exchange = self.current.exchange_id

    def record_metric(self, frame: ClientMetricFrame) -> None:
        target = next(
            (
                record
                for record in [*self.records, self.current]
                if record and record.exchange_id == frame.exchange_id
            ),
            None,
        )
        if target is not None:
            target.client_metrics.append(
                ClientMetric(frame.event, frame.sequence, frame.client_time_ms, frame.duration_ms)
            )

    def finalize(self) -> None:
        if self.current is not None and not self.current.finalized:
            self.current.finalized = True
            self.records.append(self.current)
        self.current = None


class AttributionInputProcessor(FrameProcessor):
    def __init__(self, state: AttributionState) -> None:
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, ClientMetricFrame):
            self._state.record_metric(frame)
        await self.push_frame(frame, direction)


class ExchangeEndpointProcessor(FrameProcessor):
    def __init__(self, state: AttributionState) -> None:
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._state.cancel()
            self._state.start()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._state.endpoint()
        await self.push_frame(frame, direction)


class OutputAttributionProcessor(FrameProcessor):
    """Attach one exchange id and monotonic sequence to every outbound event."""

    def __init__(self, state: AttributionState) -> None:
        super().__init__()
        self._state = state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        exchange = self._state.ensure()
        exchange_id = exchange.exchange_id
        if isinstance(frame, InterruptionFrame) and self._state.cancelled_exchange is not None:
            exchange_id = self._state.cancelled_exchange
            self._state.cancelled_exchange = None

        if isinstance(frame, OutputTransportMessageFrame | OutputTransportMessageUrgentFrame):
            frame.message = {
                **frame.message,
                "exchange_id": str(exchange_id),
                "seq": self._state.next_sequence(),
            }
        elif isinstance(frame, OutputAudioRawFrame | InterruptionFrame):
            frame.metadata["ocha.exchange_id"] = str(exchange_id)
            sequence = self._state.next_sequence()
            frame.metadata["ocha.seq"] = sequence
            if isinstance(frame, FillerAudioFrame):
                kind = 1
            elif isinstance(frame, RepairAudioFrame):
                kind = 3
            elif isinstance(frame, OutputAudioRawFrame):
                kind = 2
            else:
                kind = None
            if kind is not None:
                frame.metadata["ocha.audio_kind"] = kind
                self._state.enqueue_audio(exchange_id, sequence, kind)
        await self.push_frame(frame, direction)
