"""Version-2 exchange-attributed voice measurement.

All timestamps in one record are client-clock milliseconds. Server wall clocks
are intentionally absent: unsynchronised absolute clocks are not comparable.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ocha.speech.wire import AudioKind


class InstrumentFailure(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VisibleChange:
    exchange_id: uuid.UUID
    sequence: int
    at_ms: float


@dataclass(frozen=True, slots=True)
class AudioInterval:
    exchange_id: uuid.UUID
    sequence: int
    kind: AudioKind
    start_ms: float
    duration_ms: float
    cancelled_at_ms: float | None = None

    @property
    def end_ms(self) -> float:
        end = self.start_ms + self.duration_ms
        return min(end, self.cancelled_at_ms) if self.cancelled_at_ms is not None else end


@dataclass(frozen=True, slots=True)
class ExchangeCapture:
    exchange_id: uuid.UUID
    speech_end_ms: float
    visible: tuple[VisibleChange, ...]
    audio: tuple[AudioInterval, ...]
    asr: Literal["accepted", "rejected"]


@dataclass(frozen=True, slots=True)
class ExchangeMetrics:
    exchange_id: uuid.UUID
    g1b_ms: float
    longest_uncovered_ms: float
    asr: Literal["accepted", "rejected"]


def measure_exchange(capture: ExchangeCapture) -> ExchangeMetrics:
    scalar_values = [
        capture.speech_end_ms,
        *(event.at_ms for event in capture.visible),
        *(value for clip in capture.audio for value in (clip.start_ms, clip.duration_ms)),
        *(clip.cancelled_at_ms for clip in capture.audio if clip.cancelled_at_ms is not None),
    ]
    if any(not math.isfinite(value) for value in scalar_values):
        raise InstrumentFailure("non-finite timestamp or duration")
    sequences = [event.sequence for event in capture.visible] + [
        clip.sequence for clip in capture.audio
    ]
    if len(sequences) != len(set(sequences)):
        raise InstrumentFailure("duplicate sequence in exchange")
    if any(event.exchange_id != capture.exchange_id for event in capture.visible) or any(
        clip.exchange_id != capture.exchange_id for clip in capture.audio
    ):
        raise InstrumentFailure("cross-turn event or audio")
    if any(clip.duration_ms < 0 or clip.end_ms < clip.start_ms for clip in capture.audio):
        raise InstrumentFailure("negative audio duration")

    tutors = sorted(
        (clip for clip in capture.audio if clip.kind is AudioKind.TUTOR),
        key=lambda clip: clip.start_ms,
    )
    if not tutors:
        raise InstrumentFailure("missing tutor audio")
    first_tutor = tutors[0].start_ms
    latency = first_tutor - capture.speech_end_ms
    if latency < 0:
        raise InstrumentFailure("negative voice-to-tutor latency")

    points = sorted(
        event.at_ms
        for event in capture.visible
        if capture.speech_end_ms <= event.at_ms <= first_tutor
    )
    intervals = _union(
        (max(capture.speech_end_ms, clip.start_ms), min(first_tutor, clip.end_ms))
        for clip in capture.audio
        if clip.kind is not AudioKind.TUTOR
        and clip.end_ms > capture.speech_end_ms
        and clip.start_ms < first_tutor
    )
    boundaries = [(point, point) for point in points] + intervals
    boundaries.sort()
    cursor = capture.speech_end_ms
    longest = 0.0
    for start, end in boundaries:
        longest = max(longest, start - cursor)
        cursor = max(cursor, end)
    longest = max(longest, first_tutor - cursor)
    return ExchangeMetrics(capture.exchange_id, latency, longest, capture.asr)


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end >= start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise InstrumentFailure("percentile of an empty sample")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def gate_summary(captures: list[ExchangeCapture]) -> dict[str, object]:
    if len(captures) != 50:
        return {"eligible": False, "reason": "exactly 50 attributable iPhone turns required"}
    exchange_ids = [capture.exchange_id for capture in captures]
    if len(exchange_ids) != len(set(exchange_ids)):
        return {"eligible": False, "reason": "duplicate exchange id"}
    try:
        metrics = [measure_exchange(capture) for capture in captures]
    except InstrumentFailure as exc:
        return {"eligible": False, "reason": str(exc)}
    latencies = [metric.g1b_ms for metric in metrics]
    gaps = [metric.longest_uncovered_ms for metric in metrics]
    p50 = nearest_rank(latencies, 0.50)
    p95 = nearest_rank(latencies, 0.95)
    return {
        "eligible": True,
        "turns": 50,
        "asr_rejected": sum(metric.asr == "rejected" for metric in metrics),
        "g1a_met": all(gap <= 500 for gap in gaps),
        "g1b_met": p50 <= 3200 and p95 <= 4600,
        "p50_ms": p50,
        "p95_ms": p95,
    }
