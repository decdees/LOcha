"""Known-answer validation for the v2 voice instrument."""

from __future__ import annotations

import uuid

import pytest

from ocha.measurement import (
    AudioInterval,
    ExchangeCapture,
    InstrumentFailure,
    VisibleChange,
    gate_summary,
    measure_exchange,
    nearest_rank,
)
from ocha.speech.wire import AudioKind

X = uuid.UUID("00000000-0000-0000-0000-000000000001")
Y = uuid.UUID("00000000-0000-0000-0000-000000000002")


def capture(
    *, visible: tuple[VisibleChange, ...] = (), audio: tuple[AudioInterval, ...]
) -> ExchangeCapture:
    return ExchangeCapture(X, 1000, visible, audio, "accepted")


def clip(
    seq: int, kind: AudioKind, start: float, duration: float, **kwargs: float
) -> AudioInterval:
    return AudioInterval(X, seq, kind, start, duration, kwargs.get("cancelled_at_ms"))


def test_no_events_means_the_full_interval_is_silent() -> None:
    result = measure_exchange(capture(audio=(clip(0, AudioKind.TUTOR, 2200, 100),)))
    assert result.g1b_ms == 1200
    assert result.longest_uncovered_ms == 1200


def test_audio_duration_union_and_tail_gap_are_exact() -> None:
    result = measure_exchange(
        capture(
            visible=(VisibleChange(X, 0, 1200),),
            audio=(
                clip(1, AudioKind.FILLER, 1300, 300),
                clip(2, AudioKind.FILLER, 1500, 200),
                clip(3, AudioKind.TUTOR, 2000, 100),
            ),
        )
    )
    assert result.longest_uncovered_ms == 300  # 1700 -> 2000 tail


def test_cancellation_truncates_coverage() -> None:
    result = measure_exchange(
        capture(
            audio=(
                clip(0, AudioKind.FILLER, 1100, 700, cancelled_at_ms=1300),
                clip(1, AudioKind.TUTOR, 1800, 100),
            )
        )
    )
    assert result.longest_uncovered_ms == 500


def test_sequential_audio_clips_form_one_coverage_interval() -> None:
    result = measure_exchange(
        capture(
            audio=(
                clip(0, AudioKind.FILLER, 1100, 200),
                clip(1, AudioKind.REPAIR, 1300, 200),
                clip(2, AudioKind.TUTOR, 1700, 100),
            )
        )
    )
    assert result.longest_uncovered_ms == 200


def test_reordered_arrival_is_sorted_by_scheduled_time() -> None:
    result = measure_exchange(
        capture(
            visible=(VisibleChange(X, 9, 1250), VisibleChange(X, 1, 1100)),
            audio=(clip(8, AudioKind.TUTOR, 1500, 100), clip(4, AudioKind.FILLER, 1200, 200)),
        )
    )
    assert result.longest_uncovered_ms == 100


@pytest.mark.parametrize(
    "bad",
    [
        capture(audio=(clip(0, AudioKind.TUTOR, 900, 100),)),
        capture(audio=(clip(0, AudioKind.FILLER, 1100, 100),)),
        capture(audio=(clip(0, AudioKind.TUTOR, 1200, -1),)),
        capture(audio=(clip(0, AudioKind.TUTOR, float("nan"), 100),)),
        capture(
            visible=(VisibleChange(X, 0, 1100),),
            audio=(clip(0, AudioKind.TUTOR, 1200, 100),),
        ),
        capture(audio=(AudioInterval(Y, 0, AudioKind.TUTOR, 1200, 100),)),
    ],
)
def test_invalid_attribution_is_an_instrument_failure(bad: ExchangeCapture) -> None:
    with pytest.raises(InstrumentFailure):
        measure_exchange(bad)


def test_nearest_rank_p95_on_fifty_turns() -> None:
    values = [float(n) for n in range(1, 51)]
    assert nearest_rank(values, 0.95) == 48


def test_gate_requires_all_fifty_unambiguous_turns() -> None:
    one = capture(audio=(clip(0, AudioKind.TUTOR, 1200, 100),))
    assert gate_summary([one])["eligible"] is False
    fifty = [
        ExchangeCapture(
            uuid.uuid4(),
            1000,
            (),
            (AudioInterval(uuid.uuid4(), 0, AudioKind.TUTOR, 1200, 100),),
            "accepted",
        )
        for _ in range(50)
    ]
    # Deliberately mismatched audio ids: no post-selection; one attribution fault fails the gate.
    assert gate_summary(fifty)["eligible"] is False


def test_gate_rejects_duplicate_exchange_ids() -> None:
    repeated = capture(audio=(clip(0, AudioKind.TUTOR, 1200, 100),))
    summary = gate_summary([repeated] * 50)
    assert summary == {"eligible": False, "reason": "duplicate exchange id"}
