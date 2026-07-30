from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fsrs import Rating

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import Item, ItemScheduler, Usage, derive_rating
from ocha.scheduling.rating import ACCENT_THRESHOLD


@pytest.fixture
def sched(tmp_path: Path) -> ItemScheduler:
    conn: sqlite3.Connection = connect(tmp_path / "t.db")
    migrate(conn)
    seed(conn)
    return ItemScheduler(conn)


# ---- FR-8 derivation table ------------------------------------------------


def test_unprompted_when_due_is_good() -> None:
    assert derive_rating(Usage.UNPROMPTED, was_due=True) is Rating.Good


def test_unprompted_before_due_is_easy() -> None:
    """Produced spontaneously ahead of schedule -- the scheduler underestimated it."""
    assert derive_rating(Usage.UNPROMPTED, was_due=False) is Rating.Easy


def test_hinted_is_hard() -> None:
    assert derive_rating(Usage.HINTED) is Rating.Hard


def test_avoided_is_again() -> None:
    assert derive_rating(Usage.AVOIDED) is Rating.Again


# ---- the accent cap: PROVISIONAL, disabled by default ---------------------


def test_accent_cap_is_inert_by_default() -> None:
    """PRD FR-8 as amended: must not gate scheduling until T3.6 validates it."""
    assert derive_rating(Usage.UNPROMPTED, was_due=False, accent_score=0.0) is Rating.Easy


def test_accent_cap_caps_at_hard_when_enabled() -> None:
    got = derive_rating(
        Usage.UNPROMPTED,
        was_due=False,
        accent_score=ACCENT_THRESHOLD - 0.1,
        accent_cap_enabled=True,
    )
    assert got is Rating.Hard


def test_accent_cap_never_raises_a_rating(sched: ItemScheduler) -> None:
    """A cap only lowers. Avoided stays Again even with perfect accent."""
    got = derive_rating(Usage.AVOIDED, accent_score=1.0, accent_cap_enabled=True)
    assert got is Rating.Again


def test_accent_cap_ignores_good_accent(sched: ItemScheduler) -> None:
    got = derive_rating(Usage.UNPROMPTED, was_due=False, accent_score=0.95, accent_cap_enabled=True)
    assert got is Rating.Easy


# ---- queries --------------------------------------------------------------


def test_due_items_returns_seeded_items(sched: ItemScheduler) -> None:
    assert len(sched.due_items(limit=5)) == 5


def test_known_items_excludes_unreviewed(sched: ItemScheduler) -> None:
    assert sched.known_items(min_reps=1) == []


def test_known_items_after_reviews(sched: ItemScheduler) -> None:
    item = sched.due_items(limit=1)[0]
    for _ in range(3):
        sched.record_review(item.id, Rating.Good)
    known = sched.known_items(min_reps=3)
    assert [i.id for i in known] == [item.id]


def test_lowest_stability_excludes_unreviewed(sched: ItemScheduler) -> None:
    """An unreviewed item has stability 0 and would otherwise crowd out the
    genuinely weak ones."""
    assert sched.lowest_stability() == []
    a, b = sched.due_items(limit=2)
    sched.record_review(a.id, Rating.Easy)
    sched.record_review(b.id, Rating.Again)
    weak = sched.lowest_stability(limit=2)
    assert weak[0].id == b.id  # Again is weaker than Easy


def test_record_review_unknown_item(sched: ItemScheduler) -> None:
    with pytest.raises(KeyError):
        sched.record_review(999_999, Rating.Good)


def test_record_review_persists_and_logs(sched: ItemScheduler) -> None:
    item = sched.due_items(limit=1)[0]
    after = sched.record_review(item.id, Rating.Good, source="turn:1")
    assert after.reps == 1
    assert after.stability > 0
    assert after.due > item.due  # pushed into the future
    row = sched.conn.execute("SELECT rating, source FROM reviews").fetchone()
    assert row["rating"] == 3
    assert row["source"] == "turn:1"


def test_again_increments_lapses(sched: ItemScheduler) -> None:
    item = sched.due_items(limit=1)[0]
    sched.record_review(item.id, Rating.Good)
    after = sched.record_review(item.id, Rating.Again)
    assert after.lapses == 1
    assert after.reps == 2


# ---- 30-day simulation ----------------------------------------------------


def test_thirty_day_schedule_produces_sane_intervals(sched: ItemScheduler) -> None:
    """Repeated Good over 30 simulated days must lengthen intervals, not thrash."""
    item = sched.due_items(limit=1)[0]
    now = datetime.now(UTC)
    intervals: list[float] = []
    prev_due = datetime.fromisoformat(item.due)

    for _ in range(8):
        got: Item = sched.record_review(item.id, Rating.Good, now=now)
        due = datetime.fromisoformat(got.due)
        intervals.append((due - now).total_seconds() / 86400)
        now = due  # review exactly when it comes up
        prev_due = due

    assert all(i > 0 for i in intervals), intervals
    # monotonically non-decreasing: consistent success must never shorten a gap
    assert intervals == sorted(intervals), intervals
    assert intervals[-1] > intervals[0]
    assert intervals[-1] < 36500, "interval exceeded the maximum-interval bound"
    assert prev_due > datetime.now(UTC)


def test_again_shortens_the_interval(sched: ItemScheduler) -> None:
    item = sched.due_items(limit=1)[0]
    now = datetime.now(UTC)
    for _ in range(4):
        got = sched.record_review(item.id, Rating.Good, now=now)
        now = datetime.fromisoformat(got.due)
    long_gap = datetime.fromisoformat(got.due) - now

    lapsed = sched.record_review(item.id, Rating.Again, now=now)
    short_gap = datetime.fromisoformat(lapsed.due) - now
    assert short_gap < long_gap + timedelta(days=1)


def test_retrievability_decays(sched: ItemScheduler) -> None:
    item = sched.due_items(limit=1)[0]
    now = datetime.now(UTC)
    sched.record_review(item.id, Rating.Good, now=now)
    fresh = sched.retrievability(item.id, now=now)
    later = sched.retrievability(item.id, now=now + timedelta(days=30))
    assert fresh > later
