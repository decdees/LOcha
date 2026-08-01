"""T1.6 — Context Builder. Snapshot tests over three distinct FSRS states."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fsrs import Rating

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import ItemScheduler
from ocha.tutor.context import (
    MAX_CONTEXT_TOKENS,
    MAX_INTRODUCE,
    MAX_REPLY_TOKENS,
    build_context,
)
from ocha.tutor.firewall import SENTINEL


@pytest.fixture
def sched(tmp_path: Path) -> ItemScheduler:
    conn: sqlite3.Connection = connect(tmp_path / "t.db")
    migrate(conn)
    seed(conn)
    return ItemScheduler(conn)


def _make_known(sched: ItemScheduler, n: int, reps: int = 3) -> list[int]:
    ids = [i.id for i in sched.due_items(limit=n)]
    for item_id in ids:
        for _ in range(reps):
            sched.record_review(item_id, Rating.Good)
    return ids


# ---- state 1: cold start, nothing reviewed -------------------------------


def test_cold_start_has_no_known_items(sched: ItemScheduler) -> None:
    ctx = build_context(sched)
    assert ctx.known_ids == ()
    assert "(none yet)" in ctx.system_prompt
    # Nothing is known, so everything on offer must be in INTRODUCE.
    assert "INTRODUCE" in ctx.system_prompt
    assert "(nothing -- converse freely within KNOWN)" in ctx.system_prompt


def test_cold_start_still_carries_the_required_lines(sched: ItemScheduler) -> None:
    p = build_context(sched).system_prompt
    assert "REGISTER: Always use polite です/ます form." in p  # T0.5: load-bearing
    assert SENTINEL in p  # the firewall cannot fire if the model is never told
    assert "1-2 short sentences" in p  # FR-3
    assert "learner may speak English" in p


# ---- state 2: some items known -------------------------------------------


def test_known_items_appear_in_known_list(sched: ItemScheduler) -> None:
    known = _make_known(sched, 4)
    ctx = build_context(sched)
    assert set(ctx.known_ids) == set(known)
    for item_id in known:
        content = sched.conn.execute(
            "SELECT content FROM items WHERE id = ?", (item_id,)
        ).fetchone()[0]
        assert content in ctx.system_prompt


def test_practise_and_introduce_are_disjoint(sched: ItemScheduler) -> None:
    """The §7.1 template as written was self-contradictory: it said "use only
    KNOWN" while listing due items, which are typically NOT known. Splitting the
    target list is the fix, and the two halves must never overlap."""
    _make_known(sched, 4)
    p = build_context(sched).system_prompt
    practise = p.split("PRACTISE")[1].split("INTRODUCE")[0]
    introduce = p.split("INTRODUCE")[1].split("REGISTER")[0]
    prac_items = {s.split("=")[0].strip() for s in practise.split("、") if "=" in s}
    intro_items = {s.split("=")[0].strip() for s in introduce.split("、") if "=" in s}
    assert prac_items & intro_items == set()


def test_introduce_is_capped(sched: ItemScheduler) -> None:
    """A 1-2 sentence reply introducing at most one new word cannot steer toward
    eight of them; listing more only dilutes the instruction."""
    ctx = build_context(sched, due_limit=10, weak_limit=5)
    introduce = ctx.system_prompt.split("INTRODUCE")[1].split("REGISTER")[0]
    assert len([s for s in introduce.split("、") if "=" in s]) <= MAX_INTRODUCE


def test_kana_readings_are_not_duplicated(sched: ItemScheduler) -> None:
    """A kana word's reading equals its content. Repeating it doubles the KNOWN
    list for nothing, and KNOWN dominates prompt length -> prefill -> latency."""
    _make_known(sched, 6)
    p = build_context(sched).system_prompt
    known_line = next(line for line in p.splitlines() if line.startswith("KNOWN:"))
    assert "あれ(あれ)" not in known_line
    assert "これ(これ)" not in known_line


# ---- state 3: some items weak -------------------------------------------


def test_weak_known_item_lands_in_practise(sched: ItemScheduler) -> None:
    ids = _make_known(sched, 3)
    weakest = ids[0]
    sched.record_review(weakest, Rating.Again)  # still known (reps>=3), now weak
    ctx = build_context(sched)
    assert weakest in ctx.weak_ids
    content = sched.conn.execute("SELECT content FROM items WHERE id = ?", (weakest,)).fetchone()[0]
    practise = ctx.system_prompt.split("PRACTISE")[1].split("INTRODUCE")[0]
    assert content in practise


def test_target_ids_have_no_duplicates(sched: ItemScheduler) -> None:
    """due and lowest_stability overlap freely; the merge must de-duplicate or the
    same item gets scored twice in T1.8."""
    _make_known(sched, 5)
    ctx = build_context(sched)
    assert len(ctx.target_ids) == len(set(ctx.target_ids))


# ---- budget --------------------------------------------------------------


def test_prompt_stays_well_inside_the_context_cap(sched: ItemScheduler) -> None:
    """T0.4 revised the cap from 8k to ~2k: at 8k, TTFT was 32.6 s and decode
    fell to 14.3 tok/s -- slower than the dense model the MoE was chosen over."""
    _make_known(sched, 50)  # every seeded item known: worst realistic case
    ctx = build_context(sched)
    # ~1 token per 2 chars is conservative for mixed Japanese/ASCII.
    approx_tokens = len(ctx.system_prompt) / 2
    assert approx_tokens < MAX_CONTEXT_TOKENS, approx_tokens
    assert MAX_REPLY_TOKENS <= 128  # FR-3 enforced by max_tokens, not just prompt


def test_known_limit_bounds_the_prompt(sched: ItemScheduler) -> None:
    _make_known(sched, 20)
    small = build_context(sched, known_limit=5)
    large = build_context(sched, known_limit=20)
    assert len(small.known_ids) == 5
    assert len(small.system_prompt) < len(large.system_prompt)


# ---- snapshot ------------------------------------------------------------


def test_prompt_shape_is_stable(sched: ItemScheduler) -> None:
    """Section order is the snapshot. If a section is renamed or dropped, T1.8's
    scoring and the T0.5 probe stop describing the same prompt."""
    _make_known(sched, 4)
    p = build_context(sched).system_prompt
    order = [
        "You are a Japanese conversation partner.",
        "VOCABULARY:",
        "KNOWN:",
        "PRACTISE",
        "INTRODUCE",
        "REGISTER:",
        "AVOID:",
        "Never break character",
    ]
    positions = [p.index(s) for s in order]
    assert positions == sorted(positions), "prompt sections reordered"
