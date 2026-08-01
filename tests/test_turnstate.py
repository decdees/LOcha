"""T2.8 foundation — turn state and the G1a instrument."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import ItemScheduler
from ocha.turnstate import MAX_SILENT_GAP_S, TurnState, TurnTimeline
from ocha.tutor.firewall import SENTINEL
from ocha.tutor.grammar import load_grammar
from ocha.tutor.llm import ChatMessage, StubLlm
from ocha.tutor.turn import run_turn


class FakeClock:
    """Deterministic time. Real sleeps make the suite slow and flaky."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def timeline_with(clock: FakeClock) -> TurnTimeline:
    return TurnTimeline(_clock=clock)


# ---- the G1a check -------------------------------------------------------


def test_fast_turn_satisfies_g1a() -> None:
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.LISTENING)
    c.advance(0.2)
    tl.emit(TurnState.TRANSCRIBING)
    c.advance(0.3)
    tl.emit(TurnState.THINKING)
    c.advance(0.4)
    tl.emit(TurnState.SPEAKING)
    c.advance(0.1)
    tl.finish()
    assert tl.satisfies_g1a()
    assert tl.violations() == []


def test_a_long_silent_state_violates_g1a() -> None:
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.THINKING)
    c.advance(1.2)  # the measured LLM stage
    tl.emit(TurnState.SPEAKING)
    c.advance(0.1)
    tl.finish()
    assert not tl.satisfies_g1a()
    ((state, gap),) = tl.violations()
    assert state is TurnState.THINKING
    assert gap == pytest.approx(1.2)


def test_speaking_is_credited_with_the_audio_it_delivered() -> None:
    """A long SPEAKING stretch is the tutor talking -- for as long as there is audio.

    This test used to assert that SPEAKING was exempt outright, with no audio
    accounted for. That was the loophole T2.8 closed: a tutor that said 「ええと」
    once and then went quiet for eight seconds passed, which is relabelling the
    criterion rather than meeting it. Audio is feedback for its own duration.
    """
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.SPEAKING)
    tl.add_audio(8.0)  # eight seconds of reply actually played
    c.advance(8.0)
    tl.finish()
    assert tl.satisfies_g1a()


def test_speaking_with_no_audio_behind_it_is_still_silence() -> None:
    """The loophole, asserted directly so it cannot reopen."""
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.SPEAKING)
    tl.add_audio(0.7)  # one filled pause
    c.advance(8.0)  # ...then nothing for eight seconds
    tl.finish()
    assert not tl.satisfies_g1a(), "silence after the audio ran out is still silence"
    ((state, silent),) = tl.violations()
    assert state is TurnState.SPEAKING
    assert silent == pytest.approx(7.3)


def test_the_boundary_is_inclusive() -> None:
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.THINKING)
    c.advance(MAX_SILENT_GAP_S)
    tl.finish()
    assert tl.satisfies_g1a(), "exactly at the limit must pass"

    c2 = FakeClock()
    tl2 = timeline_with(c2)
    tl2.emit(TurnState.THINKING)
    c2.advance(MAX_SILENT_GAP_S + 0.01)
    tl2.finish()
    assert not tl2.satisfies_g1a()


def test_splitting_a_long_state_fixes_the_violation() -> None:
    """The intended remedy: emit an intermediate state rather than go faster.

    This is what makes G1a achievable at the measured 3.03 s -- feedback, not
    speed. A 1.2 s think broken by a partial-transcript update passes.
    """
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.TRANSCRIBING, "partial")
    c.advance(0.4)
    tl.emit(TurnState.TRANSCRIBING, "partial updated")
    c.advance(0.4)
    tl.emit(TurnState.THINKING)
    c.advance(0.4)
    tl.emit(TurnState.SPEAKING)
    c.advance(0.1)
    tl.finish()
    assert tl.satisfies_g1a()


def test_longest_gap_identifies_the_worst_stage() -> None:
    c = FakeClock()
    tl = timeline_with(c)
    tl.emit(TurnState.TRANSCRIBING)
    c.advance(0.3)
    tl.emit(TurnState.THINKING)
    c.advance(0.9)
    tl.emit(TurnState.SPEAKING)
    c.advance(0.1)
    tl.finish()
    worst = tl.longest_silent_gap()
    assert worst is not None
    assert worst[0] is TurnState.THINKING


def test_empty_timeline_is_vacuously_fine() -> None:
    tl = TurnTimeline()
    assert tl.satisfies_g1a()
    assert tl.longest_silent_gap() is None


# ---- every state must be renderable --------------------------------------


def test_no_catch_all_processing_state() -> None:
    """A state nobody can render does not satisfy G1a. The enum deliberately has
    no generic 'processing' bucket -- each value maps to something the client
    shows."""
    values = {s.value for s in TurnState}
    assert "processing" not in values
    assert "busy" not in values
    assert values == {
        "idle",
        "listening",
        "transcribing",
        "thinking",
        "speaking",
        "grammar",
    }


# ---- against the real turn path ------------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> tuple[sqlite3.Connection, ItemScheduler]:
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    seed(conn)
    return conn, ItemScheduler(conn)


def test_run_turn_records_a_timeline(env: tuple[sqlite3.Connection, ItemScheduler]) -> None:
    conn, sched = env
    res = run_turn(conn, sched, load_grammar(), StubLlm(), "こんにちは。")
    assert res.timeline is not None
    states = [c.state for c in res.timeline.changes]
    assert TurnState.THINKING in states
    assert TurnState.SPEAKING in states
    assert states[-1] is TurnState.IDLE  # the turn is closed


def test_grammar_turn_reports_the_grammar_state(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    """The firewall panel is its own feedback state -- the user is shown a
    reference entry, not a tutor reply."""
    conn, sched = env
    res = run_turn(conn, sched, load_grammar(), StubLlm(reply=SENTINEL), "は vs が?")
    assert res.timeline is not None
    assert TurnState.GRAMMAR in [c.state for c in res.timeline.changes]


def test_current_turn_path_violates_g1a_at_realistic_latency(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    """G1a is violable TODAY, before any voice component exists.

    T0.9 measured the LLM stage at ~0.75 s and the whole chain at 3.03 s. With a
    stub standing in for that latency, the single THINKING state exceeds 500 ms
    and nothing tells the user anything. This is the gap T2.8's UI has to close,
    and it is why the feedback states are load-bearing rather than polish.
    """
    conn, sched = env

    clock = FakeClock()

    class SlowStub(StubLlm):
        def generate(
            self,
            s: str,
            u: str,
            *,
            history: Sequence[ChatMessage] = (),
            max_tokens: int = 64,
        ) -> str:
            clock.advance(0.75)  # T0.9's measured first-sentence stage
            return super().generate(s, u, history=history, max_tokens=max_tokens)

    tl = timeline_with(clock)
    run_turn(conn, sched, load_grammar(), SlowStub(), "こんにちは。", timeline=tl)

    assert not tl.satisfies_g1a(), "if this passes, the instrument is broken"
    worst = tl.longest_silent_gap()
    assert worst is not None and worst[0] is TurnState.THINKING
