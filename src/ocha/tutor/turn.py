"""Turn orchestration (T1.8).

One turn: build context from FSRS state, generate, firewall the output, score
what the learner produced, persist. The firewall is the only path by which model
output becomes a response -- there is no branch around it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ocha.scheduling.scheduler import Item, ItemScheduler
from ocha.turnstate import TurnState, TurnTimeline
from ocha.tutor.context import MAX_REPLY_TOKENS, TurnContext, build_context
from ocha.tutor.firewall import GrammarResponse, apply_firewall
from ocha.tutor.grammar import GrammarReference
from ocha.tutor.llm import ChatMessage, LlmService
from ocha.tutor.observation import Evidence, observe_targets


@dataclass(frozen=True, slots=True)
class TurnResult:
    session_id: int
    turn_id: int
    reply: str | None
    grammar: GrammarResponse | None
    targets: tuple[str, ...]
    observations: dict[int, Evidence]
    # Compatibility fields. Free conversation is not a validated drill and does
    # not produce FSRS ratings.
    ratings: dict[int, int]
    usage: dict[int, str]
    # T2.8: what the user was told was happening, and when. Recorded from Phase 1
    # so the pipeline built in T2.1-T2.5 has a contract to emit against rather
    # than a retrofit afterwards.
    timeline: TurnTimeline | None = None

    @property
    def grammar_query(self) -> bool:
        return self.grammar is not None


def ensure_session(conn: sqlite3.Connection, session_id: int | None) -> int:
    if session_id is not None:
        row = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"no session {session_id}")
        return session_id
    cur = conn.execute("INSERT INTO sessions DEFAULT VALUES")
    return int(cur.lastrowid or 0)


def _last_tutor_text(conn: sqlite3.Connection, session_id: int) -> str | None:
    row = conn.execute(
        "SELECT tutor_text FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    return None if row is None else str(row["tutor_text"])


def conversation_history(
    conn: sqlite3.Connection, session_id: int, *, exchanges: int = 4
) -> tuple[ChatMessage, ...]:
    """Return explicit, complete non-grammar exchanges, oldest first."""
    rows = conn.execute(
        "SELECT user_text, tutor_text FROM turns "
        "WHERE session_id = ? AND grammar_query = 0 AND tutor_text <> '' "
        "ORDER BY id DESC LIMIT ?",
        (session_id, exchanges),
    ).fetchall()
    return tuple(
        message
        for row in reversed(rows)
        for message in (
            ChatMessage(role="user", content=str(row["user_text"])),
            ChatMessage(role="assistant", content=str(row["tutor_text"])),
        )
    )


def run_turn(
    conn: sqlite3.Connection,
    scheduler: ItemScheduler,
    reference: GrammarReference,
    llm: LlmService,
    user_text: str,
    *,
    session_id: int | None = None,
    timeline: TurnTimeline | None = None,
    raw: str | None = None,
) -> TurnResult:
    """Build context, generate a complete reply, then finalize it once."""
    tl = timeline if timeline is not None else TurnTimeline()
    session = ensure_session(conn, session_id)
    tl.emit(TurnState.THINKING, "building context")
    ctx = build_context(scheduler)
    if raw is None:
        raw = llm.generate(
            ctx.system_prompt,
            user_text,
            history=conversation_history(conn, session),
            max_tokens=MAX_REPLY_TOKENS,
        )
    return finalize_turn(
        conn,
        scheduler,
        reference,
        user_text,
        context=ctx,
        raw=raw,
        session_id=session,
        timeline=tl,
    )


def finalize_turn(
    conn: sqlite3.Connection,
    scheduler: ItemScheduler,
    reference: GrammarReference,
    user_text: str,
    *,
    context: TurnContext,
    raw: str,
    session_id: int | None = None,
    timeline: TurnTimeline | None = None,
) -> TurnResult:
    """Firewall one complete generation, then score and persist its safe result.

    This is the sole boundary between model output and user-visible output. Callers
    may collect generation differently, but must not emit any part of ``raw``
    before this function returns.
    """
    tl = timeline if timeline is not None else TurnTimeline()
    session = ensure_session(conn, session_id)
    previous = _last_tutor_text(conn, session)

    outcome = apply_firewall(raw, user_text, reference, conn)
    tl.emit(
        TurnState.GRAMMAR if outcome.fired else TurnState.SPEAKING,
        "reference served" if outcome.fired else "first sentence ready",
    )

    target_items: list[Item] = [
        i for i in scheduler.due_items(limit=20) if i.id in set(context.target_ids)
    ]
    # due_items cannot see targets that came from lowest_stability, so top up.
    have = {i.id for i in target_items}
    for weak in scheduler.lowest_stability(limit=10):
        if weak.id in context.target_ids and weak.id not in have:
            target_items.append(weak)
            have.add(weak.id)

    observations: dict[int, Evidence] = {}
    if not outcome.fired:
        observations = observe_targets(target_items, user_text, previous).observations

    tutor_text = outcome.reply if outcome.reply is not None else ""
    cur = conn.execute(
        "INSERT INTO turns (session_id, user_text, tutor_text, target_item_ids,"
        " derived_rating, grammar_query) VALUES (?, ?, ?, ?, ?, ?)",
        (
            session,
            user_text,
            tutor_text,
            json.dumps(list(context.target_ids)),
            "{}",
            int(outcome.fired),
        ),
    )
    turn_id = int(cur.lastrowid or 0)
    conn.executemany(
        "INSERT INTO item_observations (turn_id, item_id, evidence) VALUES (?, ?, ?)",
        ((turn_id, item_id, evidence) for item_id, evidence in observations.items()),
    )

    tl.finish()
    return TurnResult(
        session_id=session,
        turn_id=turn_id,
        reply=outcome.reply,
        grammar=outcome.grammar,
        targets=context.target_contents,
        observations=observations,
        ratings={},
        usage={},
        timeline=tl,
    )
