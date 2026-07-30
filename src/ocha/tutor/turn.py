"""Turn orchestration (T1.8).

One turn: build context from FSRS state, generate, firewall the output, score
what the learner produced, persist. The firewall is the only path by which model
output becomes a response -- there is no branch around it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ocha.scheduling.rating import Usage, derive_rating
from ocha.scheduling.scheduler import Item, ItemScheduler
from ocha.tutor.context import MAX_REPLY_TOKENS, build_context
from ocha.tutor.firewall import GrammarResponse, apply_firewall
from ocha.tutor.grammar import GrammarReference
from ocha.tutor.llm import LlmService
from ocha.tutor.usage import detect_usage


@dataclass(frozen=True, slots=True)
class TurnResult:
    session_id: int
    turn_id: int
    reply: str | None
    grammar: GrammarResponse | None
    targets: tuple[str, ...]
    ratings: dict[int, int]
    usage: dict[int, str]

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


def run_turn(
    conn: sqlite3.Connection,
    scheduler: ItemScheduler,
    reference: GrammarReference,
    llm: LlmService,
    user_text: str,
    *,
    session_id: int | None = None,
) -> TurnResult:
    session = ensure_session(conn, session_id)
    previous = _last_tutor_text(conn, session)

    ctx = build_context(scheduler)
    raw = llm.generate(ctx.system_prompt, user_text, max_tokens=MAX_REPLY_TOKENS)
    outcome = apply_firewall(raw, user_text, reference, conn)

    target_items: list[Item] = [
        i for i in scheduler.due_items(limit=20) if i.id in set(ctx.target_ids)
    ]
    # due_items cannot see targets that came from lowest_stability, so top up.
    have = {i.id for i in target_items}
    for weak in scheduler.lowest_stability(limit=10):
        if weak.id in ctx.target_ids and weak.id not in have:
            target_items.append(weak)
            have.add(weak.id)

    ratings: dict[int, int] = {}
    usage: dict[int, str] = {}

    # A grammar question is not a production attempt. Scoring it would punish the
    # learner for asking, which is the opposite of what the firewall is for.
    if not outcome.fired:
        report = detect_usage(target_items, user_text, previous)
        by_id = {i.id: i for i in target_items}
        for item_id, how in report.usage.items():
            item = by_id[item_id]
            was_due = item.id in set(i.id for i in scheduler.due_items(limit=20))
            rating = derive_rating(Usage(how), was_due=was_due)
            scheduler.record_review(item_id, rating, source=f"turn:session{session}")
            ratings[item_id] = int(rating.value)
            usage[item_id] = how.value

    tutor_text = outcome.reply if outcome.reply is not None else ""
    cur = conn.execute(
        "INSERT INTO turns (session_id, user_text, tutor_text, target_item_ids,"
        " derived_rating, grammar_query) VALUES (?, ?, ?, ?, ?, ?)",
        (
            session,
            user_text,
            tutor_text,
            json.dumps(list(ctx.target_ids)),
            json.dumps(ratings),
            int(outcome.fired),
        ),
    )

    return TurnResult(
        session_id=session,
        turn_id=int(cur.lastrowid or 0),
        reply=outcome.reply,
        grammar=outcome.grammar,
        targets=ctx.target_contents,
        ratings=ratings,
        usage=usage,
    )
