"""py-fsrs wrapper backed by SQLite.

Exposes exactly the four queries ARCHITECTURE §7.1's Context Builder needs, so
the tutor layer never touches FSRS internals or SQL.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from fsrs import Card, Rating, Scheduler


@dataclass(frozen=True, slots=True)
class Item:
    id: int
    kind: str
    content: str
    reading: str | None
    meaning_en: str
    due: str
    stability: float
    reps: int
    lapses: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Item:
        return cls(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            reading=row["reading"],
            meaning_en=row["meaning_en"],
            due=row["due"],
            stability=row["stability"],
            reps=row["reps"],
            lapses=row["lapses"],
        )


_COLS = "id, kind, content, reading, meaning_en, due, stability, reps, lapses"


class ItemScheduler:
    def __init__(self, conn: sqlite3.Connection, scheduler: Scheduler | None = None) -> None:
        self.conn = conn
        self.fsrs = scheduler or Scheduler()

    # ---- Context Builder queries (ARCHITECTURE §7.1) -------------------

    def due_items(self, limit: int = 5, now: datetime | None = None) -> list[Item]:
        """Items at or past their due date, soonest first."""
        ts = (now or datetime.now(UTC)).isoformat()
        rows = self.conn.execute(
            f"SELECT {_COLS} FROM items WHERE due <= ? ORDER BY due LIMIT ?", (ts, limit)
        )
        return [Item.from_row(r) for r in rows]

    def known_items(self, min_reps: int = 3, limit: int = 200) -> list[Item]:
        """The safe vocabulary pool -- items reviewed enough to rely on.

        py-fsrs v6 dropped reps from Card, so this reads our own column.
        """
        rows = self.conn.execute(
            f"SELECT {_COLS} FROM items WHERE reps >= ? ORDER BY stability DESC LIMIT ?",
            (min_reps, limit),
        )
        return [Item.from_row(r) for r in rows]

    def lowest_stability(self, limit: int = 3) -> list[Item]:
        """Struggling items. Only ones actually seen -- an unreviewed item has
        stability 0 and would otherwise crowd out the genuinely weak ones."""
        rows = self.conn.execute(
            f"SELECT {_COLS} FROM items WHERE reps > 0 ORDER BY stability ASC LIMIT ?",
            (limit,),
        )
        return [Item.from_row(r) for r in rows]

    # ---- mutation ------------------------------------------------------

    def record_review(
        self,
        item_id: int,
        rating: Rating,
        *,
        source: str = "derived",
        now: datetime | None = None,
    ) -> Item:
        """Apply a rating and persist the new FSRS state."""
        row = self.conn.execute(
            "SELECT fsrs_json, reps, lapses FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no item {item_id}")

        card = Card.from_json(row["fsrs_json"])
        when = now or datetime.now(UTC)
        card, log = self.fsrs.review_card(card, rating, review_datetime=when)

        reps = row["reps"] + 1
        lapses = row["lapses"] + (1 if rating is Rating.Again else 0)

        self.conn.execute("BEGIN")
        try:
            self.conn.execute(
                "UPDATE items SET fsrs_json = ?, due = ?, stability = ?, reps = ?, lapses = ?"
                " WHERE id = ?",
                (
                    card.to_json(),
                    card.due.isoformat(),
                    card.stability or 0.0,
                    reps,
                    lapses,
                    item_id,
                ),
            )
            self.conn.execute(
                "INSERT INTO reviews (item_id, rating, source, log_json, reviewed_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    item_id,
                    int(rating.value),
                    source,
                    json.dumps(log.to_dict(), default=str),
                    when.isoformat(),
                ),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        got = self.conn.execute(f"SELECT {_COLS} FROM items WHERE id = ?", (item_id,)).fetchone()
        return Item.from_row(got)

    def retrievability(self, item_id: int, now: datetime | None = None) -> float:
        row = self.conn.execute("SELECT fsrs_json FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(f"no item {item_id}")
        card = Card.from_json(row["fsrs_json"])
        return float(self.fsrs.get_card_retrievability(card, current_datetime=now))
