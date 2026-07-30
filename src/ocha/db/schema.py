"""SQLite connection and migrations.

ponytail: a numbered-SQL-file migrator, not Alembic. Alembic exists to
autogenerate diffs against SQLAlchemy models; there are no SQLAlchemy models
here and there is one user, so it would be ~200 lines of env.py to run files we
could just run. Upgrade path if this ever gets a second writer or needs
downgrades: swap this for Alembic, keeping the same file order.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

MIGRATIONS = Path(__file__).parent / "migrations"
DEFAULT_DB = Path(os.environ.get("OCHA_DB", "data/ocha.db"))


def connect(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open a connection with the pragmas NFR-4 assumes."""
    p = Path(path)
    if p.parent != Path():
        p.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because FastAPI runs sync endpoints in a threadpool,
    # so the connection opened during lifespan startup is used from worker threads.
    # sqlite3 refuses that by default. Safe here only because writes are serialised
    # by a lock in the API layer -- one user, so contention is nil.
    conn = sqlite3.connect(
        p, isolation_level=None, check_same_thread=False
    )  # explicit transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")  # off by default in sqlite3
    conn.execute("PRAGMA synchronous = NORMAL")  # safe under WAL
    return conn


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply pending migrations in filename order. Returns those applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}
    applied = []
    for sql_file in sorted(MIGRATIONS.glob("*.sql")):
        if sql_file.name in done:
            continue
        # One transaction per migration: a half-applied schema is worse than a
        # failed one, and SQLite DDL is transactional.
        #
        # The BEGIN/COMMIT go *inside* the script rather than around it.
        # executescript() implicitly commits any open transaction before it
        # runs, so an outer conn.execute("BEGIN") is discarded and the matching
        # COMMIT then fails with "no transaction is active". Recording the
        # migration inside the same script keeps schema and bookkeeping atomic.
        name = sql_file.name.replace("'", "''")
        script = (
            "BEGIN;\n"
            + sql_file.read_text(encoding="utf-8")
            + f"\nINSERT INTO schema_migrations (name) VALUES ('{name}');\nCOMMIT;\n"
        )
        try:
            conn.executescript(script)
        except Exception:
            # execute(), not executescript(): executescript() commits any pending
            # transaction before running, so rolling back via executescript would
            # COMMIT the half-applied migration first -- the exact opposite of the
            # intent. tests/test_db.py pins this.
            conn.execute("ROLLBACK")
            raise
        applied.append(sql_file.name)
    return applied
