from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ocha.db import connect, migrate
from ocha.db.seed import VOCAB, seed


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "t.db")
    migrate(c)
    return c


def test_migrations_apply_and_are_idempotent(tmp_path: Path) -> None:
    c = connect(tmp_path / "t.db")
    assert migrate(c) == [
        "001_init.sql",
        "002_item_observations.sql",
        "003_guided_progress.sql",
    ]
    assert migrate(c) == []  # re-running must be a no-op


def test_required_tables_exist(conn: sqlite3.Connection) -> None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # T1.2 names six; unauthored_grammar is T1.5's miss log.
    assert {
        "items",
        "reviews",
        "item_observations",
        "sessions",
        "turns",
        "utterances",
        "pronunciation_scores",
        "unauthored_grammar",
    } <= tables


def test_pragmas(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"  # NFR-4
    # sqlite3 leaves foreign keys OFF by default; the CASCADEs are load-bearing.
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_foreign_keys_are_enforced(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO reviews (item_id, rating, source, log_json) VALUES (999,3,'x','{}')"
        )


def test_seed_is_idempotent(conn: sqlite3.Connection) -> None:
    assert seed(conn) == len(VOCAB) == 50
    assert seed(conn) == 0
    assert conn.execute("SELECT count(*) FROM items").fetchone()[0] == 50


def test_phase3_tables_ship_empty(conn: sqlite3.Connection) -> None:
    """PRD §10: Phase 2 must not foreclose Phase 3. These exist from day one so
    raw audio and scores are never bolted on later."""
    for t in ("utterances", "pronunciation_scores"):
        assert conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == 0


def test_rating_check_constraint(conn: sqlite3.Connection) -> None:
    seed(conn)
    item = conn.execute("SELECT id FROM items LIMIT 1").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO reviews (item_id, rating, source, log_json) VALUES (?,5,'x','{}')",
            (item,),
        )


def test_failed_migration_leaves_no_partial_schema(tmp_path: Path) -> None:
    """A half-applied schema is worse than a failed one."""
    import ocha.db.schema as schema

    bad = tmp_path / "migrations"
    bad.mkdir()
    (bad / "001_ok.sql").write_text("CREATE TABLE a (id INTEGER);")
    (bad / "002_broken.sql").write_text("CREATE TABLE b (id INTEGER); CREATE TABLE b (oops);")
    original, schema.MIGRATIONS = schema.MIGRATIONS, bad
    try:
        c = connect(tmp_path / "t.db")
        with pytest.raises(sqlite3.OperationalError):
            migrate(c)
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "a" in tables  # first migration committed
        assert "b" not in tables  # second rolled back entirely
        applied = {r[0] for r in c.execute("SELECT name FROM schema_migrations")}
        assert applied == {"001_ok.sql"}
    finally:
        schema.MIGRATIONS = original
