-- Ocha initial schema.
--
-- FSRS state is stored as the library's own JSON blob plus two denormalised
-- columns. Rationale: py-fsrs owns the Card shape and it changes between major
-- versions (v6 dropped reps/lapses for state/step), so mapping every field into
-- columns creates a migration every time the library moves. `due` and
-- `stability` are denormalised because due_items() and lowest_stability() query
-- on them every turn and cannot scan JSON.

CREATE TABLE items (
    id          INTEGER PRIMARY KEY,
    kind        TEXT    NOT NULL CHECK (kind IN ('vocab', 'grammar', 'phrase')),
    content     TEXT    NOT NULL UNIQUE,      -- 食べる / particle_wa_ga
    reading     TEXT,                          -- たべる; null for grammar
    meaning_en  TEXT    NOT NULL,
    fsrs_json   TEXT    NOT NULL,              -- py-fsrs Card.to_json()
    due         TEXT    NOT NULL,              -- ISO8601 UTC, denormalised
    stability   REAL    NOT NULL DEFAULT 0,    -- denormalised
    reps        INTEGER NOT NULL DEFAULT 0,    -- py-fsrs v6 no longer tracks this
    lapses      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_items_due       ON items (due);
CREATE INDEX idx_items_stability ON items (stability);
CREATE INDEX idx_items_reps      ON items (reps);

CREATE TABLE reviews (
    id          INTEGER PRIMARY KEY,
    item_id     INTEGER NOT NULL REFERENCES items (id) ON DELETE CASCADE,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 4),  -- Again..Easy
    source      TEXT    NOT NULL,              -- how the rating was derived
    log_json    TEXT    NOT NULL,              -- py-fsrs ReviewLog.to_json()
    reviewed_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_reviews_item ON reviews (item_id);

CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at   TEXT
);

CREATE TABLE turns (
    id             INTEGER PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    user_text      TEXT    NOT NULL,
    tutor_text     TEXT    NOT NULL,
    target_item_ids TEXT   NOT NULL DEFAULT '[]',   -- JSON array
    derived_rating TEXT,                            -- JSON {item_id: rating}
    grammar_query  INTEGER NOT NULL DEFAULT 0,      -- did the firewall fire
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_turns_session ON turns (session_id);

-- Phase 2 writes this; Phase 1 ships it empty. PRD §10: Phase 2 must not
-- foreclose Phase 3, so raw 16 kHz audio is retained from the first voice turn
-- rather than the column being added later.
CREATE TABLE utterances (
    id          INTEGER PRIMARY KEY,
    turn_id     INTEGER NOT NULL REFERENCES turns (id) ON DELETE CASCADE,
    audio_path  TEXT    NOT NULL,
    sample_rate INTEGER NOT NULL DEFAULT 16000,
    duration_s  REAL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_utterances_turn ON utterances (turn_id);

-- Phase 3 writes this; ships empty. Scores are a trend, never an absolute grade
-- (PRD FR-6), and the accent cap on scheduling stays disabled until T3.6.
CREATE TABLE pronunciation_scores (
    id           INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterances (id) ON DELETE CASCADE,
    segmental    REAL,
    accent       REAL,
    rhythm       REAL,
    scorer       TEXT,                  -- model/version, so scores stay comparable
    scored_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_scores_utterance ON pronunciation_scores (utterance_id);

-- T1.5: on a grammar-reference miss the firewall returns "not yet documented"
-- and logs here. It must never fall back to generation.
CREATE TABLE unauthored_grammar (
    id           INTEGER PRIMARY KEY,
    user_text    TEXT NOT NULL,
    item_id      TEXT,                  -- the grammar.json id we looked for
    requested_at TEXT NOT NULL DEFAULT (datetime('now'))
);
