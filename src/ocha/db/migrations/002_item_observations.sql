CREATE TABLE item_observations (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    evidence TEXT NOT NULL CHECK (evidence IN ('mentioned', 'mentioned_after_prompt')),
    observed_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (turn_id, item_id)
);

CREATE INDEX idx_item_observations_item ON item_observations (item_id);
