CREATE TABLE guided_progress (
    step_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('completed', 'skipped')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
