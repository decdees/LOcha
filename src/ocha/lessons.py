"""Curated beginner speaking lessons and their small progress store."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal


@dataclass(frozen=True, slots=True)
class LessonStep:
    module_id: str
    module_title: str
    id: str
    japanese: str
    romaji: str
    meaning_en: str
    accepted_transcripts: tuple[str, ...]


def load_lessons() -> tuple[LessonStep, ...]:
    raw = files("ocha.resources").joinpath("lessons.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    return tuple(
        LessonStep(
            module_id=str(module["id"]),
            module_title=str(module["title"]),
            id=str(step["id"]),
            japanese=str(step["japanese"]),
            romaji=str(step["romaji"]),
            meaning_en=str(step["meaning_en"]),
            accepted_transcripts=tuple(str(value) for value in step["accepted_transcripts"]),
        )
        for module in payload["modules"]
        for step in module["steps"]
    )


def normalize_transcript(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).strip()
    return value.rstrip("。！？!? ")


def transcript_matches(step: LessonStep, transcript: str) -> bool:
    actual = normalize_transcript(transcript)
    return any(actual == normalize_transcript(value) for value in step.accepted_transcripts)


ProgressStatus = Literal["completed", "skipped"]


def record_progress(conn: sqlite3.Connection, step_id: str, status: ProgressStatus) -> None:
    conn.execute(
        "INSERT INTO guided_progress (step_id, status, updated_at) "
        "VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(step_id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at",
        (step_id, status),
    )


def current_step(conn: sqlite3.Connection, lessons: tuple[LessonStep, ...]) -> LessonStep | None:
    done = {str(row["step_id"]) for row in conn.execute("SELECT step_id FROM guided_progress")}
    return next((step for step in lessons if step.id not in done), None)
