from __future__ import annotations

from pathlib import Path

from ocha.db import connect, migrate
from ocha.lessons import current_step, load_lessons, record_progress, transcript_matches


def test_curated_course_loads_and_matches_only_accepted_forms(tmp_path: Path) -> None:
    db = connect(tmp_path / "lessons.db")
    migrate(db)
    lessons = load_lessons()
    assert [step.module_id for step in lessons].count("greetings") == 3
    assert [step.module_id for step in lessons].count("useful-requests") == 7

    thanks = next(step for step in lessons if step.id == "greeting-thanks")
    assert transcript_matches(thanks, " 有難うございます。 ")
    assert not transcript_matches(thanks, "ありがとう")

    assert current_step(db, lessons) == lessons[0]
    record_progress(db, lessons[0].id, "completed")
    assert current_step(db, lessons) == lessons[1]
    record_progress(db, lessons[1].id, "skipped")
    assert current_step(db, lessons) == lessons[2]
