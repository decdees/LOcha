"""Conservative vocabulary observations from free conversation.

Morphology can establish that a target surface or lemma occurred. It cannot
establish grammatical correctness, recall quality, or an FSRS rating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ocha.scheduling.scheduler import Item
from ocha.tutor.usage import content_forms

Evidence = Literal["mentioned", "mentioned_after_prompt"]


@dataclass(frozen=True, slots=True)
class ObservationReport:
    observations: dict[int, Evidence]


def observe_targets(
    targets: list[Item], learner_text: str, previous_tutor_text: str | None
) -> ObservationReport:
    """Record only target forms actually present in the learner transcript."""
    learner_forms = content_forms(learner_text)
    prompted_forms = content_forms(previous_tutor_text or "")
    observations: dict[int, Evidence] = {}
    for item in targets:
        if item.content not in learner_forms:
            continue
        observations[item.id] = (
            "mentioned_after_prompt" if item.content in prompted_forms else "mentioned"
        )
    return ObservationReport(observations)
