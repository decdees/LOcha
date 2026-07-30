"""FSRS rating derivation (PRD FR-8).

Ratings are derived from what the learner actually produced, never self-reported.
"""

from __future__ import annotations

from enum import StrEnum

from fsrs import Rating

# PRD FR-8, as amended: the accent cap is PROVISIONAL and disabled by default.
# It must not gate scheduling until T3.6 validates that scoring against a
# VOICEVOX reference ranks attempts the same way scoring against a native
# speaker does. The code path and its tests exist from Phase 1; only the score
# is absent, so turning this on later is a flag flip rather than a change.
ACCENT_CAP_ENABLED = False

# Below this, a word is not considered "known" however correct the grammar was.
# Inert while ACCENT_CAP_ENABLED is False.
ACCENT_THRESHOLD = 0.6


class Usage(StrEnum):
    """How the learner handled a target item in a turn."""

    UNPROMPTED = "unprompted"  # produced it correctly, unaided
    HINTED = "hinted"  # produced it, but only after a hint
    AVOIDED = "avoided"  # dodged it, or substituted something else


def derive_rating(
    usage: Usage,
    *,
    was_due: bool = True,
    accent_score: float | None = None,
    accent_cap_enabled: bool = ACCENT_CAP_ENABLED,
) -> Rating:
    """Map production evidence to an FSRS rating.

    `was_due` separates Good from Easy. PRD FR-8 lists "Good / Easy" for
    unprompted correct use without saying which. Producing an item
    spontaneously *before* it came due is genuine evidence the scheduler
    underestimated it -- that is what Easy means -- whereas producing it on
    schedule is exactly Good. This is the only signal available at Phase 1 that
    distinguishes them without inventing a threshold.
    """
    if usage is Usage.AVOIDED:
        rating = Rating.Again
    elif usage is Usage.HINTED:
        rating = Rating.Hard
    else:
        rating = Rating.Good if was_due else Rating.Easy

    # "A word you produce with wrong accent is not a word you know."
    if accent_cap_enabled and accent_score is not None and accent_score < ACCENT_THRESHOLD:
        rating = min(rating, Rating.Hard, key=lambda r: r.value)

    return rating
