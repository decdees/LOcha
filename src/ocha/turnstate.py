"""Turn state and the G1a instrument (T2.8 foundation).

PRD G1a: no dead air exceeding 500 ms without visible or audible feedback.

This is the part of T2.8 that must come FIRST, not last. T2.8 as written is a UI
task, and a UI cannot be built before the pipeline that feeds it. But if T2.1-T2.5
are built without emitting turn-state events, the feedback states become a
retrofit through five components. Defining the contract first is cheaper, and it
makes G1a a property that can be asserted rather than eyeballed.

G1a is already violable. POST /turn today returns nothing for roughly a second
while the LLM generates -- dead air, no feedback. So this instrument has real
behaviour to measure before any voice component exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

# PRD G1a. Above this, the user needs to see or hear something.
MAX_SILENT_GAP_S = 0.5


class TurnState(StrEnum):
    """What the user is told is happening.

    Each state must correspond to something the client can actually render. A
    state nobody can see does not satisfy G1a -- that is the whole point of the
    criterion, so the enum deliberately has no "processing" catch-all.
    """

    IDLE = "idle"  # waiting for the user to start
    LISTENING = "listening"  # mic open, speech detected
    TRANSCRIBING = "transcribing"  # ASR running; partial transcript may show
    THINKING = "thinking"  # LLM generating, nothing speakable yet
    SPEAKING = "speaking"  # TTS audio playing
    GRAMMAR = "grammar"  # firewall fired; reference panel shown


@dataclass(frozen=True, slots=True)
class StateChange:
    state: TurnState
    at: float  # monotonic seconds
    detail: str = ""


@dataclass(slots=True)
class TurnTimeline:
    """Records state transitions so G1a can be asserted rather than eyeballed.

    Deliberately a plain recorder, not a state machine with legal transitions.
    The pipeline is not built yet, and inventing a transition table before the
    thing it constrains would be guessing. What is needed now is the contract for
    WHAT gets emitted and the check on the gaps between emissions.
    """

    changes: list[StateChange] = field(default_factory=list)
    _clock: object = field(default=time.monotonic, repr=False)

    def emit(self, state: TurnState, detail: str = "") -> StateChange:
        change = StateChange(state=state, at=self._clock(), detail=detail)  # type: ignore[operator]
        self.changes.append(change)
        return change

    def finish(self) -> None:
        """Close the turn so the final state's duration is bounded."""
        self.emit(TurnState.IDLE, "turn complete")

    @property
    def gaps(self) -> list[tuple[TurnState, float]]:
        """Seconds spent in each state, from its emission to the next one."""
        return [
            (a.state, b.at - a.at) for a, b in zip(self.changes, self.changes[1:], strict=False)
        ]

    def longest_silent_gap(self) -> tuple[TurnState, float] | None:
        """The longest stretch in a single state with no new feedback."""
        return max(self.gaps, key=lambda g: g[1], default=None)

    def satisfies_g1a(self, limit: float = MAX_SILENT_GAP_S) -> bool:
        """True when no state persisted longer than `limit` without an update.

        SPEAKING is exempt: audio is itself continuous feedback, so a long
        SPEAKING stretch is the tutor talking, not the app hanging. Every other
        state is silent from the user's point of view.
        """
        return all(gap <= limit for state, gap in self.gaps if state is not TurnState.SPEAKING)

    def violations(self, limit: float = MAX_SILENT_GAP_S) -> list[tuple[TurnState, float]]:
        return [(s, g) for s, g in self.gaps if s is not TurnState.SPEAKING and g > limit]
