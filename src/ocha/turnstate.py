"""Turn state and the G1a instrument (T2.8 foundation).

PRD G1a: **no more than 500 ms may pass without an audible or visible change.**

Reworded in T2.8 from "no dead air exceeding 500 ms without visible or audible
feedback". The old wording did not say what feedback *is*, and this instrument had
to guess: it exempted the SPEAKING state entirely, on the reasoning that audio is
continuous feedback. That is true only while audio is actually playing. A tutor
that emitted one 「ええと」 and then went quiet for two seconds satisfied the old
check, which is the criterion being relabelled rather than met.

So there is no state-based exemption at all any more. Instead the timeline tracks
**when audio is actually playing**, and a stretch is silent only for the part of it
that no audio covers. A filled pause fired at the VAD endpoint keeps covering the
user while the state is still nominally `transcribing` -- which is the truth, and
was the second thing the state-based version got wrong: it credited audio to
whichever state happened to be current when the frames arrived, so a filler that
played across a state change stopped counting at the boundary.

Audio delivered in a burst still plays sequentially, so coverage is modelled as a
playback cursor rather than "each frame covers the moment it arrived".

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


@dataclass(slots=True)
class StateChange:
    state: TurnState
    at: float  # monotonic seconds
    detail: str = ""
    # Seconds of audio delivered while this state was current. Mutable and
    # accumulated by the probe as frames pass, because how long a SPEAKING stretch
    # is *entitled* to last is exactly how much audio it produced.
    audio_s: float = 0.0


@dataclass(slots=True)
class TurnTimeline:
    """Records state transitions so G1a can be asserted rather than eyeballed.

    Deliberately a plain recorder, not a state machine with legal transitions.
    The pipeline is not built yet, and inventing a transition table before the
    thing it constrains would be guessing. What is needed now is the contract for
    WHAT gets emitted and the check on the gaps between emissions.
    """

    changes: list[StateChange] = field(default_factory=list)
    # (start, end) windows during which audio is playing, in monotonic seconds.
    audio: list[tuple[float, float]] = field(default_factory=list)
    _clock: object = field(default=time.monotonic, repr=False)
    _playback_until: float = 0.0

    def emit(self, state: TurnState, detail: str = "") -> StateChange:
        change = StateChange(state=state, at=self._clock(), detail=detail)  # type: ignore[operator]
        self.changes.append(change)
        return change

    def finish(self) -> None:
        """Close the turn so the final state's duration is bounded."""
        self.emit(TurnState.IDLE, "turn complete")

    def add_audio(self, seconds: float) -> None:
        """Record `seconds` of audio as delivered now. Called by the probe per frame.

        Audio arrives in bursts -- VOICEVOX returns a whole sentence at once, and a
        filled pause is pushed as ~70 frames in one go -- but it *plays*
        sequentially. So each clip is queued after whatever is already playing
        rather than treated as covering the instant it arrived, which would let one
        burst appear to cover a gap several times over.
        """
        now = float(self._clock())  # type: ignore[operator]
        start = max(now, self._playback_until)
        self._playback_until = start + seconds
        self.audio.append((start, self._playback_until))
        if self.changes:
            self.changes[-1].audio_s += seconds

    def audio_covering(self, start: float, end: float) -> float:
        """Seconds of `[start, end)` during which something was audible."""
        return sum(max(0.0, min(end, a_end) - max(start, a_start)) for a_start, a_end in self.audio)

    @property
    def gaps(self) -> list[tuple[TurnState, float]]:
        """Seconds spent in each state, from its emission to the next one."""
        return [
            (a.state, b.at - a.at) for a, b in zip(self.changes, self.changes[1:], strict=False)
        ]

    @property
    def silent_gaps(self) -> list[tuple[TurnState, float]]:
        """Per stretch between state changes, the seconds with nothing to hear or see.

        No state is exempt. What buys time is audio that was actually playing during
        the stretch -- whatever the pipeline was calling its state at the time.
        """
        out: list[tuple[TurnState, float]] = []
        for a, b in zip(self.changes, self.changes[1:], strict=False):
            span = b.at - a.at
            out.append((a.state, max(0.0, span - self.audio_covering(a.at, b.at))))
        return out

    def longest_silent_gap(self) -> tuple[TurnState, float] | None:
        """The longest stretch with no new feedback."""
        return max(self.silent_gaps, key=lambda g: g[1], default=None)

    def satisfies_g1a(self, limit: float = MAX_SILENT_GAP_S) -> bool:
        """True when nothing audible or visible was ever more than `limit` away.

        No state-based exemption -- see the module docstring on why the previous two
        versions of this were wrong in different ways.
        """
        return all(silent <= limit for _, silent in self.silent_gaps)

    def violations(self, limit: float = MAX_SILENT_GAP_S) -> list[tuple[TurnState, float]]:
        return [(s, g) for s, g in self.silent_gaps if g > limit]
