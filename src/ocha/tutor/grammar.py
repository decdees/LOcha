"""Curated grammar reference (PRD FR-7).

This file is the ONLY source of grammar explanations shown to the user. The
firewall (FR-5) serves from here or returns "not yet documented" -- it never
falls back to generation, because a 4-bit local model will confidently explain
は vs が wrongly and the learner cannot detect it.

Validation is fail-fast at startup: a malformed reference is a correctness bug,
and discovering it mid-conversation means the firewall has nothing to serve.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_PATH = Path("data/grammar.json")

NOT_DOCUMENTED = "This grammar point is not yet documented."


class GrammarEntry(BaseModel):
    # Reject unknown keys: a typo'd field name would otherwise be silently
    # dropped, and a missing hindi_contrast is indistinguishable from a
    # misspelled one.
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    en: str = Field(min_length=1)
    examples: list[str] = Field(min_length=1)
    source: str = Field(min_length=1)
    interference_warning: bool
    # Optional, and only present where the analogy genuinely holds (FR-7).
    hindi_contrast: str | None = None


class GrammarReference:
    """Loaded, validated reference. Lookup only -- never generation."""

    def __init__(self, entries: list[GrammarEntry]) -> None:
        self._by_id: dict[str, GrammarEntry] = {}
        for e in entries:
            if e.id in self._by_id:
                raise ValueError(f"duplicate grammar id: {e.id}")
            self._by_id[e.id] = e

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._by_id

    def get(self, entry_id: str) -> GrammarEntry | None:
        """Return the entry, or None on a miss. A miss is NOT an error -- the
        caller must serve NOT_DOCUMENTED and log it, never generate."""
        return self._by_id.get(entry_id)

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    @property
    def interference_ids(self) -> list[str]:
        """Entries where Hindi intuition actively misleads. Surfaced prominently."""
        return [e.id for e in self._by_id.values() if e.interference_warning]


def load_grammar(path: Path | str = DEFAULT_PATH) -> GrammarReference:
    """Load and validate. Raises on any malformed entry -- fail fast, loudly."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "entries" not in raw:
        raise ValueError(f"{p}: missing top-level 'entries'")

    entries: list[GrammarEntry] = []
    problems: list[str] = []
    for i, item in enumerate(raw["entries"]):
        try:
            entries.append(GrammarEntry.model_validate(item))
        except ValidationError as exc:
            got = item.get("id", f"<index {i}>") if isinstance(item, dict) else f"<index {i}>"
            problems.append(f"  {got}: {exc.error_count()} error(s) -- {exc.errors()[0]['msg']}")

    if problems:
        raise ValueError(f"{p}: {len(problems)} malformed entr(ies)\n" + "\n".join(problems))
    return GrammarReference(entries)
