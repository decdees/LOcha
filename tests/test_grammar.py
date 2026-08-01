from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ocha.tutor.grammar import NOT_DOCUMENTED, GrammarReference, load_grammar

VALID = {
    "id": "particle_x",
    "en": "An explanation.",
    "examples": ["例です。"],
    "source": "Tae Kim §1",
    "interference_warning": False,
}


def write(tmp_path: Path, *entries: dict[str, object]) -> Path:
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"entries": list(entries)}, ensure_ascii=False), encoding="utf-8")
    return p


# ---- the real reference ---------------------------------------------------


def test_shipped_reference_loads() -> None:
    g = load_grammar()
    assert len(g) == 20


def test_shipped_reference_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert len(load_grammar()) == 20


def test_shipped_reference_covers_the_probe_topics() -> None:
    """T1.4: seeded with entries covering the T0.5 probe topics."""
    g = load_grammar()
    for needed in (
        "particle_wa_ga",
        "particle_wa_mo",
        "transitive_intransitive",
        "counters",
        "te_form",
        "politeness_register",
    ):
        assert needed in g, needed


def test_required_interference_warnings_are_set() -> None:
    """T1.4 names three specifically: ne/ga, gender agreement, stress-vs-pitch."""
    g = load_grammar()
    flagged = set(g.interference_ids)
    assert "particle_wa_ga" in flagged  # は/が vs Hindi ने
    assert "particle_no" in flagged  # の does not agree for gender/number
    assert "pitch_accent_basics" in flagged  # Hindi stress vs Japanese pitch


def test_hindi_contrast_is_optional_not_universal() -> None:
    """FR-7: present only where the analogy genuinely holds. If every entry had
    one, that would mean they were being invented."""
    g = load_grammar()
    with_contrast = [i for i in g.ids if g.get(i) and g.get(i).hindi_contrast]  # type: ignore[union-attr]
    assert 0 < len(with_contrast) < len(g)


# ---- fail-fast validation -------------------------------------------------


def test_loader_accepts_a_valid_entry(tmp_path: Path) -> None:
    assert len(load_grammar(write(tmp_path, VALID))) == 1


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("id", ""),  # empty
        ("id", "Particle-X"),  # uppercase and hyphen violate the pattern
        ("en", ""),  # no explanation is worse than no entry
        ("examples", []),  # an entry with no example teaches nothing
        ("source", ""),  # unsourced grammar is exactly what FR-7 forbids
    ],
)
def test_loader_rejects_malformed_entries(tmp_path: Path, field: str, bad: object) -> None:
    entry = VALID | {field: bad}
    with pytest.raises(ValueError, match="malformed"):
        load_grammar(write(tmp_path, entry))


def test_loader_rejects_missing_required_field(tmp_path: Path) -> None:
    entry = {k: v for k, v in VALID.items() if k != "interference_warning"}
    with pytest.raises(ValueError, match="malformed"):
        load_grammar(write(tmp_path, entry))


def test_loader_rejects_unknown_field(tmp_path: Path) -> None:
    """extra='forbid': a typo'd key would otherwise be silently dropped, making a
    misspelled hindi_contrast indistinguishable from an absent one."""
    with pytest.raises(ValueError, match="malformed"):
        load_grammar(write(tmp_path, VALID | {"hindi_contrst": "typo"}))


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_grammar(write(tmp_path, VALID, dict(VALID)))


def test_loader_rejects_file_without_entries(tmp_path: Path) -> None:
    p = tmp_path / "g.json"
    p.write_text('{"nope": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="entries"):
        load_grammar(p)


def test_error_names_every_bad_entry(tmp_path: Path) -> None:
    """A reference with three problems should report three, not just the first."""
    with pytest.raises(ValueError, match="3 malformed"):
        load_grammar(
            write(
                tmp_path,
                VALID | {"id": "a", "en": ""},
                VALID | {"id": "b", "examples": []},
                VALID | {"id": "c", "source": ""},
            )
        )


# ---- lookup contract -----------------------------------------------------


def test_miss_returns_none_and_never_raises() -> None:
    """A miss is not an error. The caller serves NOT_DOCUMENTED and logs it --
    it must never fall back to generation (FR-5)."""
    g = load_grammar()
    assert g.get("does_not_exist") is None
    assert NOT_DOCUMENTED  # the string the firewall serves instead


def test_entries_are_immutable() -> None:
    """The reference is curated data. Nothing downstream may edit it in place --
    a mutated explanation is indistinguishable from a generated one."""
    g = load_grammar()
    entry = g.get("particle_wa_ga")
    assert entry is not None
    with pytest.raises(ValidationError):
        entry.en = "tampered"


def test_empty_reference_is_allowed_but_serves_nothing() -> None:
    g = GrammarReference([])
    assert len(g) == 0
    assert g.get("anything") is None
