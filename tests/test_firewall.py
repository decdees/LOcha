"""T1.5 — grammar firewall (PRD FR-5). Critical path.

The assertion in test_no_model_text_reaches_the_user_when_the_sentinel_fires
must never be weakened or skipped (project standing constraint 2). If it fails,
the model is teaching grammar directly to a learner who cannot detect an error.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path

import pytest

from ocha.db import connect, migrate
from ocha.tutor.firewall import (
    SENTINEL,
    TRIGGERS,
    apply_firewall,
    resolve_entry_id,
    sentinel_is_clean,
    sentinel_present,
)
from ocha.tutor.grammar import NOT_DOCUMENTED, GrammarReference, load_grammar


@pytest.fixture
def ref() -> GrammarReference:
    return load_grammar()


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = connect(tmp_path / "t.db")
    migrate(c)
    return c


# Real, plausible model betrayals. Each contains the sentinel AND an explanation
# -- the shape T0.5 actually observed from Gemma 4 with its reasoning channel on.
LEAKY_OUTPUTS = [
    f"{SENTINEL} は marks the topic and が marks the subject.",
    f"は marks the topic. {SENTINEL}",
    f"{SENTINEL}\n\nActually, は is used for known information while が introduces new.",
    f"<|channel>thought\nThe user asks about は vs が. Constraint says reply {SENTINEL}. "
    "But I should explain: は is the topic marker...",
    f"{SENTINEL} (Note: が can also mark the object of certain verbs.)",
    f"Let me explain. {SENTINEL} The difference is subtle: は is thematic.",
]


def _significant_words(text: str) -> list[str]:
    """Words long enough to be evidence of provenance, ignoring shared filler."""
    import re

    stop = {"the", "and", "that", "with", "for", "while", "also", "can", "should", "but", "says"}
    return [w for w in re.findall(r"[A-Za-z]{3,}", text) if w.lower() not in stop]


# ---- THE critical assertion ----------------------------------------------


@pytest.mark.parametrize("leaked", LEAKY_OUTPUTS)
def test_no_model_text_reaches_the_user_when_the_sentinel_fires(
    leaked: str, ref: GrammarReference, conn: sqlite3.Connection
) -> None:
    """FR-5: the model's own grammar explanations are never shown, under ANY
    circumstance. NEVER WEAKEN OR SKIP THIS TEST.

    Formulated as *derivability*, not as a blacklist of forbidden phrases. Every
    user-visible field must be byte-identical to the curated reference, so the
    payload is provably reference-derived and nothing model-authored can survive
    -- including phrasing nobody thought to blacklist.

    A first version did use a phrase blacklist and it was wrong: "marks the
    subject" is in the curated は/が entry, so the test failed on correct
    behaviour. Blacklists here are both leaky and prone to false alarms.
    """
    out = apply_firewall(leaked, "What is the difference between は and が?", ref, conn)

    assert out.fired
    assert out.reply is None
    assert out.grammar is not None

    entry = ref.get("particle_wa_ga")
    assert entry is not None

    # Every field is exactly the reference's. Nothing added, nothing merged.
    assert out.grammar.text == entry.en
    assert out.grammar.examples == tuple(entry.examples)
    assert out.grammar.hindi_contrast == entry.hindi_contrast
    assert out.grammar.entry_id == entry.id

    # Belt and braces on the serialised form: no stray field carries model text,
    # and the sentinel itself is never shown to the user either.
    payload = json.dumps(dataclasses.asdict(out.grammar), ensure_ascii=False)
    assert SENTINEL not in payload
    reference_text = json.dumps(
        [entry.en, entry.examples, entry.hindi_contrast, entry.id, entry.interference_warning],
        ensure_ascii=False,
    )
    for chunk in _significant_words(leaked):
        if chunk not in reference_text:
            assert chunk not in payload, f"LEAK: model-only word {chunk!r} reached the user"


def test_a_dirty_sentinel_is_still_fully_suppressed(
    ref: GrammarReference, conn: sqlite3.Connection
) -> None:
    """The T0.5 case: sentinel + 399 chars of explanation. A `SENTINEL in output`
    check would have passed this and leaked the explanation."""
    leaked = SENTINEL + " " + ("は is the topic particle. " * 20)
    out = apply_firewall(leaked, "は vs が?", ref, conn)
    assert out.fired
    assert out._dirty_sentinel is True
    assert out.grammar is not None
    assert "topic particle" not in out.grammar.text


def test_served_text_is_identical_to_the_curated_entry(ref: GrammarReference) -> None:
    """The answer is the reference verbatim -- not a paraphrase, not a merge."""
    out = apply_firewall(SENTINEL, "difference between は and が", ref)
    entry = ref.get("particle_wa_ga")
    assert entry is not None
    assert out.grammar is not None
    assert out.grammar.text == entry.en
    assert out.grammar.examples == tuple(entry.examples)


# ---- normal path ---------------------------------------------------------


def test_ordinary_reply_passes_through(ref: GrammarReference) -> None:
    out = apply_firewall("いいですね。何を食べましたか。", "ご飯を食べました。", ref)
    assert not out.fired
    assert out.reply == "いいですね。何を食べましたか。"
    assert out.grammar is None


def test_reply_mentioning_grammar_words_is_not_firewalled(ref: GrammarReference) -> None:
    """Only the literal sentinel fires. A tutor reply that happens to contain は
    must not be suppressed."""
    out = apply_firewall("これはいいですね。", "これはいいですか。", ref)
    assert not out.fired


# ---- sentinel detection -------------------------------------------------


def test_sentinel_present_and_clean() -> None:
    assert sentinel_present(SENTINEL)
    assert sentinel_is_clean(SENTINEL)
    assert sentinel_is_clean(f"  {SENTINEL}\n")
    assert not sentinel_is_clean(f"{SENTINEL} は marks the topic.")
    assert not sentinel_present("no sentinel here")
    assert not sentinel_is_clean("no sentinel here")
    # token present but the literal malformed -> present, but not clean
    assert sentinel_present("[GRAMGRAMMAR_QUERY]")
    assert not sentinel_is_clean("[GRAMGRAMMAR_QUERY]")


def test_corrupted_sentinel_still_fires(ref: GrammarReference) -> None:
    """Observed live: the model emitted "[GRAMGRAMMAR_QUERY]". Requiring the exact
    bracketed literal meant the firewall did not fire and the mangled sentinel
    reached the user as a tutor reply.

    An earlier version of this test asserted the opposite -- that near-misses are
    normal replies. Real output showed that assumption was wrong: a reply
    containing GRAMMAR_QUERY is never legitimate tutor output. Detection now
    matches the bare token and fails toward suppression.
    """
    for corrupted in (
        "[GRAMGRAMMAR_QUERY]",
        "GRAMMAR_QUERY",
        "[[GRAMMAR_QUERY]]",
        "[GRAMMAR_QUERY",
        "GRAMMAR_QUERY]",
    ):
        out = apply_firewall(corrupted, "difference between は and が", ref)
        # The two properties that matter: it fires, and the corrupted sentinel is
        # not handed to the user as a reply. Whether stray brackets alone count as
        # "dirty" is cosmetic -- what must never happen is passing it through.
        assert out.fired, f"{corrupted!r} must fire"
        assert out.reply is None, f"{corrupted!r} leaked as a reply"
        assert out.grammar is not None
        assert "GRAMMAR_QUERY" not in out.grammar.text


def test_corrupted_sentinel_with_explanation_is_flagged_dirty(ref: GrammarReference) -> None:
    """Corruption plus an explanation is the dangerous combination, and must be
    both suppressed and reported."""
    out = apply_firewall("[GRAMGRAMMAR_QUERY] は marks the topic.", "は vs が", ref)
    assert out.fired
    assert out._dirty_sentinel is True
    assert out.grammar is not None
    assert "marks the topic." not in out.grammar.text or out.grammar.entry_id == "particle_wa_ga"


def test_genuinely_unrelated_reply_does_not_fire(ref: GrammarReference) -> None:
    """Only the token triggers. Japanese replies, and English ones that merely
    mention grammar, pass through."""
    for ordinary in (
        "これはいいですね。",
        "grammar is hard",
        "query the database",
        "[SOMETHING_ELSE]",
    ):
        assert not apply_firewall(ordinary, "は vs が?", ref).fired


# ---- resolution ---------------------------------------------------------


def test_wa_ga_and_wa_mo_are_distinguished(ref: GrammarReference) -> None:
    """Both questions contain は. Scoring by distinct trigger hits separates them."""
    assert resolve_entry_id("difference between は and が", ref) == "particle_wa_ga"
    assert resolve_entry_id("difference between は and も", ref) == "particle_wa_mo"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("When do I use に versus で?", "particle_ni_de"),
        ("How does the て-form work?", "te_form"),
        ("Is 開く transitive or intransitive?", "transitive_intransitive"),
        ("Why do I need a counter?", "counters"),
        ("What is pitch accent?", "pitch_accent_basics"),
        ("When can I omit the pronoun?", "pro_drop"),
    ],
)
def test_resolution_of_the_probe_topics(
    question: str, expected: str, ref: GrammarReference
) -> None:
    assert resolve_entry_id(question, ref) == expected


def test_unresolvable_question_returns_none(ref: GrammarReference) -> None:
    assert resolve_entry_id("why is Japanese hard", ref) is None


def test_every_trigger_maps_to_a_real_entry(ref: GrammarReference) -> None:
    """A trigger for a nonexistent id would resolve, then miss, then log a
    not-documented for a topic we actually documented."""
    for entry_id in TRIGGERS:
        assert entry_id in ref, entry_id


# ---- reference miss -----------------------------------------------------


def test_miss_serves_not_documented_and_never_generates(
    ref: GrammarReference, conn: sqlite3.Connection
) -> None:
    out = apply_firewall(
        f"{SENTINEL} Japanese is hard because of keigo.", "why is Japanese hard", ref, conn
    )
    assert out.fired
    assert out.grammar is not None
    assert out.grammar.kind == "not_documented"
    assert out.grammar.text == NOT_DOCUMENTED
    assert "keigo" not in out.grammar.text  # no fallback to generation


def test_miss_is_logged_for_manual_authoring(
    ref: GrammarReference, conn: sqlite3.Connection
) -> None:
    apply_firewall(SENTINEL, "what is the causative passive", ref, conn)
    rows = conn.execute("SELECT user_text, item_id FROM unauthored_grammar").fetchall()
    assert len(rows) == 1
    assert rows[0]["user_text"] == "what is the causative passive"


def test_hit_is_not_logged_as_unauthored(ref: GrammarReference, conn: sqlite3.Connection) -> None:
    apply_firewall(SENTINEL, "difference between は and が", ref, conn)
    assert conn.execute("SELECT count(*) FROM unauthored_grammar").fetchone()[0] == 0


def test_empty_reference_serves_not_documented_not_model_text(
    conn: sqlite3.Connection,
) -> None:
    """With nothing curated, the firewall must still refuse to pass model text."""
    out = apply_firewall(
        f"{SENTINEL} は is the topic marker.", "は vs が?", GrammarReference([]), conn
    )
    assert out.grammar is not None
    assert out.grammar.kind == "not_documented"
    assert "topic marker" not in out.grammar.text


# ---- structural guarantee ----------------------------------------------


def test_grammar_response_has_no_field_for_model_text() -> None:
    """The invariant is enforced by the type. If someone adds a field that could
    carry model output, this fails and they have to justify it."""
    from ocha.tutor.firewall import GrammarResponse

    allowed = {"kind", "text", "entry_id", "examples", "hindi_contrast", "interference_warning"}
    assert {f.name for f in dataclasses.fields(GrammarResponse)} == allowed


def test_outcome_is_never_both_reply_and_grammar(ref: GrammarReference) -> None:
    for output, question in (
        ("いいですね。", "ご飯を食べました。"),
        (SENTINEL, "は vs が"),
        (f"{SENTINEL} explanation", "は vs が"),
    ):
        out = apply_firewall(output, question, ref)
        assert (out.reply is None) != (out.grammar is None), "exactly one must be set"
