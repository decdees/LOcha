"""The grammar firewall (PRD FR-5). Critical path.

Generation for fluency, lookup for correctness.

When the model emits [GRAMMAR_QUERY], its own text must never reach the user. A
4-bit local model will confidently give a wrong explanation of は vs が, and an
absolute beginner cannot detect it -- that is the highest-risk failure mode in
the system.

Two hardenings that came out of T0.5 rather than from the spec:

1. The sentinel merely being PRESENT is not the trigger condition to test for.
   With its reasoning channel enabled, Gemma 4 emitted the sentinel alongside
   399 characters of grammar explanation. A `SENTINEL in output` check would pass
   that and leak exactly what this exists to block. So: if the sentinel appears
   at all, the ENTIRE model output is discarded, clean or not.

2. Suppressed model text is never carried on the object the API serialises.
   GrammarResponse structurally has no field for it -- the invariant is enforced
   by the type, not by remembering to strip it.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Literal

from ocha.tutor.grammar import NOT_DOCUMENTED, GrammarEntry, GrammarReference

SENTINEL = "[GRAMMAR_QUERY]"

# Detection matches the bare TOKEN, not the bracketed literal.
#
# Observed live: asked "What is the difference between は and が?", the model
# emitted "[GRAMGRAMMAR_QUERY]" -- a corrupted sentinel. Requiring the exact
# literal meant the firewall did not fire, and the mangled sentinel was passed
# through to the user AS A TUTOR REPLY. FR-5's safety property held (no grammar
# explanation leaked) but the learner got garbage instead of an answer.
#
# Matching the token is safe in the other direction too: "GRAMMAR_QUERY" is not
# Japanese and is not something a tutor reply would ever contain, so firing on it
# cannot suppress legitimate output. Fail toward suppression.
SENTINEL_TOKEN = "GRAMMAR_QUERY"

# Anything this short alongside the sentinel is whitespace or stray punctuation,
# not an explanation. Above it, the model tried to answer and we say so.
_DIRT_TOLERANCE = 4

# Question -> grammar id. Deterministic, inspectable, and never model-assisted:
# resolving the question with the LLM would reintroduce the model into the
# correctness path through the back door.
TRIGGERS: dict[str, frozenset[str]] = {
    "particle_wa_ga": frozenset({"は", "が", "wa", "ga", "topic", "subject"}),
    "particle_wa_mo": frozenset({"は", "も", "wa", "mo", "also", "too"}),
    "particle_wo": frozenset({"を", "wo", "object"}),
    "particle_ni_de": frozenset({"に", "で", "ni", "de", "location"}),
    "particle_no": frozenset({"の", "no", "possession", "possessive"}),
    "particle_to_ya": frozenset({"と", "や", "to", "ya", "and", "list"}),
    "transitive_intransitive": frozenset({"transitive", "intransitive", "他動詞", "自動詞"}),
    "te_form": frozenset({"て-form", "te-form", "て形", "te form"}),
    "politeness_register": frozenset(
        {"polite", "plain", "register", "politeness", "です", "ます", "casual", "formal"}
    ),
    "counters": frozenset({"counter", "counters", "count", "助数詞", "一つ", "枚", "本"}),
    "adjective_types": frozenset({"adjective", "い-adjective", "な-adjective", "形容詞"}),
    "question_ka": frozenset({"か", "ka", "question"}),
    "negation_polite": frozenset({"negation", "negative", "ません", "not"}),
    "past_polite": frozenset({"past", "past tense", "ました", "でした"}),
    "existence_aru_iru": frozenset({"ある", "いる", "aru", "iru", "exist", "existence"}),
    "word_order_sov": frozenset({"word order", "sov", "語順"}),
    "pro_drop": frozenset({"pronoun", "drop", "omit", "subject omission"}),
    "kudasai_requests": frozenset({"ください", "kudasai", "request", "please"}),
    "permission_temoii": frozenset({"permission", "てもいい", "may i", "allowed"}),
    "pitch_accent_basics": frozenset({"pitch", "accent", "アクセント", "intonation", "stress"}),
}


@dataclass(frozen=True, slots=True)
class GrammarResponse:
    """What the user gets when the firewall fires.

    Deliberately has NO field for the model's suppressed text. Nothing that is
    serialised toward the user can carry it, so a future edit cannot leak it by
    forgetting to strip a field.
    """

    kind: Literal["grammar", "not_documented"]
    text: str
    entry_id: str | None = None
    examples: tuple[str, ...] = ()
    hindi_contrast: str | None = None
    interference_warning: bool = False


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """Either a normal reply or a firewalled grammar answer -- never both."""

    reply: str | None = None
    grammar: GrammarResponse | None = None
    # Diagnostics only. The API layer must never serialise these.
    _suppressed: str | None = field(default=None, repr=False)
    _dirty_sentinel: bool = field(default=False, repr=False)

    @property
    def fired(self) -> bool:
        return self.grammar is not None


def sentinel_present(model_output: str) -> bool:
    """True if the model tried to emit the sentinel, corrupted or not."""
    return SENTINEL_TOKEN in model_output


def sentinel_is_clean(model_output: str) -> bool:
    """True when the sentinel is effectively the whole payload.

    A dirty sentinel is not a softer failure -- it is the model trying to answer
    a grammar question, which FR-5 forbids outright. It is reported so the cause
    (reasoning channel left on, prompt drift) can be fixed, not tolerated.
    """
    if not sentinel_present(model_output):
        return False
    if SENTINEL not in model_output:
        return False  # token present but the literal is malformed -> dirty
    return len(model_output.replace(SENTINEL, "").strip()) <= _DIRT_TOLERANCE


def _matches(term: str, haystack: str) -> bool:
    """Substring for Japanese, whole-word for romaji.

    The short romaji triggers -- wo, no, to, ka, de, ni, mo, ga -- occur inside
    ordinary English words. "How does the て-form work?" matched particle_wo via
    the "wo" in "work", tying with te_form and resolving to nothing. Japanese has
    no word boundaries, so those terms stay substring matches.
    """
    if term.isascii():
        return re.search(rf"\b{re.escape(term)}\b", haystack) is not None
    return term in haystack


def resolve_entry_id(user_text: str, reference: GrammarReference) -> str | None:
    """Map a grammar question to a reference id, or None.

    Scores by how many distinct trigger terms match, so "difference between は
    and も" resolves to particle_wa_mo rather than particle_wa_ga even though
    both contain は. A tie is treated as unresolved: serving a plausible-looking
    wrong entry is worse than admitting the gap.
    """
    haystack = user_text.lower()
    scores: dict[str, int] = {}
    for entry_id, terms in TRIGGERS.items():
        if entry_id not in reference:
            continue
        hits = sum(1 for t in terms if _matches(t.lower(), haystack))
        if hits:
            scores[entry_id] = hits
    if not scores:
        return None
    best = max(scores.values())
    winners = [k for k, v in scores.items() if v == best]
    return winners[0] if len(winners) == 1 else None


def _render(entry: GrammarEntry) -> GrammarResponse:
    return GrammarResponse(
        kind="grammar",
        text=entry.en,
        entry_id=entry.id,
        examples=tuple(entry.examples),
        hindi_contrast=entry.hindi_contrast,
        interference_warning=entry.interference_warning,
    )


def log_unauthored(conn: sqlite3.Connection, user_text: str, entry_id: str | None) -> None:
    conn.execute(
        "INSERT INTO unauthored_grammar (user_text, item_id) VALUES (?, ?)",
        (user_text, entry_id),
    )


def apply_firewall(
    model_output: str,
    user_text: str,
    reference: GrammarReference,
    conn: sqlite3.Connection | None = None,
) -> TurnOutcome:
    """The only path by which model output becomes a user-facing response.

    If the sentinel appears anywhere, the model's entire output is discarded and
    the answer comes from the curated reference or not at all. There is no
    branch that returns model text alongside a grammar answer.
    """
    if not sentinel_present(model_output):
        return TurnOutcome(reply=model_output)

    dirty = not sentinel_is_clean(model_output)
    entry_id = resolve_entry_id(user_text, reference)
    entry = reference.get(entry_id) if entry_id else None

    if entry is None:
        if conn is not None:
            log_unauthored(conn, user_text, entry_id)
        return TurnOutcome(
            grammar=GrammarResponse(kind="not_documented", text=NOT_DOCUMENTED, entry_id=entry_id),
            _suppressed=model_output,
            _dirty_sentinel=dirty,
        )

    return TurnOutcome(
        grammar=_render(entry),
        _suppressed=model_output,
        _dirty_sentinel=dirty,
    )
