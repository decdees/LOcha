"""Did the learner actually produce the target item? (FR-8 input)

Substring matching does not work for Japanese. 食べる appears as 食べます,
食べました, 食べて -- a naive `"食べる" in text` misses every inflected form, which
is most real usage. So the text is tokenised and each token compared on both its
surface form and its dictionary lemma.

Two unidic quirks that a lemma-only comparison gets wrong:

- Orthography drifts. ご飯 lemmatises to 御飯, so the seeded content ご飯 never
  matches the lemma. It does match the surface, hence checking both.
- unidic-lite appends English to some loanword lemmas: コーヒー has lemma
  "コーヒー-coffee". The part before the hyphen is the actual lemma.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from ocha.scheduling.rating import Usage

if TYPE_CHECKING:
    from ocha.scheduling.scheduler import Item

# Derived by running every seeded item through the tagger, not guessed. Omitting
# 代名詞 silently hid 9 of the 50 items -- 私, これ, それ, あれ, ここ, どこ, 何,
# 誰, いつ -- which are the most frequent words a beginner uses.
_CONTENT_POS = ("動詞", "名詞", "代名詞", "形容詞", "副詞", "形状詞")

# Tokens that may take part in a multi-token word. お茶 and お金 tokenise as
# 接頭辞 + 名詞, so no single token's surface equals the full item; they are only
# recoverable from a span of adjacent tokens.
_SPAN_POS = (*_CONTENT_POS, "接頭辞", "接尾辞")
_MAX_SPAN = 3


@lru_cache(maxsize=1)
def _tagger() -> object:
    import fugashi

    return fugashi.Tagger()


def content_forms(text: str) -> set[str]:
    """Every surface, lemma, and multi-token span of the content words in `text`.

    Spans are needed because some entries are more than one token. They are built
    from adjacent tokens only, never from raw substrings of the text: a raw
    substring search would match 本 inside 日本, and 日本 is a single token whose
    surface is 日本, so span-building cannot produce that false positive.
    """
    words = list(_tagger()(text))  # type: ignore[operator]

    forms: set[str] = set()
    for word in words:
        if word.feature.pos1 not in _CONTENT_POS:
            continue
        forms.add(word.surface)
        lemma = getattr(word.feature, "lemma", None)
        if lemma:
            forms.add(lemma)
            # unidic-lite appends English to some loanwords: "コーヒー-coffee"
            forms.add(lemma.split("-")[0])

    for n in range(2, _MAX_SPAN + 1):
        for i in range(len(words) - n + 1):
            window = words[i : i + n]
            if all(w.feature.pos1 in _SPAN_POS for w in window):
                forms.add("".join(w.surface for w in window))

    return forms


def produced(item: Item, text: str, forms: set[str] | None = None) -> bool:
    f = forms if forms is not None else content_forms(text)
    return item.content in f or (item.reading is not None and item.reading in f)


@dataclass(frozen=True, slots=True)
class UsageReport:
    usage: dict[int, Usage]
    elicited: frozenset[int]

    def __len__(self) -> int:
        return len(self.usage)


def detect_usage(
    targets: list[Item],
    user_text: str,
    previous_tutor_text: str | None,
) -> UsageReport:
    """Classify each target item for this turn.

    FR-8's "avoided" needs care. Read literally as "was in the target list and
    not used", it would rate five items Again on every turn -- a 1-2 sentence
    reply cannot exercise six targets, and the scheduler would be wrecked by
    normal conversation.

    So avoidance requires the item to have been ELICITED: the tutor's previous
    reply actually used it, putting it in play, and the learner did not take it
    up. That also gives the hinted/unprompted split its meaning:

      unprompted -- produced without the tutor having shown it (spontaneous)
      hinted     -- produced after the tutor showed it
      avoided    -- the tutor showed it, the learner did not produce it

    An item never surfaced by the tutor and not used by the learner is not
    evidence of anything, so it is left unscored rather than punished.
    """
    user_forms = content_forms(user_text)
    tutor_forms = content_forms(previous_tutor_text) if previous_tutor_text else set()

    usage: dict[int, Usage] = {}
    elicited: set[int] = set()

    for item in targets:
        was_elicited = bool(tutor_forms) and produced(item, "", tutor_forms)
        was_produced = produced(item, "", user_forms)
        if was_elicited:
            elicited.add(item.id)

        if was_produced and was_elicited:
            usage[item.id] = Usage.HINTED
        elif was_produced:
            usage[item.id] = Usage.UNPROMPTED
        elif was_elicited:
            usage[item.id] = Usage.AVOIDED
        # else: not in play, not used -- no evidence, no rating

    return UsageReport(usage=usage, elicited=frozenset(elicited))
