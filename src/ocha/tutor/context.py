"""Context Builder (ARCHITECTURE §7.1, T1.6).

Assembles the system prompt from live FSRS state. The vocabulary list and target
items are constraints, not suggestions -- FR-3 caps the reply at 1-2 sentences and
at most one new word, glossed.

Two lines here are load-bearing and were established by measurement, not design:

- REGISTER. §7.1's original template constrained vocabulary, target, length and
  grammar level, and said nothing about politeness. T0.5 measured Gemma mixing
  polite and plain inside single replies on 6 of 15 turns
  ("いいですね。あなたは今日、何を食べる？"), which models inconsistent politeness
  to a learner who cannot detect it. Adding this line took register compliance
  from 9/15 to 14/15 and vocabulary from 14/15 to 15/15.

- The context budget. §4 originally said cap at 8k. T0.4 measured time-to-first-
  token at 32.6 s and decode at 14.3 tok/s at 8k -- slower than the dense model
  the MoE was chosen over. Revised to ~2k, where decode holds ~36 tok/s. A
  10-turn conversation measured 560 tokens, so 2k is ample and the cap is a
  safety net rather than a constraint on the tutor.
"""

from __future__ import annotations

from dataclasses import dataclass

from ocha.scheduling.scheduler import Item, ItemScheduler
from ocha.tutor.firewall import SENTINEL

# T0.4: not 8k. See module docstring.
MAX_CONTEXT_TOKENS = 2048
# FR-3: enforced by prompt AND max_tokens, because prompts get ignored.
MAX_REPLY_TOKENS = 64

LEVEL = "beginner"

# A 1-2 sentence reply that may introduce at most one new word cannot meaningfully
# steer toward eight of them. Listing more just dilutes the instruction.
MAX_INTRODUCE = 3

# §7.1's template is corrected here, not copied. Rendering it against real FSRS
# state produced a self-contradictory prompt: VOCABULARY said "use only words
# from KNOWN" while TARGET listed due items, and due items are typically NOT
# known -- they are new or lapsed, which is why they came due. The model was
# being told to use only known words and to steer toward unknown ones in the same
# breath. TARGET is therefore split by whether the item is already known, which
# encodes FR-3's "at most one new word per turn, glossed" explicitly instead of
# leaving the model to reconcile it.
TEMPLATE = """You are a Japanese conversation partner. Reply in 1-2 short sentences.

VOCABULARY: Use only words from KNOWN, plus at most one NEW word per reply.
Never gloss a word that is already in KNOWN -- the learner knows it.
KNOWN: {known}

PRACTISE (already known, needs work -- use these freely):
{practise}

INTRODUCE (not yet known -- use AT MOST ONE per reply, glossed in English):
{introduce}

REGISTER: Always use polite です/ます form. Never mix polite and plain forms.

AVOID: Do not use grammar beyond {level}.

Never break character to explain grammar. If asked a grammar question,
respond with exactly: {sentinel}"""


@dataclass(frozen=True, slots=True)
class TurnContext:
    """A built prompt plus the state that produced it.

    The item ids are carried so T1.8 can score the turn against the same targets
    the model was actually given, rather than re-querying and drifting.
    """

    system_prompt: str
    known_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    weak_ids: tuple[int, ...]
    history: tuple[tuple[str, str], ...] = ()

    @property
    def target_contents(self) -> tuple[str, ...]:
        return tuple(self._targets)

    _targets: tuple[str, ...] = ()


def _render_known(items: list[Item]) -> str:
    """Content plus reading, but only where the reading adds information.

    A kana word's reading equals its content, and repeating it doubles the
    KNOWN list for nothing -- and KNOWN dominates prompt length, which drives
    prefill, which T0.4 showed is the dominant latency term.
    """
    if not items:
        return "(none yet)"
    parts = [
        f"{i.content}({i.reading})" if i.reading and i.reading != i.content else i.content
        for i in items
    ]
    return "、".join(parts)


def _render_target(items: list[Item], empty: str) -> str:
    if not items:
        return empty
    return "、".join(f"{i.content} = {i.meaning_en}" for i in items)


def build_context(
    scheduler: ItemScheduler,
    *,
    known_limit: int = 60,
    min_reps: int = 3,
    due_limit: int = 5,
    weak_limit: int = 3,
    history: tuple[tuple[str, str], ...] = (),
) -> TurnContext:
    """Assemble the system prompt from current FSRS state.

    known_limit is a real constraint, not arbitrary: the KNOWN list dominates
    prompt length, and prompt length drives prefill, which T0.4 showed is the
    dominant latency term. 60 items is roughly 400 tokens.
    """
    known = scheduler.known_items(min_reps=min_reps, limit=known_limit)
    due = scheduler.due_items(limit=due_limit)
    weak = scheduler.lowest_stability(limit=weak_limit)

    known_ids = {i.id for i in known}

    # Merge due and weak, de-duplicated. Weak items are worth steering toward even
    # when not strictly due -- that is the whole point of tracking stability.
    target: list[Item] = []
    seen: set[int] = set()
    for item in [*due, *weak]:
        if item.id not in seen:
            target.append(item)
            seen.add(item.id)

    # Split by whether the learner already has the word. An item that is known
    # but weak is safe to use freely; one that is not known can only be
    # introduced, and only one per reply (FR-3).
    practise = [i for i in target if i.id in known_ids]
    introduce = [i for i in target if i.id not in known_ids][:MAX_INTRODUCE]

    prompt = TEMPLATE.format(
        known=_render_known(known),
        practise=_render_target(practise, "(nothing -- converse freely within KNOWN)"),
        introduce=_render_target(introduce, "(nothing new this turn)"),
        level=LEVEL,
        sentinel=SENTINEL,
    )

    return TurnContext(
        system_prompt=prompt,
        known_ids=tuple(i.id for i in known),
        target_ids=tuple(i.id for i in [*practise, *introduce]),
        weak_ids=tuple(i.id for i in weak),
        history=history,
        _targets=tuple(i.content for i in [*practise, *introduce]),
    )
