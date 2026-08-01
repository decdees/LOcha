"""T1.8 — turn orchestration, including the 10-turn integration test."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fsrs import Rating

from ocha.db import connect, migrate
from ocha.db.seed import seed
from ocha.scheduling import ItemScheduler
from ocha.scheduling.rating import Usage
from ocha.tutor.firewall import SENTINEL
from ocha.tutor.grammar import load_grammar
from ocha.tutor.llm import StubLlm
from ocha.tutor.turn import ensure_session, run_turn
from ocha.tutor.usage import content_forms, detect_usage, produced


@pytest.fixture
def env(tmp_path: Path) -> tuple[sqlite3.Connection, ItemScheduler]:
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    seed(conn)
    return conn, ItemScheduler(conn)


def _known(sched: ItemScheduler, n: int) -> list[int]:
    ids = [i.id for i in sched.due_items(limit=n)]
    for i in ids:
        for _ in range(3):
            sched.record_review(i, Rating.Good)
    return ids


# ---- usage detection ----------------------------------------------------


def test_inflected_verbs_are_matched(env: tuple[sqlite3.Connection, ItemScheduler]) -> None:
    """Substring matching would miss 食べました for target 食べる, which is most
    real usage."""
    _, sched = env
    item = next(i for i in sched.due_items(limit=50) if i.content == "食べる")
    assert produced(item, "ご飯を食べました。")
    assert produced(item, "食べています。")
    assert not produced(item, "水を飲みました。")


def test_orthography_drift_is_handled(env: tuple[sqlite3.Connection, ItemScheduler]) -> None:
    """ご飯 lemmatises to 御飯, so a lemma-only comparison misses the seeded form."""
    _, sched = env
    item = next(i for i in sched.due_items(limit=50) if i.content == "ご飯")
    assert produced(item, "ご飯を食べました。")


def test_loanword_lemma_suffix_is_stripped(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    """unidic-lite gives コーヒー the lemma 'コーヒー-coffee'."""
    _, sched = env
    item = next(i for i in sched.due_items(limit=50) if i.content == "コーヒー")
    assert produced(item, "コーヒーを飲みます。")
    assert "コーヒー" in content_forms("コーヒーを飲みます。")


def test_every_seeded_item_is_detectable() -> None:
    """The systematic guard. Spot checks missed two whole classes of failure:

    - 代名詞 was absent from the content-POS list, hiding 9 of 50 items
      (私, これ, それ, あれ, ここ, どこ, 何, 誰, いつ) -- the words a beginner
      uses most.
    - お茶 and お金 tokenise as 接頭辞 + 名詞, so no single token's surface equals
      the item and they were unrecoverable without multi-token spans.

    Both were invisible to hand-picked examples and obvious to an exhaustive one.
    """
    from ocha.db.seed import VOCAB

    undetectable = []
    for content, _reading, _meaning in VOCAB:
        if not any(
            content in content_forms(t.format(content))
            for t in ("{}です。", "{}を見ました。", "{}がいいです。", "{}ました。")
        ):
            undetectable.append(content)
    assert undetectable == [], undetectable


def test_spans_do_not_create_substring_false_positives() -> None:
    """本 must not match inside 日本. Spans are built from adjacent TOKENS, and
    日本 is a single token, so the unigram 本 never appears."""
    forms = content_forms("日本語がわかります。")
    assert "本" not in forms


def test_three_usage_classes(env: tuple[sqlite3.Connection, ItemScheduler]) -> None:
    _, sched = env
    items = {i.content: i for i in sched.due_items(limit=50)}
    targets = [items["食べる"], items["水"]]

    # tutor used 食べる, learner used it too -> hinted
    r = detect_usage(targets, "ご飯を食べました。", "何を食べますか。")
    assert r.usage[items["食べる"].id] is Usage.HINTED

    # tutor used 食べる, learner used 水 instead -> avoided + unprompted
    r = detect_usage(targets, "水を飲みました。", "何を食べますか。")
    assert r.usage[items["食べる"].id] is Usage.AVOIDED
    assert r.usage[items["水"].id] is Usage.UNPROMPTED


def test_unelicited_and_unused_items_are_not_scored(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    """FR-8 read literally would rate five items Again every turn -- a 1-2 sentence
    reply cannot exercise six targets. Avoidance requires the tutor to have put
    the item in play."""
    _, sched = env
    targets = sched.due_items(limit=6)
    r = detect_usage(targets, "はい。", "そうですか。")
    assert r.usage == {}


# ---- orchestration ------------------------------------------------------


def test_turn_creates_a_session_and_persists(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    conn, sched = env
    res = run_turn(conn, sched, load_grammar(), StubLlm(), "こんにちは。")
    assert res.session_id > 0
    row = conn.execute("SELECT * FROM turns WHERE id = ?", (res.turn_id,)).fetchone()
    assert row["user_text"] == "こんにちは。"
    assert row["grammar_query"] == 0


def test_unknown_session_raises(env: tuple[sqlite3.Connection, ItemScheduler]) -> None:
    conn, sched = env
    with pytest.raises(KeyError):
        ensure_session(conn, 999_999)


def test_grammar_turn_is_not_scored(env: tuple[sqlite3.Connection, ItemScheduler]) -> None:
    """A grammar question is not a production attempt. Scoring it would punish the
    learner for asking -- the opposite of what the firewall is for."""
    conn, sched = env
    _known(sched, 4)
    before = conn.execute("SELECT count(*) FROM reviews").fetchone()[0]
    res = run_turn(
        conn,
        sched,
        load_grammar(),
        StubLlm(reply=SENTINEL),
        "What is the difference between は and が?",
    )
    assert res.grammar_query
    assert res.ratings == {}
    assert conn.execute("SELECT count(*) FROM reviews").fetchone()[0] == before
    assert (
        conn.execute("SELECT grammar_query FROM turns WHERE id = ?", (res.turn_id,)).fetchone()[0]
        == 1
    )


def test_no_model_text_in_the_api_payload_on_a_grammar_turn(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    """FR-5 at the orchestration boundary, not just inside the firewall."""
    conn, sched = env
    leaky = f"{SENTINEL} は marks the topic, が marks the subject."
    res = run_turn(conn, sched, load_grammar(), StubLlm(reply=leaky), "は vs が?")
    assert res.reply is None
    assert res.grammar is not None
    assert SENTINEL not in res.grammar.text
    # and the stored tutor_text must not carry it either
    stored = conn.execute("SELECT tutor_text FROM turns WHERE id = ?", (res.turn_id,)).fetchone()[0]
    assert SENTINEL not in stored


def _a_target(sched: ItemScheduler) -> str:
    """Pick a word the Context Builder is actually steering toward.

    Hardcoding one couples the test to seed ordering and to which items happen to
    be due. An earlier version assumed 食べる; it is item 37 and is never in the
    target list for a freshly seeded database, so the test failed for a reason
    that had nothing to do with the code under test.
    """
    from ocha.tutor.context import build_context

    ctx = build_context(sched)
    names = {i.id: i.content for i in sched.due_items(limit=50)}
    names |= {i.id: i.content for i in sched.known_items(min_reps=1, limit=50)}
    return names[ctx.target_ids[0]]


def test_usage_produces_an_observation_without_a_rating(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    conn, sched = env
    _known(sched, 6)
    word = _a_target(sched)
    # the tutor elicits it, then the learner produces it
    first = run_turn(
        conn, sched, load_grammar(), StubLlm(reply=f"{word}はいいですね。"), "こんにちは。"
    )
    res = run_turn(
        conn,
        sched,
        load_grammar(),
        StubLlm(reply="そうですか。"),
        f"{word}が好きです。",
        session_id=first.session_id,
    )
    assert res.observations, "producing an elicited target must be observable"
    assert set(res.observations.values()) == {"mentioned_after_prompt"}
    assert res.ratings == {}
    assert res.usage == {}


def test_consecutive_turns_pass_four_explicit_complete_exchanges(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    conn, sched = env
    llm = StubLlm(reply="最初の返答です。")
    first = run_turn(conn, sched, load_grammar(), llm, "最初の質問です。")

    for n in range(1, 6):
        llm.reply = f"返答{n}です。"
        run_turn(
            conn,
            sched,
            load_grammar(),
            llm,
            f"質問{n}です。",
            session_id=first.session_id,
        )

    history = llm.histories[-1]
    assert len(history) == 8
    assert [message.role for message in history] == ["user", "assistant"] * 4
    assert history[0].content == "質問1です。"
    assert history[-1].content == "返答4です。"
    assert all("You are a Japanese conversation partner" not in m.content for m in history)


# ---- the 10-turn integration test --------------------------------------


def test_ten_turn_free_conversation_never_evolves_fsrs_state(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    """Free conversation creates evidence, never inferred spaced-repetition reviews."""
    conn, sched = env
    ref = load_grammar()
    _known(sched, 10)

    llm = StubLlm()
    session: int | None = None
    prev_word: str | None = None
    reps_before = conn.execute("SELECT sum(reps) FROM items").fetchone()[0]

    for n in range(10):
        # Produce what the tutor elicited last turn on even turns; dodge on odd.
        user = f"{prev_word}が好きです。" if (n % 2 == 0 and prev_word) else "はい、そうですね。"
        word = _a_target(sched)
        llm.reply = f"{word}はどうですか。"
        res = run_turn(conn, sched, ref, llm, user, session_id=session)
        session = res.session_id
        prev_word = word

    assert (
        conn.execute("SELECT count(*) FROM turns WHERE session_id = ?", (session,)).fetchone()[0]
        == 10
    )

    # FSRS state did not move: only a validated drill may rate recall.
    reps_after = conn.execute("SELECT sum(reps) FROM items").fetchone()[0]
    assert reps_after == reps_before

    # Reviews are attributable to the session that produced them.
    assert (
        conn.execute("SELECT count(*) FROM reviews WHERE source LIKE 'turn:%'").fetchone()[0] == 0
    )
    assert conn.execute("SELECT count(*) FROM item_observations").fetchone()[0] > 0

    # The firewall never fired -- none of these turns was a grammar question.
    assert conn.execute("SELECT sum(grammar_query) FROM turns").fetchone()[0] == 0


def test_malformed_japanese_is_only_an_observation(
    env: tuple[sqlite3.Connection, ItemScheduler],
) -> None:
    conn, sched = env
    item = conn.execute("SELECT * FROM items WHERE content = '食べる'").fetchone()
    assert item is not None
    conn.execute("UPDATE items SET due = '2000-01-01T00:00:00+00:00' WHERE id = ?", (item["id"],))
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert item is not None
    before = tuple(item[key] for key in ("reps", "stability", "due", "lapses"))

    res = run_turn(
        conn,
        sched,
        load_grammar(),
        StubLlm(reply="そうですか。"),
        "ご飯を食べるです。",
    )

    assert res.observations[item["id"]] == "mentioned"
    after = conn.execute("SELECT * FROM items WHERE id = ?", (item["id"],)).fetchone()
    assert tuple(after[key] for key in ("reps", "stability", "due", "lapses")) == before
    assert res.ratings == {}
    assert res.usage == {}
