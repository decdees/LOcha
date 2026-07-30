"""T0.5 — generation probe.

Replaces the original grammar-explanation probe. FR-5 firewalls the model's
explanations, so explanation quality cannot change any decision; what the product
depends on is whether the model PRODUCES acceptable Japanese under constraint.

Four checks, all mechanically decidable without Japanese grammar expertise:

  1. vocabulary  -- content words within KNOWN, plus at most one glossed new word
  2. register    -- polite/plain consistent at SENTENCE-FINAL position only
  3. length      -- 1-2 sentences
  4. sentinel    -- [GRAMMAR_QUERY] emitted instead of answering grammar questions

What this does NOT measure: whether the Japanese is natural. A model can pass all
four and still be stilted. That needs a Japanese-competent reviewer; all replies
are saved to generation-probe-replies.json for exactly that.
"""

from __future__ import annotations

import json
import pathlib
import re

OUT = pathlib.Path(__file__).parent / "generation-probe.json"
REPLIES = pathlib.Path(__file__).parent / "generation-probe-replies.json"

MODELS = {
    "gemma-4-26b-a4b": "mlx-community/gemma-4-26b-a4b-it-4bit",
    "Qwen3.5-9B": "mlx-community/Qwen3.5-9B-4bit",
}

KNOWN = [
    "私", "あなた", "これ", "それ", "あれ", "ここ", "そこ", "どこ", "何", "誰",
    "今日", "明日", "昨日", "朝", "夜", "水", "コーヒー", "ご飯", "店", "駅",
    "家", "友達", "先生", "本", "車", "食べる", "飲む", "行く", "来る", "見る",
    "する", "ある", "いる", "好き", "大きい", "小さい", "いい", "高い", "安い", "元気",
]
TARGET = "食べる"

# ARCHITECTURE §7.1 template, verbatim.
SYSTEM = f"""You are a Japanese conversation partner. Reply in 1–2 short sentences.

VOCABULARY: Use only words from KNOWN. If you must introduce a new word,
introduce exactly one and gloss it in English in parentheses.
KNOWN: {"、".join(KNOWN)}

TARGET: Steer the conversation so the learner naturally needs these:
{TARGET}

AVOID: Do not use grammar beyond beginner.

Never break character to explain grammar. If asked a grammar question,
respond with exactly: [GRAMMAR_QUERY]"""

CONVERSATIONAL = [
    "こんにちは。", "今日は元気ですか。", "私は水を飲みました。", "明日どこに行きますか。",
    "駅に行きます。", "友達と店に行きました。", "コーヒーが好きです。", "本を見ました。",
    "先生はどこですか。", "車は高いです。", "昨日ご飯を食べました。", "家にいます。",
    "これはいいですか。", "夜は何をしますか。", "朝コーヒーを飲みます。",
]
GRAMMAR = [
    "What is the difference between は and が?",
    "Why is it を here and not が?",
    "When do I use に versus で?",
    "How does the て-form work?",
    "Is 開く transitive or intransitive?",
]

SENT_END = re.compile(r"[。！？!?]")
POLITE_TAIL = re.compile(r"(です|ます|ました|ません|でした|ください|ましょう|でしょう)(か|ね|よ|)$")
PLAIN_TAIL = re.compile(r"(だ|だった|る|た|ない|なかった|よ|ね)$")
GLOSS = re.compile(r"[（(][^）)]*[A-Za-z][^）)]*[）)]")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def check_length(reply: str) -> tuple[bool, str]:
    """Count JAPANESE sentences only.

    FR-3 explicitly permits an English gloss for a new word, so the gloss is not
    part of the reply length. The first version counted it: "駅に行きますか。
    (Do you go to the station?)" scored 3 sentences and failed, when it is one
    compliant sentence plus a permitted gloss. That understated Qwen3.5-9B at
    9/15 when it actually passes.
    """
    n = len(sentences(GLOSS.sub("", reply)))
    return n <= 2, f"{n} sentence(s)"


def check_register(reply: str, tagger) -> tuple[bool, str]:
    """Compare SENTENCE-FINAL forms only.

    Plain form is legitimate inside subordinate clauses ("食べる時"), so scanning
    the whole reply would flag correct Japanese. Only the final form of each
    sentence carries register.
    """
    kinds = []
    for s in sentences(reply):
        if POLITE_TAIL.search(s):
            kinds.append("polite")
        elif PLAIN_TAIL.search(s):
            kinds.append("plain")
    if len(set(kinds)) > 1:
        return False, f"mixed: {kinds}"
    return True, kinds[0] if kinds else "none detected"


def check_vocab(reply: str, tagger) -> tuple[bool, str]:
    known = set(KNOWN)
    stripped = GLOSS.sub("", reply)  # a glossed word is allowed; drop the gloss text
    out = []
    for w in tagger(stripped):
        pos = w.feature.pos1
        if pos not in ("名詞", "動詞", "形容詞"):
            continue
        lemma = getattr(w.feature, "lemma", None) or w.surface
        if lemma in known or w.surface in known:
            continue
        if re.fullmatch(r"[ぁ-んァ-ヶーA-Za-z0-9]+", w.surface) and len(w.surface) <= 1:
            continue
        out.append(w.surface)
    seen = list(dict.fromkeys(out))
    n_gloss = len(GLOSS.findall(reply))
    # at most one new word, and it must be glossed
    ok = len(seen) == 0 or (len(seen) <= 1 and n_gloss >= 1)
    return ok, f"{len(seen)} out-of-vocab {seen[:6]}, {n_gloss} gloss(es)"


def check_sentinel(reply: str) -> tuple[bool, str]:
    if "[GRAMMAR_QUERY]" not in reply:
        return False, "sentinel absent"
    rest = reply.replace("[GRAMMAR_QUERY]", "").strip()
    if len(rest) > 4:
        return False, f"sentinel + {len(rest)} chars of extra text"
    return True, "clean sentinel"


def run(repo: str) -> list[dict]:
    from mlx_lm import generate, load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(repo)
    rows = []
    for kind, prompts in (("conversational", CONVERSATIONAL), ("grammar", GRAMMAR)):
        for p in prompts:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
            # enable_thinking=False is REQUIRED, not cosmetic. Both candidates are
            # reasoning models: by default they emit a thinking channel
            # (<|channel>thought / "Thinking Process:") before the reply. That
            # breaks this probe, blows the latency budget, and -- critically --
            # leaks grammar explanations alongside the [GRAMMAR_QUERY] sentinel,
            # which is exactly what FR-5 forbids. See generation-probe.md.
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            reply = generate(model, tok, text, max_tokens=120, verbose=False).strip()
            rows.append({"kind": kind, "prompt": p, "reply": reply})
    del model, tok
    return rows


def score(rows: list[dict]) -> dict:
    import fugashi

    tagger = fugashi.Tagger()
    res = {"vocab": [0, 0], "register": [0, 0], "length": [0, 0], "sentinel": [0, 0]}
    for r in rows:
        if r["kind"] == "conversational":
            for name, fn in (
                ("vocab", lambda x: check_vocab(x, tagger)),
                ("register", lambda x: check_register(x, tagger)),
                ("length", check_length),
            ):
                ok, why = fn(r["reply"])
                res[name][1] += 1
                res[name][0] += ok
                r[name] = why
        else:
            ok, why = check_sentinel(r["reply"])
            res["sentinel"][1] += 1
            res["sentinel"][0] += ok
            r["sentinel"] = why
    return res


def main() -> None:
    all_rows, all_scores = {}, {}
    for name, repo in MODELS.items():
        print(f"=== {name}")
        rows = run(repo)
        all_scores[name] = score(rows)
        all_rows[name] = rows
        for k, (a, b) in all_scores[name].items():
            print(f"    {k:10} {a}/{b}")
        print()

    OUT.write_text(json.dumps(all_scores, indent=2) + "\n")
    REPLIES.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n")

    print(f"{'model':20} {'vocab':>10} {'register':>10} {'length':>10} {'sentinel':>10}")
    for n, s in all_scores.items():
        print(
            f"{n:20} {s['vocab'][0]:>4}/{s['vocab'][1]:<5} {s['register'][0]:>4}/{s['register'][1]:<5} "
            f"{s['length'][0]:>4}/{s['length'][1]:<5} {s['sentinel'][0]:>4}/{s['sentinel'][1]:<5}"
        )
    print(f"\nreplies saved to {REPLIES.name} for later naturalness review")


def _self_check() -> None:
    """Calibrate the checkers against fixtures. An uncalibrated instrument
    produces a confident wrong table -- the T0.3 lesson."""
    import fugashi

    t = fugashi.Tagger()
    assert check_length("はい。元気です。")[0]
    assert not check_length("はい。元気です。そうですね。")[0], "3 sentences must fail"
    # an FR-3 gloss is not a sentence
    assert check_length("駅に行きますか。(Do you go to the station?)")[0], "gloss must not count"
    assert check_length("高いですか？（Is it expensive?）\n安いですか？（Or cheap?）")[0]
    assert check_register("水を飲みます。ご飯を食べます。", t)[0]
    assert not check_register("水を飲みます。ご飯を食べる。", t)[0], "mixed register must fail"
    # plain form inside a subordinate clause is legitimate, not a violation
    assert check_register("食べる時、水を飲みます。", t)[0], "subordinate plain form must pass"
    assert check_vocab("水を飲みます。", t)[0]
    assert not check_vocab("図書館に行きます。", t)[0], "out-of-vocab word must fail"
    assert check_vocab("図書館（library）に行きます。", t)[0], "one glossed new word is allowed"
    assert check_sentinel("[GRAMMAR_QUERY]")[0]
    assert not check_sentinel("は marks the topic.")[0], "answering must fail"
    assert not check_sentinel("[GRAMMAR_QUERY] は marks the topic and が marks the subject.")[0]
    print("all checkers calibrated")


if __name__ == "__main__":
    import sys

    _self_check() if "--self-check" in sys.argv else main()
