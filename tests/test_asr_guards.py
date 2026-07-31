"""T2.3 — the two transcript guards, both added after an end-to-end run.

Neither is a theoretical defence. Each is here because it happened, in an 8-turn
run against real audio (`benchmarks/voice-loop.md`):

- whisper produced 「ご視聴ありがとうございました」 twice on near-silence
- one turn came back as `火が火に火に火に火に火に…`

The risk of both guards is the same: a false positive silently eats a real
utterance, which costs a whole turn and reads to the learner as being ignored.
So the Japanese in these tests matters — the fixtures are ordinary sentences and
words that legitimately repeat short spans, and if a guard starts dropping them
this file fails.
"""

from __future__ import annotations

import pytest

from ocha.speech.asr import HALLUCINATIONS, is_looping

# Real degenerate output. The first is verbatim from the run that motivated this.
LOOPS = [
    "火が火に火に火に火に火に火に火に火に火に火に",
    "カードレス" * 40,
    "あああああああああああああああ",
    "はいはいはいはいはいはいはいはいはい",
]

# Ordinary Japanese, including the words that make a naive repetition check
# unusable: 「だんだん」, 「いろいろ」, 「はいはい」 all repeat a short span on purpose.
NOT_LOOPS = [
    "はじめましてよろしくお願いします",
    "すみません、駅はどこですか?",
    "だんだん暖かくなってきましたね。",
    "いろいろな種類がありますよ。",
    "はいはい、わかりました。",
    "写真を撮ってもいいですか?",
    "これをください",
    "ありがとうございます。",
    "今日はご飯を食べました。",
    "コーヒーを飲みたいです。",
]


@pytest.mark.parametrize("text", LOOPS)
def test_a_decoder_loop_is_caught(text: str) -> None:
    assert is_looping(text), f"loop not detected: {text!r}"


@pytest.mark.parametrize("text", NOT_LOOPS)
def test_ordinary_japanese_survives(text: str) -> None:
    """The expensive failure. Dropping this text eats the learner's sentence."""
    assert not is_looping(text), f"false positive on real speech: {text!r}"


def test_short_text_is_never_a_loop() -> None:
    """「はいはい」 is a word. Below the length floor nothing is judged at all."""
    assert not is_looping("はいはい")
    assert not is_looping("そうそう")


def test_the_hallucination_list_is_what_whisper_actually_says() -> None:
    """Pinned so the list stays a record of observed output, not a guess.

    「ご視聴ありがとうございました」 — "thank you for watching" — is a YouTube
    sign-off that saturates the training data and is what whisper reaches for when
    handed silence.
    """
    assert "ご視聴ありがとうございました" in HALLUCINATIONS
    # Not a substring check on ordinary gratitude: the learner will say this.
    assert not any(h in "ありがとうございます。" for h in HALLUCINATIONS)
