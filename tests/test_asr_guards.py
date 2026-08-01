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

import sys
import types
from pathlib import Path

import pytest

import ocha.speech.asr as asr_module
from ocha.speech.asr import HALLUCINATIONS, OchaWhisper, decide_transcript, is_looping

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
    "本当に本当に本当に本当に大好きです。",
]


@pytest.mark.parametrize("text", LOOPS)
def test_a_decoder_loop_is_caught(text: str) -> None:
    assert is_looping(text), f"loop not detected: {text!r}"


@pytest.mark.parametrize("text", NOT_LOOPS)
def test_ordinary_japanese_survives(text: str) -> None:
    """The expensive failure. Dropping this text eats the learner's sentence."""
    assert not is_looping(text), f"false positive on real speech: {text!r}"


def test_warm_uses_a_resolved_local_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    fake = types.ModuleType("mlx_whisper")

    def transcribe(_samples: object, *, path_or_hf_repo: Path, language: object) -> dict[str, str]:
        del language
        seen.append(path_or_hf_repo)
        return {"text": ""}

    fake.transcribe = transcribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    monkeypatch.setattr(asr_module, "resolve_cached_model", lambda _repo: tmp_path)

    OchaWhisper(sample_rate=16_000).warm()
    assert seen == [tmp_path]


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


def test_known_hallucinations_require_an_exact_normalized_match() -> None:
    rejected = decide_transcript("  ご視聴ありがとうございました。 ")
    assert not rejected.accepted
    assert rejected.reason == "known_hallucination"

    longer = decide_transcript("昨日はご視聴ありがとうございましたと言いました。")
    assert longer.accepted


def test_good_night_is_legitimate_japanese_not_a_hallucination() -> None:
    assert "おやすみなさい" not in HALLUCINATIONS
    assert decide_transcript("おやすみなさい。").accepted
    assert decide_transcript("子供におやすみなさいと言いました。").accepted


def test_decoder_loop_returns_a_visible_rejection_reason() -> None:
    decision = decide_transcript(LOOPS[0])
    assert not decision.accepted
    assert decision.reason == "decoder_loop"
