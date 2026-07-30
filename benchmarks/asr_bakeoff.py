"""T0.3 — ASR bake-off on the T0.2 corpus.

Runs all three candidates through the SAME backend (transformers + MPS). MLX was
preferred, but no MLX conversion of kotoba-whisper-bilingual-v1.0 exists, and
mixing backends across candidates would make the latency column meaningless.

All three are invoked with language="ja", task="transcribe" -- the configuration
a Japanese tutor would actually deploy. None of the three auto-detects language;
the bilingual model's card is explicit that you pass the language per call. Its
hypothesised advantage is therefore not auto-switching but that English was in
its training, so it may render English spans faithfully even when told "ja".

Writes results.json (raw outputs, for reproducible Stage 1 judgment) and prints
the Stage 1 and Stage 2 tables.
"""

from __future__ import annotations

import json
import pathlib
import re
import time
import unicodedata

CORPUS = pathlib.Path(__file__).parent / "corpus"
OUT = pathlib.Path(__file__).parent / "asr-results.json"

MODELS = {
    "kotoba-whisper-v2.0": "kotoba-tech/kotoba-whisper-v2.0",
    "kotoba-whisper-bilingual-v1.0": "kotoba-tech/kotoba-whisper-bilingual-v1.0",
    "whisper-large-v3": "openai/whisper-large-v3",
}

# Pre-registered in TASKS.md T0.3 before any audio was recorded.
PUNCT = "、。！？「」，．,.!?…・「」『』（）()"


def normalize(s: str) -> str:
    """NFKC, strip ALL whitespace, strip punctuation, lowercase ASCII."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    s = "".join(c for c in s if c not in PUNCT)
    return s.lower()


def transcribe_all(model_id: str) -> list[dict]:
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        dtype=torch.float16,
        device="mps",
    )
    # transformers 5.x: a model's own generation_config.forced_decoder_ids takes
    # precedence over the language/task kwargs and silently ignores them. Without
    # this, forcing "ja" is a no-op -- verified by forced-ja and auto-detect
    # returning byte-identical output on the first run.
    pipe.model.generation_config.forced_decoder_ids = None
    # no_repeat_ngram_size applies to ALL three models, so the comparison stays
    # fair. Without it whisper-large-v3 degenerates on utterance 18 into ~200
    # repetitions of カードレス (557 insertions, 70 s). That is a decoding-config
    # artifact, not a model property -- with it, the same utterance takes 4.07 s.
    # Reporting the unmitigated run would have disqualified the model for a
    # limitation of the harness. 4 is conservative: Japanese legitimately repeats
    # short spans, so a smaller window would suppress correct output.
    kw = {"language": "ja", "task": "transcribe", "no_repeat_ngram_size": 4}
    probe = pipe.model.generation_config.forced_decoder_ids
    assert probe is None, f"language forcing will be ignored: {probe}"

    data = json.loads((CORPUS / "transcripts.json").read_text(encoding="utf-8"))
    items = data["utterances"]

    # Warm: first call pays model compile/alloc; timing it would misattribute
    # cold-start cost to utterance 01.
    pipe(str(CORPUS / f"{items[0]['id']}.wav"), generate_kwargs=kw)

    rows = []
    for u in items:
        p = CORPUS / f"{u['id']}.wav"
        t0 = time.perf_counter()
        text = pipe(str(p), generate_kwargs=kw)["text"]
        rows.append(
            {
                "id": u["id"],
                "subset": u["subset"],
                "ref": u["text"],
                "hyp": text.strip(),
                "seconds": round(time.perf_counter() - t0, 3),
                "alt_ok": u.get("alt_ok", []),
            }
        )
        print(f"    {u['id']} {rows[-1]['seconds']:5.2f}s  {text.strip()[:60]}")
    del pipe
    return rows


def score(rows: list[dict]) -> dict:
    import jiwer

    def cer_for(subset: str | None) -> dict:
        sel = [r for r in rows if subset is None or r["subset"] == subset]
        refs = [normalize(r["ref"]) for r in sel]
        hyps = [normalize(r["hyp"]) for r in sel]
        o = jiwer.process_characters(refs, hyps)
        n = sum(len(r) for r in refs)
        return {
            "cer": round(o.cer * 100, 2),
            "sub": o.substitutions,
            "ins": o.insertions,
            "del": o.deletions,
            "ref_chars": n,
        }

    lat = sorted(r["seconds"] for r in rows)
    return {
        "overall": cer_for(None),
        "pure_ja": cer_for("pure_ja"),
        "code_switched": cer_for("code_switched"),
        "latency_p50": round(lat[len(lat) // 2], 3),
        "latency_p95": round(lat[int(len(lat) * 0.95) - 1], 3),
        "latency_total": round(sum(lat), 1),
    }


def main() -> None:
    results = {}
    for name, model_id in MODELS.items():
        print(f"\n=== {name}  ({model_id})")
        rows = transcribe_all(model_id)
        results[name] = {"model_id": model_id, "rows": rows, "scores": score(rows)}
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")

    print("\n" + "=" * 78)
    print("STAGE 1 — code-switched outputs, for per-utterance usability judgment")
    print("=" * 78)
    for name, r in results.items():
        print(f"\n--- {name}")
        for row in r["rows"]:
            if row["subset"] != "code_switched":
                continue
            print(f"  {row['id']} REF {row['ref']}")
            print(f"     HYP {row['hyp']}")

    print("\n" + "=" * 78)
    print("STAGE 2 — CER (only for models surviving Stage 1)")
    print("=" * 78)
    print(f"{'model':32} {'pureJA':>7} {'codeSW':>7} {'wtd':>7} {'overall':>8} {'p50':>6} {'p95':>6}")
    for name, r in results.items():
        s = r["scores"]
        wtd = 0.65 * s["code_switched"]["cer"] + 0.35 * s["pure_ja"]["cer"]
        print(
            f"{name:32} {s['pure_ja']['cer']:7.2f} {s['code_switched']['cer']:7.2f} "
            f"{wtd:7.2f} {s['overall']['cer']:8.2f} {s['latency_p50']:6.2f} {s['latency_p95']:6.2f}"
        )
    print("\nsub/ins/del per subset (insertion-heavy = decoder looping):")
    for name, r in results.items():
        for sub in ("pure_ja", "code_switched"):
            s = r["scores"][sub]
            print(f"  {name:32} {sub:14} S{s['sub']:4} I{s['ins']:4} D{s['del']:4}")


def _self_check() -> None:
    """The normalizer is the instrument; an uncalibrated one yields a confident wrong table."""
    assert normalize("これは、いくらですか。") == "これはいくらですか"
    assert normalize("Ｗｉ－Ｆｉ") == "wi-fi", normalize("Ｗｉ－Ｆｉ")  # NFKC + lowercase
    assert normalize("a b\tc\n") == "abc"
    assert normalize("えーと、how do you say?") == "えーとhowdoyousay"
    # identical strings must score 0, and normalization must not create matches
    import jiwer

    assert jiwer.process_characters([normalize("これをください。")], [normalize("これをください")]).cer == 0
    r = jiwer.process_characters([normalize("これ")], [normalize("それ")])
    assert 0 < r.cer <= 1, r.cer
    print("normalizer self-check passed")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
