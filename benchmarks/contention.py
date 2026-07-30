"""T0.7 — contention test: ASR + LLM + VOICEVOX, the full turn.

Measures voice-to-first-audio, which is PRD G1's actual metric, with all three
services live. TTS fires on the FIRST SENTENCE as soon as the LLM completes it,
per ARCHITECTURE §5.2 rule 2 -- not after the full reply. The difference between
those two is measured here rather than assumed.

ARCHITECTURE §4's memory budget and §5.1's latency budget both assume whisper,
the LLM and VOICEVOX are resident simultaneously. T0.3 and T0.4 measured each
ALONE, which cannot detect the interaction. This measures it.

Deviation from the original plan, for the better: that plan used
VOICEVOX-synthesised audio as ASR input because no corpus existed yet. The corpus
now exists, so this uses the real recordings — the user's own accented voice,
which is what the product actually receives.

VOICEVOX runs as a separate CPU process on :50021, so it contends for CPU rather
than GPU -- a different resource from ASR and LLM, which is why §4 could plausibly
assume all three coexist.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import time

import mlx.core as mx

HERE = pathlib.Path(__file__).parent
CORPUS = HERE / "corpus"
OUT = HERE / "contention.json"

ASR = "mlx-community/whisper-large-v3-mlx"
LLM = "mlx-community/gemma-4-26b-a4b-it-4bit"

# Standalone baselines, from asr.md and llm.md.
BASE_ASR_P50 = 1.25
BASE_LLM_TTFT = 0.50
BASE_LLM_TPS = 39.4

SYSTEM = """You are a Japanese conversation partner. Reply in 1–2 short sentences.

VOCABULARY: Use only words from KNOWN. If you must introduce a new word,
introduce exactly one and gloss it in English in parentheses.
KNOWN: 私、これ、それ、どこ、何、今日、明日、水、コーヒー、ご飯、店、駅、家、友達、食べる、飲む、行く、来る、見る、する、ある、いる、好き、いい、高い、安い、元気

TARGET: Steer the conversation so the learner naturally needs these:
食べる

REGISTER: Always use polite です/ます form. Never mix polite and plain forms.

AVOID: Do not use grammar beyond beginner.

Never break character to explain grammar. If asked a grammar question,
respond with exactly: [GRAMMAR_QUERY]"""


def swap() -> tuple[int, int]:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = lambda k: int(next(l for l in out.splitlines() if k in l).split()[-1].rstrip("."))
    return g("Swapins"), g("Swapouts")


def rss_gb() -> float:
    out = subprocess.run(["ps", "-Ao", "rss,comm"], capture_output=True, text=True).stdout
    tot = sum(int(l.split()[0]) for l in out.splitlines()[1:] if l.split()[0].isdigit())
    return tot / 1048576


def main() -> None:
    import mlx_whisper
    from mlx_lm import generate, load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    sw0 = swap()
    base_rss = rss_gb()
    print(f"baseline: all-process RSS {base_rss:.1f} GB, swapins/outs {sw0}\n")

    # --- load both, co-resident, and never free either ---
    t = time.perf_counter()
    mlx_whisper.transcribe(str(CORPUS / "03.wav"), path_or_hf_repo=ASR, language="ja")
    asr_load = time.perf_counter() - t
    asr_mem = mx.get_active_memory() / 1e9
    print(f"ASR loaded   {asr_load:5.1f}s   mlx active {asr_mem:5.2f} GB")

    t = time.perf_counter()
    model, tok = load(LLM)
    llm_load = time.perf_counter() - t
    both_mem = mx.get_active_memory() / 1e9
    print(f"LLM loaded   {llm_load:5.1f}s   mlx active {both_mem:5.2f} GB  (both resident)")
    print(f"             all-process RSS {rss_gb():.1f} GB\n")

    cache = make_prompt_cache(model)
    prime = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}],
        add_generation_prompt=False, tokenize=False, enable_thinking=False,
    )
    for _ in stream_generate(model, tok, prime, max_tokens=1, prompt_cache=cache):
        pass

    # --- warm both, then run turns with BOTH resident ---
    mlx_whisper.transcribe(str(CORPUS / "03.wav"), path_or_hf_repo=ASR, language="ja")

    rows = []
    for uid in ["03", "04", "05", "07", "09", "12", "14", "19"]:
        wav = str(CORPUS / f"{uid}.wav")

        t = time.perf_counter()
        text = mlx_whisper.transcribe(wav, path_or_hf_repo=ASR, language="ja")["text"].strip()
        t_asr = time.perf_counter() - t

        turn = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        t = time.perf_counter()
        ttft, n, gen0 = None, 0, None
        reply = []
        for r in stream_generate(model, tok, turn, max_tokens=48, prompt_cache=cache):
            if ttft is None:
                ttft = time.perf_counter() - t
                gen0 = time.perf_counter()
            reply.append(r.text)
            n += 1
        dec = time.perf_counter() - gen0
        tps = (n - 1) / dec if n > 1 and dec > 0 else 0.0
        # first sentence is what §5.1 actually waits on before TTS can start
        txt = "".join(reply)
        first_sent = txt.split("。")[0] + "。" if "。" in txt else txt
        n_first = len(tok.encode(first_sent))
        t_first_sentence = ttft + (n_first / tps if tps else 0)

        # TTS on the first sentence, as §5.2 rule 2 requires
        import urllib.parse, urllib.request
        t = time.perf_counter()
        q = urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:50021/audio_query?text={urllib.parse.quote(first_sent)}&speaker=3",
            method="POST"), timeout=60).read()
        t_query = time.perf_counter() - t
        t = time.perf_counter()
        wav = urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:50021/synthesis?speaker=3", data=q,
            headers={"Content-Type": "application/json"}, method="POST"), timeout=60).read()
        t_synth = time.perf_counter() - t
        t_tts = t_query + t_synth

        rows.append({
            "id": uid, "asr_s": round(t_asr, 3), "ttft_s": round(ttft, 3),
            "tts_query_s": round(t_query, 3), "tts_synth_s": round(t_synth, 3),
            "tts_s": round(t_tts, 3), "wav_bytes": len(wav),
            "voice_to_first_audio_s": round(t_asr + t_first_sentence + t_tts, 3),
            "tok_per_s": round(tps, 1), "first_sentence_s": round(t_first_sentence, 3),
            "partial_chain_s": round(t_asr + t_first_sentence, 3),
            "transcript": text, "reply": txt.strip(),
        })
        r = rows[-1]
        print(f"  {uid}  ASR {r['asr_s']:5.2f}  LLM-1st {r['first_sentence_s']:5.2f}"
              f"  TTS {r['tts_s']:5.2f}  ->  voice-to-first-audio {r['voice_to_first_audio_s']:5.2f}s")

    peak = mx.get_peak_memory() / 1e9
    sw1 = swap()
    med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]

    print(f"\nmlx peak {peak:.2f} GB   all-process RSS {rss_gb():.1f} GB"
          f"   swapins/outs {sw0} -> {sw1}")
    print("\n" + "=" * 66)
    print(f"{'stage':22}{'standalone':>12}{'co-resident':>13}{'delta':>10}")
    for label, got, base in (
        ("ASR p50", med("asr_s"), BASE_ASR_P50),
        ("LLM TTFT p50", med("ttft_s"), BASE_LLM_TTFT),
        ("LLM tok/s p50", med("tok_per_s"), BASE_LLM_TPS),
    ):
        d = (got - base) / base * 100
        print(f"{label:22}{base:12.2f}{got:13.2f}{d:+9.0f}%")
    print(f"\n{'TTS p50 (query+synth)':22}{'':12}{med('tts_s'):13.2f}")
    v50 = med("voice_to_first_audio_s")
    vs = sorted(r["voice_to_first_audio_s"] for r in rows)
    print(f"\nVOICE-TO-FIRST-AUDIO  p50 {v50:.2f}s   p95 {vs[int(len(vs)*.95)-1]:.2f}s"
          f"   (+VAD ~150ms, +network ~30ms not included)")
    print(f"PRD G1 target: 1.20 s p50  ->  {'MET' if v50 + 0.18 < 1.2 else f'MISSED by {v50 + 0.18 - 1.2:.2f}s'}"
          f"  (adding 180ms for VAD+network)")

    OUT.write_text(json.dumps({
        "asr": ASR, "llm": LLM, "rows": rows,
        "mlx_peak_gb": round(peak, 2), "all_process_rss_gb": round(rss_gb(), 1),
        "swap_before": sw0, "swap_after": sw1,
        "voicevox": "0.25.2 CPU, speaker=3, separate process on :50021",
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
