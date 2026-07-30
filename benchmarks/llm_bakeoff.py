"""T0.4 — LLM bake-off via MLX.

Measures cold load, time-to-first-token, sustained tok/s, and resident memory at
8k context. Then derives EFFECTIVE bandwidth (bytes_per_token x tok/s) and
compares it against T0.1's measured 103.2 GB/s.

That last number is the point of this task. ARCHITECTURE.md §3 predicts 25-35
tok/s for the MoE from a 2.2 GB/token figure, which assumes MoE weight reads
stream as efficiently as dense ones. Per-token expert routing makes them
scattered, so the achieved fraction of peak bandwidth is expected to be worse for
the MoE than for the dense model. Measuring both is what makes that visible.

qwen3.5:27b is skipped: T0.1 measured a base M4, and TASKS.md gates that
candidate on M4 Pro-class bandwidth.
"""

from __future__ import annotations

import gc
import json
import pathlib
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

OUT = pathlib.Path(__file__).parent / "llm-results.json"
MEASURED_BW_GBS = 103.2  # benchmarks/hardware.md

MODELS = {
    # name: (repo, GB read per token at 4-bit)
    # MoE reads only its ~3.8B active params; dense reads everything.
    "gemma-4-26b-a4b (MoE)": ("mlx-community/gemma-4-26b-a4b-it-4bit", 2.2),
    "Qwen3.5-9B (dense)": ("mlx-community/Qwen3.5-9B-4bit", 5.0),
}

# A tutor turn, not a creative-writing prompt -- reply length is what PRD FR-3
# constrains and what the latency budget assumes.
PROMPT = "日本語で短く返事してください。1、2文だけ。質問：週末は何をしましたか。"
MAX_TOKENS = 64
CTX_TOKENS = 8192


def bench(repo: str, gb_per_token: float) -> dict:
    mx.reset_peak_memory()

    t0 = time.perf_counter()
    model, tokenizer = load(repo)
    cold_load = time.perf_counter() - t0
    weights_gb = mx.get_active_memory() / 1e9

    def run(prompt: str, n: int) -> tuple[float, float, int]:
        """Return (ttft, tok_per_sec, generated)."""
        t = time.perf_counter()
        ttft = None
        count = 0
        gen_start = None
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=n):
            if ttft is None:
                ttft = time.perf_counter() - t
                gen_start = time.perf_counter()
            count += 1
        elapsed = time.perf_counter() - gen_start if gen_start else 0.0
        # tok/s over tokens AFTER the first, so prefill is not double-counted
        tps = (count - 1) / elapsed if elapsed > 0 and count > 1 else 0.0
        return ttft or 0.0, tps, count

    run(PROMPT, 8)  # warm: first call pays kernel compilation

    ttft, tps, n = run(PROMPT, MAX_TOKENS)
    short_peak = mx.get_peak_memory() / 1e9

    # 8k context: the KV-cache growth §4 warns will silently push into swap.
    filler = "これは文脈を長くするための文章です。" * 400
    long_prompt = filler[: CTX_TOKENS * 2]
    ntok = len(tokenizer.encode(long_prompt))
    mx.reset_peak_memory()
    ttft_8k, tps_8k, _ = run(long_prompt, 32)
    peak_8k = mx.get_peak_memory() / 1e9

    del model, tokenizer
    gc.collect()

    return {
        "cold_load_s": round(cold_load, 2),
        "weights_gb": round(weights_gb, 2),
        "ttft_s": round(ttft, 3),
        "tok_per_s": round(tps, 1),
        "generated": n,
        "peak_gb_short": round(short_peak, 2),
        "ctx_tokens": ntok,
        "ttft_8k_s": round(ttft_8k, 3),
        "tok_per_s_8k": round(tps_8k, 1),
        "peak_gb_8k": round(peak_8k, 2),
        "gb_per_token": gb_per_token,
        "effective_bw_gbs": round(gb_per_token * tps, 1),
        "pct_of_measured_bw": round(gb_per_token * tps / MEASURED_BW_GBS * 100, 1),
        "ceiling_tok_per_s": round(MEASURED_BW_GBS / gb_per_token, 1),
    }


def main() -> None:
    import subprocess

    others = subprocess.run(
        ["ps", "-Ao", "rss,comm"], capture_output=True, text=True
    ).stdout.splitlines()
    top = sorted((int(l.split()[0]) for l in others[1:] if l.split()[0].isdigit()), reverse=True)[:5]
    print(f"baseline: top-5 resident processes sum to {sum(top)/1048576:.1f} GB\n")

    results = {}
    for name, (repo, gpt) in MODELS.items():
        print(f"=== {name}  ({repo})")
        results[name] = bench(repo, gpt)
        for k, v in results[name].items():
            print(f"    {k:22} {v}")
        print()

    OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("=" * 86)
    print(f"{'model':24} {'load':>6} {'TTFT':>6} {'tok/s':>7} {'ceil':>6} {'eff BW':>8} {'%peak':>6} {'8k GB':>7}")
    for n, r in results.items():
        print(
            f"{n:24} {r['cold_load_s']:6.1f} {r['ttft_s']:6.2f} {r['tok_per_s']:7.1f} "
            f"{r['ceiling_tok_per_s']:6.1f} {r['effective_bw_gbs']:8.1f} "
            f"{r['pct_of_measured_bw']:5.0f}% {r['peak_gb_8k']:7.2f}"
        )
    print("\nceil = T0.1 bandwidth / bytes-per-token. %peak = achieved fraction of 103.2 GB/s.")


if __name__ == "__main__":
    main()
