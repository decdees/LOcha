"""T0.4 follow-up — prefill cost and KV growth at REAL context lengths.

The first pass asked for 8k context and actually got 3600 tokens (the filler was
sliced by characters, not tokens), so the headline "8k" figures were measured at
under half that. This re-measures against token-counted prompts.

Two things matter here and neither was visible in the first pass:

1. ARCHITECTURE §4 budgets ~1.0 GB of KV cache at 8k. If that is low, NFR-1's
   27 GB ceiling is tighter than it looks.
2. §5.1 budgets 200 ms TTFT. Prefill is O(context), and a conversation grows.
   If prefill at realistic context blows the budget, the loop needs prompt-cache
   reuse across turns rather than re-prefilling every turn -- an architectural
   requirement, not an optimisation.
"""

from __future__ import annotations

import json
import pathlib
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

OUT = pathlib.Path(__file__).parent / "llm-context.json"
REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"  # the §3 default pick
LENGTHS = [256, 1024, 4096, 8192]


def main() -> None:
    model, tok = load(REPO)
    base = mx.get_active_memory() / 1e9
    print(f"weights resident: {base:.2f} GB\n")

    unit = tok.encode("これは文脈を長くするための文章です。")
    rows = []
    for target in LENGTHS:
        ids = (unit * (target // len(unit) + 2))[:target]
        prompt = tok.decode(ids)
        n = len(tok.encode(prompt))

        mx.reset_peak_memory()
        t = time.perf_counter()
        ttft = None
        count = 0
        gen_start = None
        for _ in stream_generate(model, tok, prompt, max_tokens=16):
            if ttft is None:
                ttft = time.perf_counter() - t
                gen_start = time.perf_counter()
            count += 1
        decode_s = time.perf_counter() - gen_start
        tps = (count - 1) / decode_s if count > 1 else 0.0
        peak = mx.get_peak_memory() / 1e9

        rows.append(
            {
                "requested": target,
                "actual_tokens": n,
                "ttft_s": round(ttft, 3),
                "prefill_tok_per_s": round(n / ttft, 0),
                "decode_tok_per_s": round(tps, 1),
                "peak_gb": round(peak, 2),
                "kv_gb": round(peak - base, 2),
            }
        )
        r = rows[-1]
        print(
            f"  ctx {n:5}  TTFT {r['ttft_s']:6.2f}s  prefill {r['prefill_tok_per_s']:6.0f} tok/s"
            f"  decode {r['decode_tok_per_s']:5.1f} tok/s  peak {r['peak_gb']:5.2f} GB"
            f"  KV {r['kv_gb']:4.2f} GB"
        )

    OUT.write_text(json.dumps({"repo": REPO, "weights_gb": base, "rows": rows}, indent=2) + "\n")

    print("\nvs ARCHITECTURE.md:")
    k8 = next(r for r in rows if r["requested"] == 8192)
    print(f"  §4 KV @8k budgeted ~1.00 GB   measured {k8['kv_gb']:.2f} GB")
    print(f"  §4 total budgeted  ~15.5 GB   measured {k8['peak_gb']:.2f} GB (model+KV)")
    print(f"  §5.1 TTFT budgeted  0.20 s    measured {k8['ttft_s']:.2f} s at 8k context")
    small = next(r for r in rows if r["requested"] == 256)
    print(f"                                measured {small['ttft_s']:.2f} s at 256 tokens")


if __name__ == "__main__":
    main()
