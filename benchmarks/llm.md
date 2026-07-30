# T0.4 — LLM bake-off

**Date:** 30 July 2026 · base M4, 32 GB · measured bandwidth 103.2 GB/s (`hardware.md`)
**Candidates:** `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/Qwen3.5-9B-4bit`
**Skipped:** `qwen3.5:27b` — TASKS.md gates it on M4 Pro-class bandwidth; T0.1 measured base M4.

**Conditions:** container runtime shut down; top-5 resident processes summed to 2.7 GB at start; swap 0.00 MB throughout.

---

## Headline

| model | cold load | TTFT (40 tok prompt) | tok/s | ceiling | effective BW | % of peak |
|---|---|---|---|---|---|---|
| **gemma-4-26b-a4b (MoE)** | 7.4 s | 0.32 s | **39.4** | 46.9 | 86.7 GB/s | **84%** |
| Qwen3.5-9B (dense) | 5.4 s | 0.33 s | 20.5 | 20.6 | 102.3 GB/s | **99%** |

Ceiling = 103.2 GB/s ÷ bytes-per-token (2.2 GB MoE active, 5.0 GB dense).

**Throughput winner: `gemma-4-26b-a4b`.** Nearly 2× the dense 9B at realistic context, and §4's memory estimate holds.

> **This is NOT the shipped choice.** `DECISION.md` names **`mlx-community/Qwen3.5-9B-4bit`**. Gemma won on speed and lost on correctness: it produced `今日は何を食べるですか？`, where `です` cannot attach to a plain-form verb (correct: `食べますか` / `食べるんですか`). A tutor teaching a chapter-3 conjugation error to a learner who cannot detect it is worse than a slower tutor that is correct — and the 1.9× was being spent on the wrong bottleneck, since ASR is 5× over budget while the LLM is 1.6×.
>
> Gemma stays measured here as the **faster-but-less-accurate** option. Switching back is a config edit (`OCHA_LLM_MODEL`), not a code change.

---

## The MoE scattered-read hypothesis: confirmed, and quantified

T0.1 predicted that per-token expert routing would make MoE weight reads scattered rather than contiguous, so the MoE should achieve a *lower fraction of peak bandwidth* than a dense model. That is exactly what happened:

- Dense Qwen3.5-9B achieves **99% of measured peak** — essentially perfectly bandwidth-bound, reading its weights contiguously.
- MoE gemma-4-26b-a4b achieves **84% of measured peak** — a ~15% efficiency penalty.

This is the first direct evidence for the mechanism §3 reasons about but never measured. **It does not change the conclusion.** The MoE reads 2.2 GB/token against the dense model's 5.0, so even at 84% efficiency it delivers 39.4 tok/s against 20.5.

### §3's realistic estimate was pessimistic

§3 predicts "40–50 tok/s theoretical, 25–35 realistic" for the MoE. Measured: **39.4 tok/s**, above the top of the realistic band. §3's caution was warranted in mechanism but overstated in magnitude — corrected in place.

---

## Prefill is the real problem, and §5.1 does not account for it

§5.1 budgets **200 ms** for LLM time-to-first-token. TTFT is prefill plus one decode step, and prefill is O(context):

| context | TTFT | prefill throughput | decode | peak mem | KV |
|---|---|---|---|---|---|
| 256 tok | 1.60 s | 160 tok/s | 37.7 tok/s | 14.48 GB | 0.27 GB |
| 1024 tok | 2.60 s | 394 tok/s | 36.0 tok/s | 15.01 GB | 0.80 GB |
| 4096 tok | 11.03 s | 372 tok/s | 32.1 tok/s | 15.95 GB | 1.75 GB |
| 8192 tok | 32.58 s | 251 tok/s | 14.3 tok/s | 16.14 GB | 1.94 GB |

**At 8k context TTFT is 32.6 seconds** — 163× the budget. Even at 256 tokens it is 1.60 s, 8× the budget.

### Prompt-cache reuse rescues it — and is therefore mandatory

Simulating a 10-turn conversation (216-token system prompt) two ways:

| | p50 TTFT | p95 | worst | grows with conversation? |
|---|---|---|---|---|
| Re-prefill every turn | 1.81 s | 2.19 s | 3.24 s | **yes** — 1.23 s → 2.19 s over 10 turns |
| KV-cache reuse across turns | **0.50 s** | 0.54 s | 0.54 s | **no** — flat at 0.46–0.54 s |

**3.6× faster at p50, and — more importantly — flat.** Without reuse, TTFT degrades as the conversation continues, so the product gets worse the longer you talk to it. With reuse, only the new tokens are prefilled and cost stays constant.

**This is an architectural requirement for Phase 2, not an optimisation.** `mlx_lm.models.cache.make_prompt_cache` supports it; the cache must be held across turns in the LLM service and invalidated only when the Context Builder changes the prefix.

### Even with caching, §5.1's budget does not close

Measured LLM TTFT with reuse is **0.50 s against a 200 ms budget** — 2.5× over. Recomputing §5.1 with what is measured so far, and the rest still at their budgeted values:

```
VAD endpoint          150 ms   (budgeted, unmeasured)
ASR                   250 ms   (budgeted, unmeasured — T0.3)
LLM TTFT              500 ms   MEASURED, was budgeted 200
first sentence        400 ms   MEASURED 37.7 tok/s, ~15 tokens
VOICEVOX              200 ms   (budgeted, unmeasured)
network                30 ms
────────────────────────────
total               ~1530 ms   vs §5.1's ~1100 ms and PRD G1's 1200 ms p50
```

**PRD G1 (p50 < 1200 ms) is at risk on the LLM stage alone.** Not yet a failure — VAD, ASR and TTS are unmeasured and T0.7 measures the real chain — but the LLM contributes 300 ms more than budgeted and there is no slack elsewhere to absorb it.

---

## Decode degrades with context — §3 assumes it does not

§3's whole argument treats tok/s as a constant. It is not: gemma falls from **37.7 tok/s at 256 tokens to 14.3 at 8192**, a 62% loss. Qwen3.5-9B degrades far less (20.5 → 19.9 at 3.6k).

Consequence worth stating plainly: **at 8k context the MoE's throughput advantage is nearly gone** (14.3 vs ~19 dense). The MoE wins decisively at the context lengths this product actually uses — a 10-turn tutor conversation measured **560 tokens** — but "MoE is 2× faster" is only true below roughly 4k.

This strengthens §4's existing advice to cap context, and sharpens it: **the cap should be ~2k, not 8k.** At 2k the MoE still runs ~36 tok/s; at 8k it runs slower than the dense model would.

---

## Memory: §4 is close but its KV estimate is half the truth

| | §4 budget | measured |
|---|---|---|
| model weights | ~14.5 GB | **14.20 GB** ✓ |
| KV cache @ 8k | ~1.0 GB | **1.94 GB** ✗ (1.9×) |
| model + KV @ 8k | ~15.5 GB | **16.14 GB** |

Weights are accurate. The KV estimate is ~2× low, which eats 0.9 GB of §4's 5.4 GB headroom. Not fatal, and the recommended ~2k cap makes it moot (KV ≈ 0.8 GB there), but §4 is corrected.

---

## Reproduce

```bash
uv run python benchmarks/llm_bakeoff.py       # headline table
uv run python benchmarks/llm_context.py       # prefill + KV growth
uv run python benchmarks/llm_promptcache.py   # cache reuse
```

## Measurement notes and limitations

- **A first pass reported "8k context" figures that were actually 3600 tokens** — the filler was sliced by characters, not tokens. `llm_context.py` re-measures against token-counted prompts. The numbers in this report are the corrected ones.
- tok/s excludes the first token, so prefill is not double-counted into decode throughput.
- Each model gets one warm-up generation before timing; cold-load is timed separately.
- Bytes-per-token for the MoE (2.2 GB) is §3's figure, not independently measured. Effective-bandwidth percentages inherit that assumption.
- Single run per configuration. Decode throughput was stable across context sweeps, but these are not averaged over repeats.
- No quantisation sweep. Only 4-bit was tested, per §2's footprint budget.
