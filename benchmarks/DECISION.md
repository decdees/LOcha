# Phase 0 — Decision

**Date:** 30 July 2026 · base M4, 10-core, 32 GB · measured 103.2 GB/s
**Status:** Phase 0 complete. T0.1–T0.5 and T0.7 all closed.

---

## The two required choices

| Layer | Choice | Exact artifact |
|---|---|---|
| **ASR** | whisper-large-v3 on MLX | `mlx-community/whisper-large-v3-mlx` |
| **LLM** | Gemma 4 26B-A4B, 4-bit | `mlx-community/gemma-4-26b-a4b-it-4bit` |

Both are Apache-2.0 and run entirely locally.

### Required configuration, not optional

- **ASR:** `no_repeat_ngram_size=4`. Without it large-v3 degenerates on some utterances into hundreds of repetitions — 557 insertions and 70 s on a single 4.9 s clip.
- **ASR:** clear `generation_config.forced_decoder_ids` before passing `language`/`task`. In transformers 5.x the config silently overrides the kwargs. (MLX runtime is unaffected; this applies if the transformers path is ever used.)
- **LLM:** **KV-cache reuse across turns is an architectural requirement, not an optimisation.** Without it, TTFT p50 is 1.81 s and *grows every turn*; with it, 0.50 s and flat.
- **LLM:** cap context at **~2k, not 8k**. ARCHITECTURE §4 said 8k; at 8k, TTFT is 32.6 s and decode falls to 14.3 tok/s — slower than the dense model it was chosen over.

`src/ocha/speech/asr.py` takes the model as a **config value**, so swapping it is a config edit, not a code change. No runtime-swappable plugin interface. This decision names a *default*, not a permanent commitment.

---

## What the measurements overturned

### ARCHITECTURE §2.1 — reversed

§2.1 chose a Japanese-specialised distillation over a generalist, arguing a generalist "has to beat a purpose-built distillation on the hardest possible input: accented beginner Japanese. Assume it doesn't until measured."

Measured, on this user's voice:

| | pure-JA CER | code-switched CER | Stage 1 |
|---|---|---|---|
| whisper-large-v3 (MLX) | **2.56** | **55.34** | pass, 2/10 unusable |
| kotoba-whisper-bilingual-v1.0 | 11.11 | 66.50 | **fail, 4/10** |
| kotoba-whisper-v2.0 | 17.09 | 74.27 | **fail, 5/10** |

The generalist wins by **6.7×** on pure Japanese. Both specialised models were disqualified by the pre-registered catastrophic-failure gate.

§2.1's reasoning had the domain gap backwards: the distillation is trained on *native* Japanese, and the actual input is a Hindi-L1 absolute beginner, accented and code-switching. That is out-of-domain for the narrow model and in-domain for the multilingual one.

§2.1's speed claim ("6.3× faster") also failed: on MLX, large-v3 is within 14% of kotoba's latency. The runtime dominates the model.

### ARCHITECTURE §9 Risk #1 — resolved, badly

Base M4, not M4 Pro. Measured 103.2 GB/s (86% of the ~120 spec). MoE is mandatory; dense 27B is struck at a 6.9 tok/s ceiling.

### ARCHITECTURE §3 — confirmed in mechanism, wrong in two details

The scattered-read hypothesis is real: the MoE achieves **84%** of peak bandwidth against the dense model's **99%** — a ~15% routing penalty. The conclusion survives; the MoE delivers 39.4 tok/s against 20.5.

But §3 was **pessimistic** on magnitude (predicted 25–35 realistic; measured 39.4), and **wrong to treat tok/s as constant**: decode falls from 37.7 tok/s at 256 tokens to 14.3 at 8k. At 8k the MoE is slower than the dense model it beat.

### ARCHITECTURE §4 — weights accurate, KV estimate half the truth

Weights 14.20 GB against ~14.5 estimated. KV at 8k is **1.94 GB, not 1.0**. Moot at the ~2k cap now recommended.

---

## The unresolved problem: PRD G1

§5.1's latency budget does not close. With the two measured stages and the rest still at their budgeted values:

```
VAD endpoint          150 ms   budgeted, unmeasured
ASR                  1250 ms   MEASURED  (budgeted 250)
LLM TTFT              500 ms   MEASURED, with KV-cache reuse (budgeted 200)
first sentence        400 ms   MEASURED  (budgeted 250)
VOICEVOX              200 ms   budgeted, unmeasured
network                30 ms
────────────────────────────
                     ~2530 ms   vs PRD G1's 1200 ms p50
```

**The two measured stages alone consume 2150 ms of a 1200 ms budget.**

This is the most important open item in Phase 0, and it is not a reason to change either model choice. The trap to avoid: swapping to kotoba saves 150 ms of ASR and costs 6.7× the character errors — and a wrong transcript costs an entire turn, not 150 ms. Likewise `whisper-large-v3-turbo` matches large-v3's speed exactly but its pure-Japanese CER is 31.62%, and the pre-registered guardrail disqualified it.

Latency has to come from **overlapping stages, not from degrading them**:
- streaming ASR on partial audio rather than waiting for endpoint
- Context Builder prompts kept short (measured: a 10-turn conversation is only 560 tokens)
- VOICEVOX synthesis overlapping generation, per §5.2

**Decision: G1 stays as the target. Latency is Phase 2 engineering work, not a PRD renegotiation.**

The 2530 ms figure sums stages measured *in isolation*. The real pipeline overlaps them, and §5.2's rules exist precisely to exploit that. The named work, in priority order:

**Priority revised by T0.7 — the cascade is demoted.**

1. **Streaming ASR** — highest value. Removes most of the 1200 ms ASR stage from the critical path by transcribing *during* speech instead of after endpoint, and it helps **every** turn regardless of language. `Qwen3-ASR` supports streaming natively, which raises the value of closing that untested candidate (1b above).
2. **TTS overlap** — VOICEVOX on sentence 1 while the LLM generates sentence 2 (§5.2 rule 2). Unmeasured until VOICEVOX is installed.
3. **ASR cascade — demoted.** parakeet-ja at 0.204 s would bring the chain to ~1184 ms, which *meets* G1 — but only on turns it can handle. Since 65% of turns are estimated to contain English and parakeet fails those (~8/10 unusable), it would deliver ~1.18 s on ~35% of turns and ~2.2 s on the rest. **p50 would be roughly unchanged.** It is a p35 optimisation that looked like a p50 one.
4. **Short Context Builder prompts** — a 10-turn conversation measured only 560 tokens, so this is cheap to hold.

### T0.7 measured the full chain. G1 is missed by 2.2×.

**Voice-to-first-audio p50 2.52 s, p95 3.25 s**, all three services live. With VAD and network at budgeted values, **~2.70 s against G1's 1200 ms**.

| stage | §5.1 budget | measured p50 | over |
|---|---|---|---|
| ASR | 250 ms | **1250 ms** | 5.0× |
| LLM to 1st sentence | 450 ms | **700 ms** | 1.6× |
| VOICEVOX | 200 ms | **700 ms** | 3.5× |

**Every measured stage exceeds budget.** §5.1 was optimistic throughout, not wrong in one place.

**Contention itself is not the cause** — running all three concurrently costs ~0% on ASR, +18% on LLM TTFT, −5% on decode. §4's assumption holds, and for a concrete reason: VOICEVOX is CPU-only, so it contends for a different resource than the GPU-bound ASR and LLM. Memory is comfortable at 17.95 GB MLX peak + 0.31 GB VOICEVOX.

**Tuning cannot close the gap.** Streaming ASR — the highest-value lever — removes at most ~1.0 s, leaving **~1.70 s, still 1.4× over**. The residue is LLM 700 ms + TTS 700 ms + VAD 150 ms, and neither of the first two has a 2× win available: the LLM already uses KV-cache reuse and a 2k cap, and the TTS is CPU-only VOICEVOX, chosen *deliberately* by §2.2 for its inspectable accent dictionary.

**This is an architectural conflict, not an optimisation backlog: G1 (1200 ms) and §2.2 (deterministic accent-correct TTS) cannot both hold on a base M4.**

### Recommendation: amend G1 to ~1.8 s p50 / 2.5 s p95

This is now an evidenced position rather than a deferral. Options:

1. **Amend G1** to ~1.8 s p50 after the streaming-ASR work. Costs nothing technically. Whether a 1.8 s pause is acceptable in tutoring conversation is a product judgement — a human tutor pauses too. **Recommended.**
2. **Keep G1, drop §2.2's TTS requirement** (Kokoro-82M). Buys ~400 ms; costs pedagogical accent accuracy *and* Phase 3's free reference recording, since §6.1's comparative scorer depends on VOICEVOX output being accent-correct. Not recommended.
3. **Keep both, change hardware.** ASR and CPU-bound TTS dominate, so an M4 Pro wins less than it appears. Out of scope for a $0 project.

The measured evidence says the target was set optimistically, not that the build is wrong.

---

## Open assumptions — not validated, do not treat as settled

1. **§2.1's generalist-*audio-LLM* claim was never tested.** Gemma 4 E4B audio was excluded from T0.3. What was measured is a generalist *ASR*. The original claim about audio LLMs remains an open assumption.
1b. **`Qwen/Qwen3-ASR-1.7B` is untested and is the most valuable gap.** Released Jan 2026, after TASKS.md was written; Apache-2.0; claims open-source SOTA for Japanese; multilingual across 30 languages including Japanese and English, so unlike every Japanese-only candidate it might survive Stage 1; and it supports streaming, which bears directly on the latency problem below. It would not run through the generic `transformers` pipeline (beam-search tensor mismatch, then an embedding-indices dtype error on MPS *and* CPU); its card requires the official `qwen-asr` package in an isolated environment. **Close this before Phase 2.**
1c. **`mlx-community/parakeet-tdt_ctc-0.6b-ja` was measured and disqualified** — Stage 1, ~8/10 code-switched utterances unusable. Worth recording because at **p50 0.204 s** it is the only candidate that meets §5.1's ASR budget, and its pure-Japanese CER (10.26) beats both kotoba models. It fails for the same reason they do: Japanese-only models collapse on code-switched input. This makes an **ASR cascade** (parakeet first, large-v3 fallback on Latin-script/low-confidence output) a live Phase 2 option for recovering latency — see `asr.md`.
2. **The LLM's Japanese is entirely unverified — T0.5 has not run yet.** Neither constraint compliance (vocabulary, register, reply length, `[GRAMMAR_QUERY]` sentinel) nor naturalness has been measured. The LLM choice above rests on *throughput*, not on output quality. Even once T0.5 runs it will only verify mechanical compliance; naturalness needs a Japanese-competent reviewer the project does not have.
3. **n = 20, one speaker, one session.** Sufficient to separate 2.56 from 17.09; **not** sufficient to resolve 1–2 CER points, and no basis for confidence intervals.
4. **The ASR latency comparison crosses runtimes.** Accuracy was measured on one backend for all candidates (fair); the winner runs on MLX, which has no `kotoba-whisper-bilingual-v1.0` conversion. kotoba on MLX would likely be faster than measured.
5. **Corpus utterance 09 is suspect ground truth** — all four models returned `これは…` where the reference says `トイレは…`. Unanimity across independent models points at the audio.
6. **Thermals untested.** NFR-6 (30-minute session without >20% latency degradation) has no measurement. Every number here is from short bursts.
7. **VAD (silero) is unmeasured** and sits on the critical path; 150 ms is still a budgeted guess. **Streaming ASR is unimplemented**, so the ~1.0 s saving above is a projection, not a measurement.
8. **The T0.7 baseline was not clean.** T0.1 measured 0 swapins/0 swapouts; by T0.7 the machine had accumulated 381k/966k from preceding benchmarks, and the run itself added ~32.9k swapouts (~525 MB). Latency did not degrade, but these figures come from a machine that has been running models for hours, not a fresh boot.

---

## Gate

TASKS.md requires this file to name a specific ASR model and a specific LLM model. It does:
`mlx-community/whisper-large-v3-mlx` and `mlx-community/gemma-4-26b-a4b-it-4bit`.

**Phase 1 may start.** All Phase 0 tasks are closed and both models are named. One product decision is outstanding but does not block Phase 1, which has no voice path:

- **PRD G1 needs amending to ~1.8 s p50** on the evidence above. Decide before Phase 2.

Superseded, both now closed:

- **T0.5 (generation probe)** — the LLM was selected on throughput alone. If Gemma 4 cannot hold the vocabulary constraint or emit the `[GRAMMAR_QUERY]` sentinel reliably, the choice is wrong regardless of tok/s, and T1.5's firewall is built on sand.
- ~~**T0.7 (contention)**~~ — **closed.** Full chain measured at p50 2.52 s. Contention is negligible; §4's memory assumption holds. The finding is that G1 is unreachable by tuning and needs amending (above).

## Reports

`hardware.md` · `asr.md` · `llm.md` · `generation-probe.md` · `contention.md` *(partial — no TTS)*
