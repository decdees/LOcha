# T0.7 — contention test

**Date:** 30 July 2026 · base M4, 32 GB · ASR + LLM + VOICEVOX all live
**Scope:** complete turn, voice-to-first-audio. VOICEVOX 0.25.2 (CPU) on :50021, speaker 3.

Deviation for the better: the plan used VOICEVOX-synthesised audio as ASR input because no corpus existed. It does now, so this uses the real recordings — the user's own accented voice, which is what the product actually receives.

---

## Headline

**Voice-to-first-audio: p50 2.52 s, p95 3.25 s.** Adding VAD (~150 ms) and network (~30 ms), still unmeasured: **~2.70 s against PRD G1's 1200 ms.**

**G1 is missed by 1.5 s — a factor of 2.2.**

| stage | §5.1 budget | measured p50 | over by |
|---|---|---|---|
| VAD endpoint | 150 ms | *unmeasured* | — |
| ASR | 250 ms | **1250 ms** | 5.0× |
| LLM to first sentence | 450 ms | **700 ms** | 1.6× |
| VOICEVOX | 200 ms | **700 ms** | 3.5× |
| network | 30 ms | *unmeasured* | — |
| **total** | **~1080 ms** | **~2700 ms** | **2.5×** |

**Every measured stage exceeds its budget.** This is not one bad component; §5.1 was optimistic throughout.

---

## Contention itself is not the problem

| stage | standalone | all three live | delta |
|---|---|---|---|
| ASR p50 | 1.25 s | 1.25 s | 0% |
| LLM TTFT p50 | 0.50 s | 0.59 s | +18% |
| LLM decode | 39.4 tok/s | 37.6 tok/s | −5% |

Running all three concurrently costs almost nothing. ARCHITECTURE §4's assumption holds, and it holds for a reason worth recording: **VOICEVOX runs CPU-only** (`voicevox_core::synthesizer: CPUを利用します`), so it contends for a different resource than the GPU-bound ASR and LLM.

The +18% on LLM TTFT is the only real interaction and it is small in absolute terms (90 ms).

**Memory is comfortable:** MLX peak **17.95 GB**, VOICEVOX **0.31 GB**, all-process RSS 13.7 GB. Well inside 32 GB. §4's total is now ~18.3 GB rather than its budgeted 15.5 GB — because T0.3 replaced the ~1.5 GB kotoba model with large-v3 at 3.09 GB — but the headroom absorbs it.

---

## Why G1 cannot be reached by tuning

The three mitigations in `DECISION.md`, costed against measured numbers:

| lever | saving | resulting p50 |
|---|---|---|
| baseline measured | — | 2.70 s |
| **streaming ASR** (removes most of 1.25 s) | ~1.0 s | ~1.70 s |
| **TTS overlap** — *already counted*, TTS fires on sentence 1 | 0 | ~1.70 s |
| **ASR cascade** — helps only the ~35% non-code-switched turns | ~0 at p50 | ~1.70 s |

**Even with perfect streaming ASR, the floor is ~1.7 s — still 1.4× over G1.** The remaining budget is LLM 700 ms + TTS 700 ms + VAD 150 ms, and none of those has an obvious 2× win available:

- **LLM 700 ms** is already using KV-cache reuse and a 2k context cap. The MoE is the fastest viable model measured; the alternative (Qwen3.5-9B) is 1.9× slower.
- **TTS 700 ms** is CPU-only VOICEVOX. ARCHITECTURE §2.2 *chose* VOICEVOX over faster neural TTS deliberately, because its OpenJTalk accent dictionary is inspectable and correct — and for a pronunciation-teaching product the TTS output *is* the model being internalised. Swapping to Kokoro-82M for speed would trade away the thing §2.2 identified as non-negotiable.

**This is a genuine architectural conflict, not an optimisation backlog.** G1 (1200 ms) and §2.2 (deterministic, accent-correct TTS) cannot both hold on a base M4.

### Note on the TTS trade specifically

VOICEVOX exposes per-mora accent data (`ご飯を` → accent type 1, morae `ゴハンオ`), confirming §2.2's rationale and supplying exactly what Phase 3's comparative accent scorer needs as its reference recording. Replacing it with a faster neural TTS would cost that too — the Phase 3 design in §6.1 depends on a paired accent-correct reference existing for free.

---

## Recommendation: G1 needs renegotiating, with evidence

Earlier this report deferred the question pending measurement. It has now been measured, and the honest position is:

- **~1.7 s p50 is the realistic floor** for this stack on this machine, after the streaming-ASR work.
- Options, in order of what they cost:
  1. **Amend G1 to ~1.8 s p50 / 2.5 s p95.** Costs nothing technically. Whether a 1.8 s pause is acceptable in a tutoring conversation is a product judgement, not an engineering one — and it may well be fine, since a human tutor pauses too.
  2. **Keep G1 and drop §2.2's TTS requirement** (Kokoro-82M). Buys perhaps 400 ms, costs pedagogical accent accuracy and Phase 3's free reference recording. Not recommended.
  3. **Keep both and change hardware.** M4 Pro's ~273 GB/s would roughly halve the LLM stage, but ASR and CPU-bound TTS dominate, so the win is smaller than it looks. Out of scope for a $0 project.

**Recommendation: option 1.** The measured evidence says the target was set optimistically, not that the build is wrong.

---

## Outstanding

- **VAD (silero) unmeasured** — 150 ms remains a budgeted guess and it sits on the critical path.
- **Streaming ASR is unimplemented**, so the ~1.0 s saving above is a projection, not a measurement. `Qwen3-ASR` supports streaming natively and is still untested.
- **Thermals (NFR-6) unmeasured.** No 30-minute sustained session.
- **The baseline was not clean.** T0.1 measured 0 swapins/0 swapouts; this run started at 519k/1035k accumulated from prior benchmarks and added ~78k swapouts. Latency did not visibly degrade, but a cold-boot re-run would be a better baseline.
- n = 8 utterances, single run, no repeats.

## Reproduce

```bash
~/Downloads/macos-arm64/run --host 127.0.0.1 --port 50021   # in another shell
uv run python benchmarks/contention.py
```
