# T0.3 — ASR bake-off

**Date:** 30 July 2026 · base M4 · corpus: 20 utterances, 10 pure JA / 10 code-switched, this user's voice
**Decision rule and CER methodology were pre-registered in TASKS.md before the corpus was recorded.**

---

## Recommendation

**`mlx-community/whisper-large-v3-mlx`** — whisper-large-v3 weights on the MLX runtime.

It is the only candidate to pass the pre-registered Stage 1 gate, and it also wins Stage 2 outright: best pure-Japanese CER by a factor of 4–7, best weighted CER, and latency indistinguishable from the "fast" distilled models.

**This contradicts ARCHITECTURE §2.1, which is amended.**

---

## Results

| model | runtime | pure-JA CER | code-sw CER | weighted | p50 | p95 | Stage 1 |
|---|---|---|---|---|---|---|---|
| **whisper-large-v3** | **MLX** | **2.56** | **55.34** | **36.87** | **1.25 s** | 1.53 s | **PASS** (2/10) |
| whisper-large-v3 | transformers+MPS | 4.27 | 58.25 | 39.36 | 3.34 s | 4.58 s | PASS (1/10) |
| whisper-large-v3-turbo | transformers+MPS | 31.62 | 55.34 | 47.04 | 1.25 s | 1.58 s | **FAIL** — guardrail |
| kotoba-whisper-bilingual-v1.0 | transformers+MPS | 11.11 | 66.50 | 47.11 | 1.10 s | 1.27 s | **FAIL** (4/10) |
| kotoba-whisper-v2.0 | transformers+MPS | 17.09 | 74.27 | 54.26 | 1.13 s | 1.26 s | **FAIL** (5/10) |

Weighted = 0.65 × code-switched + 0.35 × pure-JA, per the pre-registered estimate of English-containing turns.

### How the pre-registered rules applied

**Stage 1 — catastrophic-failure gate** (>2 of 10 code-switched utterances unusable as LLM input ⇒ disqualified):

- **kotoba-whisper-v2.0 — 5/10 unusable.** #11 → `イト、どういうスイット、レシピン、ジャパネース?`; #17 dropped the entire English clause *and* the closing, returning only `ごめんなさい。`; #13, #16, #18 similarly destroyed.
- **kotoba-whisper-bilingual-v1.0 — 4/10 unusable.** Better than v2.0 and it *did* preserve the English clause verbatim in #17 (`ごめんなさい i don't understand…`), which is the bilingual training showing. Still fails.
- **whisper-large-v3 (MLX) — 2/10.** Passes at the threshold. #16 lost the price question (`までいくら` → `メイドエクォール`) and #18 dropped everything but `カドデスカ`.

**Guardrail** (pure-JA CER more than +3.0 points above the best performer ⇒ disqualified): **turbo fails at +29.1 points** (31.62 vs 2.56). turbo matches large-v3's speed and its code-switched score, but its 4-layer decoder gives up almost all of the Japanese accuracy. Exactly the trade the guardrail exists to refuse.

**Looping floor** (insertions > 1.5 × substitutions on any subset): all candidates pass **as finally run**. See the harness note below — they did not, before a decoding fix.

### Post-hoc: candidates that postdate TASKS.md (July 2026 web/HF check)

TASKS.md's candidate list predates several releases. Checked and, where runnable, measured against the same corpus and the same pre-registered rules.

| candidate | released | pure-JA | code-sw | p50 | verdict |
|---|---|---|---|---|---|
| `mlx-community/parakeet-tdt_ctc-0.6b-ja` | NVIDIA, CC-BY-4.0 | 10.26 | 74.24 | **0.204 s** | **FAIL Stage 1** (~8/10) |
| `Qwen/Qwen3-ASR-1.7B` | Jan 2026, Apache-2.0 | — | — | — | **UNTESTED** — see below |
| `mistralai/Voxtral-Mini-4B-Realtime-2602` | Feb 2026, Apache-2.0 | — | — | — | untested |

**parakeet-ja is the interesting failure.** At **p50 0.204 s** it is the only candidate that *meets* §5.1's 250 ms ASR budget — 6× faster than the recommended model — and its pure-Japanese CER of 10.26 beats both kotoba variants. But its code-switched output is the worst measured: `東京` → `土器` ("pottery"), `七時` → `6年71`, and the English spans dissolve entirely. Roughly 8/10 unusable.

That is the same failure as kotoba, and it confirms the §2.1 lesson rather than complicating it: **every Japanese-only model in this bake-off collapses on code-switched input, regardless of how good its Japanese is.** The property that matters for this product is multilinguality, not Japanese specialisation.

### T0.8 — Qwen3-ASR-1.7B measured. Disqualified on every axis.

| | pure-JA CER | code-sw CER | weighted | p50 | Stage 1 |
|---|---|---|---|---|---|
| whisper-large-v3 (MLX) | **2.56** | **55.34** | **36.87** | **1.25 s** | pass (2/10) |
| Qwen3-ASR-1.7B *(best config)* | 48.72 | 58.25 | 54.91 | 1.72 s | **fail (3/10)** |

**19× worse on pure Japanese, slightly worse on code-switched, and 38% slower.** It fails the Stage 1 gate at 3/10 unusable (`東京 station までいくらですか` → `東京ステーションメイトイクルデスuka`, `ごめんなさい` → `Governmentなさい`, `Can I pay with カード` → `彼はフェウェドカドゥデシュカ`) and misses the pure-JA guardrail by **+46.2 points** against a +3.0 limit.

**It does not help latency either**, which was the entire reason it was promoted to a task. The streaming lever this candidate was supposed to provide is gone with it.

#### Two findings worth keeping

**1. Auto-detect transcribes this speaker as Hindi, in Devanagari.**

```
ref              すみません、駅はどこですか。
auto-detect      सुमिरन सेन एकीवा दोगुं तेसु का
ref              ありがとうございます、また明日。
auto-detect      अरिगातोगरिमा माता अशिता
```

The model's language ID hears Hindi-accented Japanese and concludes Hindi, then transcribes phonetically in Devanagari. This is the L1 interference in the project's domain notes showing up in a place nobody predicted — not in the learner's grammar, but in the ASR's language classifier. Any multilingual ASR used here must have its language **forced**, never auto-detected.

**2. Configuration moved it 33 CER points, and it still lost.**

| config | pure-JA CER |
|---|---|
| auto-detect | — (Devanagari, unscoreable) |
| forced Japanese | 82.05 |
| forced Japanese + context hint | **48.72** |

Forcing the language alone still produced katakana-rendered Japanese (`スミマセン`, `ハリガト`). A context hint asking for kanji and hiragana recovered much of the orthography. The number reported above is its **best** configuration — it was not disadvantaged by a lazy invocation, which is the same discipline applied to whisper's `forced_decoder_ids`.

Raw outputs: `asr-qwen3asr-t08.json`.

---

**Qwen3-ASR-1.7B was the significant gap.** It claims open-source SOTA for Japanese (independent benchmark: CER 0.140 vs whisper-large-v3-turbo's 0.184 on native media speech) and it is multilingual across 30 languages including Japanese *and* English — so unlike parakeet and kotoba it might survive Stage 1. It also supports streaming, which is directly relevant to the latency problem. It was measured in T0.8 above, in an isolated venv with the official `qwen-asr` package, and **disqualified**. The gap is now closed.

Note the benchmark that motivated this check ranks `whisper-large-v3-turbo` second overall for Japanese — while turbo scored 31.62% here and was disqualified by the guardrail. Not a contradiction: that benchmark uses native conversational media, this corpus is an accented L2 beginner. It is a reminder that published Japanese ASR rankings do not transfer to this speaker, which is the entire reason Phase 0 exists.

### An architectural option this opens: ASR cascade

parakeet at 0.204 s and large-v3 at 1.25 s are not mutually exclusive. A turn could run parakeet first, and fall back to large-v3 only when the output contains Latin script, low confidence, or known-garbage markers. Most turns would land inside the latency budget; only code-switched ones would pay the full cost. Recorded as a Phase 2 option, not a Phase 0 recommendation — it needs a fallback trigger that is itself measured, and it doubles the resident ASR footprint.

---

## ARCHITECTURE §2.1 is wrong on both of its claims

§2.1 argues for kotoba-whisper-v2.0 over a generalist, on two grounds. Both fail against measurement:

1. **"A generalist has to beat a purpose-built distillation on the hardest possible input: accented beginner Japanese. Assume it doesn't until measured."** Measured: the generalist wins by **6.7×** on pure Japanese (2.56 vs 17.09) and by **19 points** on code-switched. The distillation is not merely matched, it is beaten decisively on exactly the input §2.1 predicted would favour it.

2. **"roughly 6.3× faster than large-v3."** True in the abstract, not in practice here. On MLX, large-v3 runs at **p50 1.25 s against kotoba's 1.10 s on transformers** — a 14% difference. The runtime dominates the model. §2.1's speed argument was comparing a distilled model on a slow runtime against a full model on the same slow runtime, and concluding the distillation was necessary.

Why the distillation loses: it is trained on native Japanese. This corpus is a Hindi-L1 beginner speaking accented Japanese and code-switching mid-sentence. That is out-of-domain for a narrow distillation and in-domain for a multilingual generalist. §2.1's reasoning inverted the direction of the domain gap.

---

## The latency budget does not close

§5.1 budgets **250 ms** for ASR. Measured **1250 ms** with the recommended model — 5× over. Combined with T0.4's measured LLM stages:

```
VAD endpoint          150 ms   (budgeted, unmeasured)
ASR                  1250 ms   MEASURED, was budgeted 250
LLM TTFT              500 ms   MEASURED (with KV-cache reuse), was budgeted 200
first sentence        400 ms   MEASURED
VOICEVOX              200 ms   (budgeted, unmeasured)
network                30 ms
────────────────────────────
total                ~2530 ms   vs PRD G1's 1200 ms p50
```

**PRD G1 is at serious risk — the two measured stages alone consume 2150 ms of a 1200 ms budget.** Note the trap this table sets: choosing kotoba to save 150 ms of ASR would cost 6.7× the word errors, and a wrong transcript costs a whole turn. Latency must be recovered elsewhere (streaming ASR on partial audio, shorter prompts, VOICEVOX overlap), not by degrading the transcript. T0.7 measures the real chain.

---

## Harness errors found and fixed — both would have produced wrong conclusions

Recorded because in each case the first run produced a confident, plausible, wrong table.

1. **Language forcing was silently ignored.** `generate_kwargs={"language":"ja"}` has no effect in transformers 5.x when the model's own `generation_config.forced_decoder_ids` is set — forced-`ja` and auto-detect returned byte-identical output. Fixed by clearing it, with an assert so it cannot regress.

2. **Decoder looping was a harness artifact, not a model property.** Unmitigated, large-v3 degenerated on #18 into ~200 repetitions of `カードレス` — **557 insertions, 70.5 s for one utterance** — which tripped the pre-registered looping floor and would have disqualified the eventual winner. `no_repeat_ngram_size=4` fixes it: same utterance, 4.07 s. Window 4 is deliberately conservative; Japanese legitimately repeats short spans.

   > ### ⚠ CORRECTION (T2.6, 31 July 2026): the bake-off was confounded
   >
   > This paragraph claimed the mitigation was "applied uniformly to all candidates so the comparison stays fair". **It was not, and it could not have been.** `no_repeat_ngram_size` is a `transformers` generation parameter. The kotoba candidates ran on `transformers`; whisper-large-v3 ran on **MLX**, where the parameter does not exist — `mlx_whisper.transcribe` raises `DecodingOptions.__init__() got an unexpected keyword argument 'no_repeat_ngram_size'`. Discovered when the production ASR service was written against the same call.
   >
   > So the winner was measured **without** repetition mitigation and the losers **with** it. The confound points *against* the winner: large-v3 scored 2.56% CER while carrying a failure mode its competitors had suppressed. `benchmarks/contention.py`, which produced the shipped 1.25 s and the T0.7/T0.9 chain figures, passed no such argument either — so those numbers are consistent with the shipped configuration, and 2.56 is the un-mitigated figure.
   >
   > **The decision stands; the margin does not.** A 6.7× gap (2.56 vs 17.09) is far too large to be an artifact of one decoding flag, and the Stage 1 catastrophic-failure gate disqualified both kotoba variants on code-switched output, which repetition mitigation does not touch. But "6.7×" should not be quoted as a measured ratio — it compares two different inference backends with two different decoding configurations. Not re-run: the conclusion does not change and the compute is better spent on Phase 2.
   >
   > MLX's own guards are `compression_ratio_threshold` and temperature fallback, both on by default, plus `condition_on_previous_text=False` so a loop cannot persist across turns.
   >
   > **Correction, same day: they are not sufficient.** An end-to-end run
   > (`benchmarks/voice-loop.md`) produced `火が火に火に火に火に火に火に火に火に火に火に`
   > from one utterance — the same degenerate repetition T0.3 saw on transformers,
   > on the MLX path, with those guards active. An earlier version of this note
   > claimed no looping had been observed here; that was true of the runs made at
   > the time and is now false. The failure mode survives the backend change, and
   > the mitigation that fixed it does not exist on this backend. Open.
   >
   > Now standing constraint 8: never compare models across different backends or decoding-parameter sets.

3. **The corpus itself was invalid on the first attempt** and is documented in `corpus/CHECKLIST.md`. `record.py --verify` now gates T0.3.

---

## Limitations, stated plainly

- **2.56 is not a load-bearing figure**, for two reasons now. The first is the confound above: it was measured without the repetition mitigation its competitors had. The second is the input distribution. It was measured on **rehearsed read-aloud speech in a quiet room** — the speaker reading a prepared romaji line, one utterance at a time, with retakes allowed. Real conversational input is a different distribution: hesitation, self-correction, mid-sentence restarts, trailing off, false starts, thinking noises. **None of that is tested.** Expect the real-world figure to be materially worse, and do not let 2.56 be quoted as the ASR error rate of the product.
- **The latency column is not apples-to-apples.** Accuracy was measured for all candidates on one backend (transformers+MPS), which is fair. The winning configuration is on MLX, which has no equivalent for `kotoba-whisper-bilingual-v1.0` — no MLX conversion exists. kotoba on MLX would likely be faster than 1.10 s. This does not change the decision, since large-v3 wins accuracy by 4–7×, but "large-v3 is as fast as kotoba" compares across runtimes.
- **n = 20, one speaker, one session.** Enough to separate 2.56 from 17.09. **Not** enough to resolve differences of 1–2 CER points, and no basis at all for a confidence interval. Treat the ordering as robust and the exact values as indicative.
- **The corpus is 50/50 by construction; real usage is not.** Aggregate CER is the least meaningful column. Stage 1 and the per-subset columns carry the decision.
- **§2.1's other claim remains untested.** Gemma 4 E4B audio was excluded from this bake-off, so "a specialised ASR beats a generalist *audio LLM*" was never measured. What was measured is a generalist *ASR*. Open assumption.
- **Utterance 09 is suspect ground truth.** All four models returned `これは…` where the reference says `トイレは…`. Unanimous agreement across independent models points at the audio, not the models. Left as-is (it penalises all candidates equally) but flagged.
- Single run per configuration; no repeats, no averaging.

## Reproduce

```bash
uv run python benchmarks/corpus/record.py --verify   # gate: expect 0 of 20
uv run python benchmarks/asr_bakeoff.py              # the three-model table
```
