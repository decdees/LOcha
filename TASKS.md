# Ocha — Task List

Sequenced. Each task has context, acceptance criteria, and a done checkbox.
**Do not start a phase until the previous phase's gate is passed.**

Reference documents in this repo:
- `PRD.md` — what and why
- `ARCHITECTURE.md` — how (component choices, latency budget, memory budget)

---

## Phase 0 — Bake-off *(gate: no application code until complete)*

> **Why this phase exists.** Every model choice in `ARCHITECTURE.md` is a hypothesis based on published benchmarks, not measurements on this machine with this user's voice. Building on unverified assumptions is the single most expensive mistake available here. Phase 0 costs one evening and can invalidate half the architecture.

### T0.1 — Hardware identification
- [x] Record chip variant (`sysctl -n machdep.cpu.brand_string`), core counts, and measured memory bandwidth.
- **Why it matters:** base M4 (~120 GB/s) vs M4 Pro (~273 GB/s) is a >2x gap that decides whether a dense model is viable at all.
- **Acceptance:** `benchmarks/hardware.md` states the variant and measured bandwidth. ✅ **Base M4, 103.2 GB/s measured.**

### T0.2 — Voice sample corpus
- [x] Recording kit delivered: `record.py` (device pinned, level-guarded, resumable), `transcripts.json` (20 entries, ground truth pre-filled), `CHECKLIST.md`. **Awaiting the user's recording session.**
- [x] 10 pure Japanese, 10 deliberately code-switched Japanese/English — sentences authored, covering ん, geminate っ, long ー, katakana, counters, and fillers えーと/あの/えっと.
- [x] Ground truth pre-filled (sentences authored before recording, so truth is known in advance). `record.py` prompts per take and writes back any deviation.
- **Acceptance:** `benchmarks/corpus/` with 20 WAVs and `transcripts.json`.

### T0.3 — ASR bake-off
- [x] Benchmark on the T0.2 corpus: `kotoba-whisper-v2.0`, `kotoba-whisper-bilingual-v1.0`, `whisper-large-v3`.
- [x] Report CER (overall, pure-JA subset, code-switched subset) and wall-clock latency per utterance.
- **Why bilingual is included:** a beginner code-switches constantly, and the Japanese-only model degrades badly on that input.
- **Why large-v3 is included:** as a diagnostic baseline. Distilled two-layer decoders can loop or hallucinate on out-of-domain audio; large-v3 tells you whether a weird output is the distillation or the microphone.

**Pre-registered CER methodology** (fixed before recording): NFKC normalize, strip all whitespace (Japanese is unspaced; ASR whitespace is noise), strip `、。！？「」,.!?`, lowercase ASCII, then Levenshtein / len(reference). Via `jiwer`, for the substitution/insertion/deletion split — that is what makes decoder looping visible instead of merely inflating one number.

**Pre-registered decision rule** (fixed before recording — a rule written after seeing data is not a rule). Weighted CER alone is rejected as the primary criterion: a Japanese-only model handed English does not degrade gracefully, it garbles or drops the English span, the LLM receives nonsense, and the turn dies. That is not commensurable with a 2-point pure-JA CER loss, which is cosmetic. Averaging them hides the only difference that matters.

- **Stage 1 — catastrophic-failure gate, before any CER is computed.** For each model, judge each of the 10 code-switched utterances individually as **usable / unusable as LLM input** (unusable = English span dropped entirely, or rendered as nonsense katakana). The report shows **the actual output text for all 10, per model** — not a percentage. **More than 2 of 10 unusable → disqualified**, regardless of aggregate CER.
  - **Faithful katakana transliteration is USABLE, not a catastrophic failure.** `receipt` → `レシート` is a correct rendering the LLM can act on; only a *dropped* span or genuinely nonsense output fails Stage 1. But it is still a CER error against the English-orthography reference in Stage 2. `transcripts.json` carries an `alt_ok` list per affected utterance for exactly this: it gates Stage 1 without being normalized away in Stage 2. Collapsing the two would hide a real difference between the models — a Japanese-only model that transliterates cleanly is usable; one that drops English is not; both would score similar CER.
- **Stage 2 — among survivors only.** Weighted CER = `0.65 × code-switched + 0.35 × pure-JA`, lowest wins. 0.65 is the estimated share of real turns containing English.
- **Guardrail.** Disqualify any model whose pure-JA CER exceeds the best pure-JA performer by more than **+3.0 absolute points**. Pure Japanese is the target end state; a model weak there gets *worse* over time, not better.
- **Looping floor.** Disqualify any model with an insertion rate above 1.5× its substitution rate on any subset.
- **Tie-break** (within 0.5 CER points): lower p95 per-utterance latency.
- The report must state that **the corpus is 50/50 by construction and real usage is not, so aggregate CER is the least meaningful column.** Stage 1 and the per-subset columns carry the decision.

- **Acceptance:** `benchmarks/asr.md` with the Stage 1 per-utterance outputs, the Stage 2 table, and a stated recommendation naming an exact HF repo ID.

### T0.4 — LLM bake-off
- [x] Benchmark via MLX: `mlx-community/gemma-4-26b-a4b-it-4bit` and `mlx-community/Qwen3.5-9B-4bit`. ~~plus `qwen3.5:27b` **only if** T0.1 shows M4 Pro-class bandwidth~~ — **T0.1 resolved: base M4 at 103.2 GB/s, so `qwen3.5:27b` is skipped** (6.9 tok/s ceiling).
- [x] Measure: cold load time, time-to-first-token, sustained tok/s, resident memory at 8k context.
- [x] Derive effective bandwidth (`bytes_per_token × tok/s`) and compare against T0.1's measured 103.2 GB/s. MoE expert routing scatters reads, so a shortfall against the 46.9 tok/s ceiling is expected — quantify it.
- [x] Quit Docker Desktop first (8.32 GB VM reservation) and record what else is resident.
- **Acceptance:** `benchmarks/llm.md` with a table and a recommendation.

### T0.5 — Generation probe ✅ *(replaces the former grammar-explanation probe)*
- [x] Build one fixed Context-Builder-shaped prompt from `ARCHITECTURE.md` §7.1 verbatim, with a frozen `KNOWN` list (~40 items) and a frozen `TARGET` item.
- [x] 20 fixed user turns: 15 conversational + 5 grammar questions. Greedy decoding / fixed seed — 20 distinct prompts, not 20 samples of one, so the run is deterministic and re-runnable.
- [x] Score each reply mechanically on four checks:
  - stayed inside the vocabulary constraint (`fugashi` + `unidic-lite`; every content word ∈ `KNOWN` ∪ at most one new word, glossed)
  - politeness register consistent within the reply — compare **sentence-final** forms only (です/ます/ました vs plain だ/る/た). Plain form is legitimate in subordinate clauses, so mid-sentence occurrences are not violations
  - obeyed the 1–2 sentence limit (count 。！？ terminators)
  - emitted `[GRAMMAR_QUERY]` instead of answering, on the 5 grammar turns — sentinel present **and** no substantive text alongside it
- [x] Save all replies per model to a reviewable file for later naturalness review.
- **Why this and not explanation quality:** FR-5 firewalls the model's explanations, so explanation quality cannot change any decision. What the product depends on is whether the model *produces* acceptable Japanese under constraint. All four checks are decidable without Japanese grammar expertise.
- **Known blind spot, to be stated in the report:** these checks verify constraint *compliance*, not naturalness. A model can pass all four and still produce stilted Japanese. Phase 0 cannot detect that; it needs a Japanese-competent reviewer. Record as an open assumption, do not bury it behind a passing table.
- **Acceptance:** `benchmarks/generation-probe.md` with a per-model pass rate on each of the four checks, plus the held-back reply sample.

### T0.7 — Contention test ✅
- [x] Load kotoba-whisper, the LLM, and VOICEVOX simultaneously and warm. Script one synthetic turn end to end: audio → ASR → Context Builder prompt → LLM → sentence chunker → VOICEVOX first sentence.
- [x] Input audio is VOICEVOX-synthesized, so no corpus is needed and this runs alongside T0.4. **Any CER from T0.7 is meaningless** — this is clean native-synthetic speech. T0.7 measures latency and memory only.
- [x] VOICEVOX runs as the **native macOS arm64 engine, not Docker.** Docker Desktop reserves ~8.3 GB for its Linux VM, which exceeds `ARCHITECTURE.md` §4's entire headroom, and it is not the production topology (NFR-5 runs VOICEVOX under `launchd`).
- [x] Record what else is running. An unrecorded baseline makes the memory column unreproducible.
- **Why:** §5's latency budget and §4's memory budget both assume all three resident at once on one memory bus on a base M4. Measuring each alone cannot detect the interaction. This is the untested architectural risk.
- **Acceptance:** `benchmarks/contention.md` with per-stage wall clock, the **contention delta** against the standalone T0.3/T0.4 numbers, peak summed RSS vs §4's budget, and `vm_stat` swapins/swapouts across the run.

### T0.8 — Qwen3-ASR-1.7B ✅ *(promoted from "known gap")*
- [x] `Qwen/Qwen3-ASR-1.7B` (Jan 2026, Apache-2.0) in an **isolated venv** — its card requires the official `qwen-asr` package and the generic `transformers` pipeline fails on it (beam-search tensor mismatch, then an embedding-indices dtype error on MPS *and* CPU).
- [x] Score against the T0.2 corpus using the same pre-registered Stage 1 / Stage 2 rules.
- **Why it is now a task, not a gap:** ASR is the dominant latency cost (1250 ms, 5× over budget) and Qwen3-ASR streams natively, which is the top latency lever. **PRD G1 cannot be amended responsibly until this lands.**
- **Acceptance:** numbers appended to `benchmarks/asr.md`; DECISION.md updated if it wins.

### T0.9 — Cold-boot re-run of T0.7 ✅
- [x] Rebooted, re-ran on a 15-minute-old machine, both shipped and prior LLM.
- **Why:** the 2.52 s figure was taken with ~525 MB of accumulated swap after hours of benchmarking. That number is what a G1 amendment would rest on, so it needs a clean baseline.
- **Acceptance:** `contention.md` records both. ✅ **Clean gemma 2.49 s vs dirty 2.52 s — a 1% delta, so T0.7 was NOT distorted. Shipped Qwen 3.03 s.**

### T0.6 — Phase 0 report
- [x] Consolidate into `benchmarks/DECISION.md`: chosen ASR, chosen LLM, and any `ARCHITECTURE.md` revisions required.
- [x] State plainly in the body, not a footnote: `ARCHITECTURE.md` §2.1's justification for a Japanese-specialised ASR over a generalist audio LLM was **never tested**, because Gemma 4 E4B audio was excluded from T0.3. Open assumption, not a validated finding.
- [x] Record as a Phase 2 design constraint: `src/ocha/speech/asr.py` takes the model as a **config value** so swapping it is a config edit, not a code change. No runtime-swappable plugin interface.
- [x] Update `ARCHITECTURE.md` in place where measurements contradict it.

> **GATE:** `benchmarks/DECISION.md` exists and names a specific ASR model and a specific LLM model. Do not proceed otherwise.

---

## Phase 1 — Text loop

> **Why text first.** The Context Builder and grammar firewall are the two components that carry the product's actual value and the most logic risk. Debugging them through a voice pipeline is masochism. Everything built here is reused verbatim in Phase 2; only the transport changes.
>
> Phase 1 produces no user-facing surface. That is deliberate; see PRD §10.

### T1.1 — Project scaffold ✅
- [x] `uv`-managed Python project. FastAPI + Uvicorn. `Makefile` with `dev`, `test`, `lint`, `fmt`, `check`.
- [x] `ruff` + `mypy --strict` configured and passing. `benchmarks/` and `*.md` excluded — Phase 0 output whose numbers are cited in reports; reformatting them risks editing published figures.
- **Acceptance:** `make dev` serves `/health`. ✅ **200 in 1.3 ms.**

### T1.2 — Data model ✅
- [x] SQLite schema (WAL mode) via a numbered-SQL migrator. Tables: `items`, `reviews`, `sessions`, `turns`, `utterances`, `pronunciation_scores`, plus `unauthored_grammar` for T1.5's miss log.
- [x] `items` carries FSRS state, item type (vocab / grammar / phrase), and content.
- [x] `turns` stores user transcript, tutor reply, target item IDs, derived rating, and whether the firewall fired.
- **Acceptance:** migrations run clean and are idempotent; seed inserts 50 starter vocabulary items. ✅ **9 tests green.**

### T1.3 — FSRS integration ✅
- [x] Wrapped `py-fsrs`. `due_items(limit)`, `known_items(min_reps)`, `lowest_stability(limit)`, `record_review(item_id, rating)`, plus `retrievability(item_id)`.
- [x] PRD FR-8 derivation table implemented. The accent cap is **PROVISIONAL and disabled by default** per the amended FR-8 — code path and tests exist, only the score is absent, so T3.6 enabling it is a flag flip.
- **Acceptance:** every rating path unit-tested; 30-day simulation gives 0 → 2 → 13 → 55 → 179 → 535 → 1436 → 3407 day intervals. ✅ **27 tests green.**

### T1.4 — Grammar reference ✅
- [x] `data/grammar.json` conforming to PRD FR-7 — 20 entries covering the T0.5 probe topics.
- [x] `hindi_contrast` on 15 of 20, only where the analogy holds. `interference_warning: true` on 9, including all three named: ने/が (`particle_wa_ga`), gender agreement (`particle_no`), stress-vs-pitch (`pitch_accent_basics`).
- [x] Pydantic loader, fail-fast, reports **every** malformed entry rather than just the first. `extra="forbid"` so a typo'd key is an error, not a silent drop.
- **Acceptance:** 20 valid entries load; malformed entries rejected. ✅ **45 tests green.**

### T1.5 — Grammar firewall ✅ *(critical path)*
- [x] Detect the literal `[GRAMMAR_QUERY]` sentinel in model output. **Assert it is the ENTIRE payload, not merely present** — T0.5 observed the sentinel emitted alongside 399 chars of grammar explanation when thinking was enabled; a substring test would have passed that and leaked it.
- [x] On detection, suppress the model's response entirely and serve from `grammar.json`. Verbatim — a test asserts every field is byte-identical to the curated entry.
- [x] On reference miss: return "not yet documented", log to `unauthored_grammar`. **Never** falls back to generation. Entry resolution is a deterministic trigger table — using the LLM to resolve would put it back in the correctness path.
- **Acceptance:** a test asserts that when the sentinel fires, no model-generated text reaches the response payload. This test must never be weakened. ✅ **Formulated as derivability — every user-visible field byte-identical to the reference — not a phrase blacklist. 72 tests green.**

### T1.6 — Context Builder ✅
- [x] Assembles the system prompt from `known_items`, `due_items`, `lowest_stability`; history is carried on `TurnContext` for T1.8.
- [x] Reply length enforced by prompt *and* `MAX_REPLY_TOKENS`.
- [x] **REGISTER line included** — T0.5 measured it worth 9/15 → 14/15.
- [x] **§7.1's template corrected, not copied.** As written it was self-contradictory against real FSRS state: "use only words from KNOWN" while `TARGET` listed due items, which are typically *not* known. `TARGET` is now split into PRACTISE (known, weak) and INTRODUCE (not known, at most one per reply), which encodes FR-3 explicitly.
- [x] Context cap 2048, not 8k — T0.4 measured TTFT 32.6 s and decode 14.3 tok/s at 8k.
- **Acceptance:** snapshot tests over three FSRS states (cold start, some known, some weak). ✅ **83 tests green.**

### T1.7 — LLM service ✅
- [x] MLX-backed local inference, `mlx-community/gemma-4-26b-a4b-it-4bit` per `benchmarks/DECISION.md`, as a **config value** (`OCHA_LLM_MODEL`).
- [x] Loaded once in the FastAPI lifespan, kept warm. Streaming interface present for Phase 2's sentence chunker.
- [x] **`enable_thinking=False`.**
- [x] **KV-cache reuse across turns**, keyed on a hash of the system prompt and rebuilt when it changes — reusing a cache built on a different prefix would silently feed the model the wrong context.
- [x] Fails loudly if the model is unavailable; `generate()` before `load()` raises rather than lazy-loading.
- **Acceptance:** `/health` reports model loaded and resident memory. ✅ **`model_loaded: true`, `resident_memory_gb: 14.2`** — matching T0.4. End-to-end smoke: cold load 8.9 s, then 2.65 → 1.08 → 0.97 s per turn as the cache warms; a grammar question produced a clean bare sentinel and the firewall served `particle_wa_ga`. **96 tests green.**

### T1.8 — Turn orchestration ✅
- [x] `POST /turn` — accepts text, returns tutor reply (or firewalled grammar answer), targets, derived ratings and usage.
- [x] Usage detection via `fugashi` lemmas, not substring matching — 食べる appears as 食べます/食べました/食べて. Handles unidic's orthography drift (ご飯→御飯), loanword lemma suffixes (コーヒー-coffee), and multi-token items (お茶 = 接頭辞+名詞).
- [x] **FR-8's "avoided" narrowed to *elicited* items.** Read literally it would rate five items `Again` every turn — a 1–2 sentence reply cannot exercise six targets. Avoidance now requires the tutor to have put the item in play.
- [x] Grammar-query turns are not scored — asking a question is not a production attempt.
- **Acceptance:** 10-turn integration test asserts FSRS state evolves; plus HTTP-level tests. ✅ **118 tests green. Live: p50 1.14 s over 6 turns, firewall fired on both grammar questions, 0 unauthored misses.**

### T1.9 — *(deleted)* Minimal PWA
Cut. Phase 1 exposes `POST /turn` and nothing else. The PWA is built once, voice-first, as T2.7. See PRD §10.

### T1.10 — No-network test ✅
- [x] `src/ocha/net_guard.py` intercepts `socket.connect`/`connect_ex` and refuses anything outside loopback and the Tailscale CGNAT range (100.64.0.0/10).
- [x] The guard is itself proven: a deliberate outbound connection raises `OutboundNetworkError`, and a test asserts the patch is removed afterwards so it cannot silently disable later tests.
- **Acceptance:** a full `/turn`, a `/health`, and a firewalled grammar turn all complete with zero outbound connections. ✅ **125 tests green.**

> **GATE.** T1.8's 10-turn integration test passes and T1.10 (no-network) passes. Proceed directly to Phase 2 — this is not a go/no-go, and Phase 1 is not a stopping point. See PRD §10.

---

## Phase 2 — Voice loop *(in scope)*

### T2.1 — Pipecat scaffold, pinned version, WebRTC transport
- [ ] Verify audio quality on day one. `SmallWebRTCTransport` had a choppy-audio regression around v0.0.62 — pin a version that sounds correct and record which.

### T2.2 — silero-vad endpointing and barge-in
### T2.3 — ASR service wrapping the Phase 0 choice
### T2.4 — Streaming LLM into a sentence chunker (split on 。！？)
### T2.5 — VOICEVOX service, synthesis per sentence
### T2.6 — End-to-end latency instrumentation

### T2.7 — PWA, voice-first *(absorbs the deleted T1.9)*
- [ ] React PWA: continuous-listening voice UI, live transcript (user + tutor), furigana on tutor output, grammar panel on `[GRAMMAR_QUERY]`. Text input as a fallback mode, not the primary interface.
- [ ] Installable to iOS home screen; reachable over Tailscale.
- [ ] Pitch-accent visualisation is a Phase 3 slot — leave the panel space, ship nothing in it.
- **Acceptance:** the user completes a 10-turn *spoken* exchange from an iPhone.

### T2.8 — Conversational feedback states *(load-bearing, not polish)*
- [ ] Transcript appears as ASR resolves; a visible listening state; a visible thinking state.
- **Why this is not polish:** the restructured G1 makes "no dead air beyond ~500 ms without visible or audible feedback" a first-class criterion. At a measured 2.5 s voice-to-first-audio, feedback is what decides whether the app feels broken or merely deliberate. A silent 2.5 s gap reads as a crash; a 2.5 s gap with a live transcript reads as listening.
- **Acceptance:** no state in a real turn leaves the user without feedback for more than 500 ms.

### T2.9 — NFR-6 thermal soak
- [ ] 30-minute sustained session; assert p50 latency degrades by less than 20%.
- **Why here and not Phase 0:** a sustained-session test only means anything once there is a session to sustain. Every Phase 0 figure came from short bursts.
- **Acceptance:** a written result in `benchmarks/thermals.md`.

> **The single biggest failure mode:** breaking the streaming chain with a buffering `FrameProcessor` or a non-streaming HTTP TTS call. Every service downstream of the LLM must consume `TextFrame` as it arrives. Audit this before optimising anything else.

- **Gate:** p50 voice-to-first-audio < 1200 ms over 50 real turns.

---

## Phase 3 — Pronunciation *(deferred to post-October 2026 — deliberate trade, see PRD §10)*

### T3.0 — Evaluate `onsei` as prior art *(ahead of all other Phase 3 work)*
- [ ] Clone `itsupera/onsei` (MIT). Read `onsei/*.py` and the notebook.
- [ ] Report: (a) what its phoneme segmentation depends on and whether that builds on Apple Silicon — documented as Ubuntu-tested with several compile-from-source dependencies, so Docker may be required; (b) whether its FastAPI app can be wrapped as-is with the reference swapped to VOICEVOX output; (c) what we would have to write ourselves.
- [ ] **Evaluate explicitly: collapse the two segmentation mechanisms.** As first drafted, §6.1 would run two — wav2vec2 CTC for GOP, and onsei's Julius segmentation-kit for DTW boundaries. Julius is Perl + C + HTK models and is very likely the Apple Silicon problem this task uncovers. Frame-level CTC posteriors already yield phoneme boundaries, so **one model can serve both T3.2 and T3.4 and Julius disappears.** Option to assess: reuse onsei's DTW and scoring logic (MIT) with CTC posteriors substituted for its Julius segmentation.
- [ ] **Do not read that as "drop segmentation."** onsei's author found intensity-based DTW gave mixed results and moved to phoneme segmentation *deliberately*. Segmentation stays; only its source changes.
- [ ] **Contingent on T3.2a.** If T3.2a finds no clean phone-set mapping, CTC posteriors may not be available in a usable inventory and Julius returns to the table. Sequence T3.2a's report before committing to the collapse.
- [ ] **Report before proposing an implementation.** Do not write a scorer first.
- [ ] If any of it is reused: add the MIT notice to `THIRD_PARTY`.
- **Acceptance:** a written prior-art report covering the collapse option. No scorer code.

### T3.1 — `pyopenjtalk` accent type
- [ ] Accent type (0/1/2+) for display, and for sanity-checking the reference recording. **Not the grading target** — see ARCHITECTURE §6.1.

### T3.2a — Phone-set reconciliation *(blocks T3.2; go/no-go for the whole GOP approach)*
- [ ] `wav2vec2-xlsr-53-espeak-cv-ft` emits **espeak IPA** (387 labels). GOP needs the reference phoneme sequence in that *same* inventory. `pyopenjtalk` emits correct Japanese phones in a **different, non-IPA** set. Going via espeak's own Japanese G2P instead means espeak must resolve kanji readings, which it does badly.
- [ ] Determine whether `pyopenjtalk`'s phone set maps to this model's inventory with acceptable loss, **or** whether a Japanese-specific phoneme CTC model is required instead.
- [ ] Quantify "acceptable loss": count phones with no target, phones collapsing many-to-one, and whether any collapse merges a pair the learner must distinguish.
- [ ] **Report before implementing.** If no clean mapping exists, alignment-free GOP needs **replacing, not patching** — do not ship a lossy mapping to preserve the plan.
- [ ] Deps for this task: `espeak-ng`, `phonemizer`. **Pin `transformers`** — a `Wav2Vec2Processor` TypeError is reported on 4.46, and the checkpoint is from Dec 2021 (untouched ~4.5 years).
- **Acceptance:** a written mapping report with a go/no-go on GOP. Recorded as ARCHITECTURE §9 risk 9.

### T3.2 — Alignment-free GOP → segmental score *(contingent on T3.2a)*
### T3.3 — F0 extraction via `pyworld`
### T3.4 — Comparative accent scorer *(contingent on T3.2a)*
- [ ] DTW-align student to the VOICEVOX reference, apply the alignment to both pitch signals, normalize, compute mean distance.
- [ ] **Boundaries come from T3.2's CTC posteriors — not Julius.**

### T3.5 — Three-score scorer; accent cap wired but **disabled by default** (PRD FR-8)

### T3.6 — Reference validation *(gates enabling the accent cap)*
- [ ] Collect ~10 native recordings of the T0.2 sentences.
- [ ] Check that scoring against native and against VOICEVOX **ranks the same attempts in the same order.** Rank correlation, not absolute agreement.
- **Acceptance:** a written result. The FR-8 accent cap stays disabled until this passes.

### T3.7 — Pitch contour visualisation in the PWA
- [ ] Trend over time, never an absolute grade (PRD FR-6).

> **Validate early:** alignment-free GOP is published on English and child-speech corpora (speechocean762, CMU Kids). Japanese transfer is untested. Run T3.2 against 10 deliberately mispronounced utterances and inspect before building T3.5 on top of it.

---

## Phase 4 — Polish *(deferred)*

### T4.1 — Nightly batch Gemma 4 audio analysis (qualitative coaching notes only, never a score)
### T4.2 — Progress views, session history
### T4.3 — `launchd` daemonisation of the model server
