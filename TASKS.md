# Ocha — Task List

Sequenced. Each task has context, acceptance criteria, and a done checkbox.
**Do not start a phase until the previous phase's gate is passed.**

Reference documents in this repo:
- `PRD.md` — what and why
- `ARCHITECTURE.md` — how (component choices, latency budget, memory budget)

## Correctness remediation — 1 August 2026

- [x] Package curated grammar data; require model weights to be local before startup.
- [x] Quarantine complete LLM output and share one firewall/finalization path.
- [x] Remove mutable prompt caching; pass at most four explicit role-tagged exchanges; keep every MLX call on one inference worker.
- [x] Turn suspicious ASR into a visible event plus pre-synthesized repair audio.
- [x] Persist free-conversation observations without creating FSRS ratings.
- [x] Attribute all WebSocket events/audio with OCH1 exchange UUIDs and sequences; add known-answer v2 measurement tests.
- [ ] Re-record ASR separately through iPhone microphone and AirPods/HFP.
- [ ] Run the unfiltered 50-turn iPhone v2 gate. **G1a and G1b remain unproven.**

## Beginner-first learning — 1 August 2026

- [x] Make Guided Lessons the first-run default and keep Conversation selectable.
- [x] Add two curated listen/repeat/recall modules with exact accepted transcripts.
- [x] Keep guided progress separate from FSRS and pronunciation scoring.
- [x] Add local-only English directions, ordered Japanese playback, and microphone suppression while Ocha speaks.
- [x] Show validated Japanese, local romaji, and complete English meaning in Conversation mode.
- [ ] Complete the live iPhone walkthrough of both modes.

---

## Phase 0 — Bake-off *(gate: no application code until complete)*

> **Why this phase exists.** Every model choice in `ARCHITECTURE.md` is a hypothesis based on published benchmarks, not measurements on this machine with this user's voice. Building on unverified assumptions is the single most expensive mistake available here. Phase 0 costs one evening and can invalidate half the architecture.

### T0.1 — Hardware identification ✅
- [x] Record chip variant (`sysctl -n machdep.cpu.brand_string`), core counts, and measured memory bandwidth.
- **Why it matters:** base M4 (~120 GB/s) vs M4 Pro (~273 GB/s) is a >2x gap that decides whether a dense model is viable at all.
- **Acceptance:** `benchmarks/hardware.md` states the variant and measured bandwidth. ✅ **Base M4, 103.2 GB/s measured.**

### T0.2 — Voice sample corpus ✅
- [x] Recording kit delivered: `record.py` (device pinned, level-guarded, resumable), `transcripts.json` (20 entries, ground truth pre-filled), `CHECKLIST.md`. **Awaiting the user's recording session.**
- [x] 10 pure Japanese, 10 deliberately code-switched Japanese/English — sentences authored, covering ん, geminate っ, long ー, katakana, counters, and fillers えーと/あの/えっと.
- [x] Ground truth pre-filled (sentences authored before recording, so truth is known in advance). `record.py` prompts per take and writes back any deviation.
- **Acceptance:** `benchmarks/corpus/` with 20 WAVs and `transcripts.json`.

### T0.3 — ASR bake-off ✅
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

### T0.4 — LLM bake-off ✅
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

### T0.6 — Phase 0 report ✅
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
- **Acceptance:** migrations run clean and are idempotent; seed inserts 50 starter vocabulary items. ✅

### T1.3 — FSRS integration ✅
- [x] Wrapped `py-fsrs`. `due_items(limit)`, `known_items(min_reps)`, `lowest_stability(limit)`, `record_review(item_id, rating)`, plus `retrievability(item_id)`.
- [x] PRD FR-8 derivation table implemented. The accent cap is **PROVISIONAL and disabled by default** per the amended FR-8 — code path and tests exist, only the score is absent, so T3.6 enabling it is a flag flip.
- **Acceptance:** every explicit scheduler rating path is unit-tested; 30-day simulation gives 0 → 2 → 13 → 55 → 179 → 535 → 1436 → 3407 day intervals. ✅

### T1.4 — Grammar reference ✅
- [x] `src/ocha/resources/grammar.json` packaged with the wheel — 20 entries covering the T0.5 probe topics.
- [x] `hindi_contrast` on 15 of 20, only where the analogy holds. `interference_warning: true` on 9, including all three named: ने/が (`particle_wa_ga`), gender agreement (`particle_no`), stress-vs-pitch (`pitch_accent_basics`).
- [x] Pydantic loader, fail-fast, reports **every** malformed entry rather than just the first. `extra="forbid"` so a typo'd key is an error, not a silent drop.
- **Acceptance:** 20 valid entries load; malformed entries rejected. ✅

### T1.5 — Grammar firewall ✅ *(critical path)*
- [x] Detect the contiguous `GRAMMAR_QUERY` token anywhere in the completed model output, including damaged brackets. Presence suppresses the entire output; arbitrary marker-free explanations are not claimed detectable.
- [x] On detection, suppress the model's response entirely and serve from `grammar.json`. Verbatim — a test asserts every field is byte-identical to the curated entry.
- [x] On reference miss: return "not yet documented", log to `unauthored_grammar`. **Never** falls back to generation. Entry resolution is a deterministic trigger table — using the LLM to resolve would put it back in the correctness path.
- **Acceptance:** a test asserts that when the sentinel fires, no model-generated text reaches text or audio. This test must never be weakened. ✅

### T1.6 — Context Builder ✅
- [x] Assembles the system prompt from `known_items`, `due_items`, `lowest_stability`; history is carried on `TurnContext` for T1.8.
- [x] Reply length enforced by prompt *and* `MAX_REPLY_TOKENS`.
- [x] **REGISTER line included** — T0.5 measured it worth 9/15 → 14/15.
- [x] **§7.1's template corrected, not copied.** As written it was self-contradictory against real FSRS state: "use only words from KNOWN" while `TARGET` listed due items, which are typically *not* known. `TARGET` is now split into PRACTISE (known, weak) and INTRODUCE (not known, at most one per reply), which encodes FR-3 explicitly.
- [x] Context cap 2048, not 8k — T0.4 measured TTFT 32.6 s and decode 14.3 tok/s at 8k.
- **Acceptance:** snapshot tests over three FSRS states (cold start, some known, some weak). ✅

### T1.7 — LLM service ✅
- [x] MLX-backed local inference, `mlx-community/gemma-4-26b-a4b-it-4bit` per `benchmarks/DECISION.md`, as a **config value** (`OCHA_LLM_MODEL`).
- [x] Loaded once in the FastAPI lifespan, kept warm. Streaming interface present for Phase 2's sentence chunker.
- [x] **`enable_thinking=False`.**
- [x] Mutable prompt caches removed. History is explicit, role-tagged, limited to four complete exchanges and pruned to a 2,048-token rendered prompt.
- [x] Fails loudly if local model weights are unavailable; `generate()` before `load()` raises rather than lazy-loading or downloading.
- **Acceptance:** `/health` reports model loaded and resident memory, and real-model slow tests cover owner-thread status/generation and consecutive history. ✅

### T1.8 — Turn orchestration ✅
- [x] `POST /turn` — accepts text, returns tutor reply (or firewalled grammar answer), targets and observations; compatibility `ratings`/`usage` are empty.
- [x] Usage detection via `fugashi` lemmas, not substring matching — 食べる appears as 食べます/食べました/食べて. Handles unidic's orthography drift (ご飯→御飯), loanword lemma suffixes (コーヒー-coffee), and multi-token items (お茶 = 接頭辞+名詞).
- [x] **FR-8's "avoided" narrowed to *elicited* items.** Read literally it would rate five items `Again` every turn — a 1–2 sentence reply cannot exercise six targets. Avoidance now requires the tutor to have put the item in play.
- [x] Grammar-query turns are not scored — asking a question is not a production attempt.
- **Acceptance:** 10-turn free conversation leaves FSRS state unchanged while observations persist; HTTP-level firewall tests pass. ✅

### T1.9 — *(deleted)* Minimal PWA
Cut. Phase 1 exposes `POST /turn`; the voice-first PWA is built once in T2.7. Native iOS remains deferred.

### T1.10 — No-network test ✅
- [x] `src/ocha/net_guard.py` intercepts `socket.connect`/`connect_ex` and refuses anything outside loopback and the Tailscale CGNAT range (100.64.0.0/10).
- [x] The guard is itself proven: a deliberate outbound connection raises `OutboundNetworkError`, and a test asserts the patch is removed afterwards so it cannot silently disable later tests.
- **Acceptance:** a full `/turn`, a `/health`, and a firewalled grammar turn all complete with zero outbound connections. ✅

> **GATE.** T1.8's 10-turn integration test passes and T1.10 (no-network) passes. Proceed directly to Phase 2 — this is not a go/no-go, and Phase 1 is not a stopping point. See PRD §10.

---

## Phase 2 — Voice loop *(in scope)*

### T2.1 — Pipecat scaffold, pinned version, WebSocket transport
- [x] **Pre-flight audio test done — BOTH GATES PASS.** (b) Output stayed on the AirPods with the mic live; iOS did *not* force the built-in speaker. (a) `getUserMedia` works in standalone home-screen mode. Continuous listening is viable as designed. `benchmarks/ios-audio.md`.
- [x] ~~`SmallWebRTCTransport` had a choppy-audio regression around v0.0.62 — pin a version that sounds correct.~~ **DOUBLY MOOT.** Pipecat is now **1.6.0** (that warning describes the 0.0.x series), and the transport is no longer WebRTC — see ARCHITECTURE §2.3.
- [ ] **Pipecat 1.0 restructured the API.** Services take submodule imports (`pipecat.services.X` → `pipecat.services.X.llm`); transports moved out of `services/`; provider-specific contexts are replaced by a universal `LLMContext`; `on_client_close` → `on_client_disconnected`. Most relevant here: **VAD config moved off transport params onto `LLMUserAggregatorParams.vad_analyzer`**, which changes T2.2.
- [x] ~~**`pipecat-ai[mlx-whisper]` exists** — the ASR integration is likely a config, not a custom service.~~ **Not usable as-is.** `pipecat.services.whisper.stt` imports `faster_whisper` at module scope even for the MLX class, so using it means installing CTranslate2 to run a model we already call directly. Wrote `speech/asr.py` against `mlx_whisper` instead — which also keeps transcription on the event-loop thread, as constraint 6 requires.
- [ ] **`PipelineTask`/`PipelineRunner` are deprecated** (Pipecat 1.3, removed at 2.0) in favour of `PipelineWorker`/`WorkerRunner`. Using the new pair. Note `WorkerRunner.add_workers` is a **coroutine** despite the name — calling it unawaited is a silent no-op and the runner then blocks forever with no workers, which presents as a transport hang.
- [x] **VOICEVOX has no Pipecat service** — wrote `speech/tts.py` against its local HTTP API. `outputSamplingRate` is set on the query, so VOICEVOX renders at 16 kHz directly and no resampler sits in the latency path.
- [x] **Instrumentation built FIRST, before any real `FrameProcessor`.** `src/ocha/speech/probe.py` — `TurnStateProbe` is a pass-through tap that maps frames to `TurnState` and times the §5.1 stages. A test asserts it never buffers or reorders, which is §5.2's named worst failure.
- [x] **Pipecat 1.6.0 installed** with `[webrtc,silero]`. API confirmed against the running library rather than the migration guide.
- [x] **`FastAPIWebsocketTransport` mounted on the existing FastAPI app**, `/ws`, 16 kHz mono PCM in and out. The PWA and server own both ends, so `speech/wire.py` uses a small OCH1 binary audio header plus JSON control messages instead of imposing protobuf and generated client bindings.
- [x] Pipeline assembled: `transport.input() → VAD → ASR → context/LLM → chunker → TTS → transport.output()`, with `TurnStateProbe` tapped once, immediately before `transport.output()`. **Loopback first, not stubs:** the pipeline today is `input -> loopback -> probe -> output`, which is the day-one audio test rather than scaffolding — it answers whether HFP audio survives the round trip, by ear, before ASR exists to blame.
- [ ] Wired to a live connection (needs Tailscale for the phone).

### T2.2 — silero-vad endpointing and barge-in
- [x] `VADProcessor(vad_analyzer=SileroVADAnalyzer(...))`, a standalone processor. **Correction to the note in T2.1:** `vad_analyzer` does live on `LLMUserAggregatorParams`, but that path requires adopting Pipecat's LLM context aggregators, which this pipeline does not use (`speech/tutor_stage.py`). `VADProcessor` is the wiring for a pipeline that owns its own context.
- [x] **`VADUserStartedSpeakingFrame` is NOT a subclass of `UserStartedSpeakingFrame`** — both are bare SystemFrames. `VADProcessor` emits only the VAD pair, so the probe had to map them; without that, G1a reported an empty timeline and passed.
- [x] **`stop_secs` MEASURED and it was wrong.** 0.2 s endpointed mid-utterance on 6 of 8 recordings. The push-to-talk client now uses 1.0 s silence or an explicit Stop tap; this sits inside voice-to-first-audio and must be remeasured.
- [ ] **Wire up Pipecat smart-turn — scoped, not a config change.** `LocalSmartTurnAnalyzerV3` exists and is genuinely local (ONNX, offline, one-time model download). But it attaches through `TurnAnalyzerUserTurnStopStrategy` → `LLMUserAggregatorParams.user_turn_strategies`, i.e. hanging off Pipecat's LLM aggregator — **the same component this pipeline deliberately does not use**, because Phase 1 owns context, history and the firewall (`speech/tutor_stage.py`). Two options: adopt the aggregator (two implementations of the turn — rejected once already) or drive the strategy directly. The latter looks viable: it exposes `process_frame`, `handle_user_turn_started/stopped` and a `trigger_user_turn_stopped` event, so a ~50-line adapter can feed it frames and act on its verdict.
  - **Validate on the target learner before building the adapter** (constraint 9). Current releases support Japanese, but viability on Hindi-L1 absolute-beginner speech, mid-sentence pauses and code-switching is unproven. Run the analyzer over that corpus first.
- [ ] Barge-in on-device: the client stops playback on `interrupt`; untested without the app.
### T2.3 — ASR service wrapping the Phase 0 choice
- [x] `OchaWhisper(SegmentedSTTService)` with `wants_wav_segments = False`; `no_repeat_ngram_size=4` carried over from T0.3, where its absence produced 557 insertions on one clip. Warmed at lifespan on the event-loop thread, next to the LLM.
- [ ] **Re-measure CER through the production audio path — and through BOTH capture devices.** The T2.1 pre-flight found the input device is not stable: two runs on the same phone with the same headset connected gave `AirPods` and `iPhone Microphone` respectively. The corpus characterises the MacBook's built-in mic, which is neither of them. The T0.2 corpus was recorded on the MacBook's built-in mic, so `2.56` describes an input path the product will never see. Re-record part of the corpus through AirPods → iPhone → transport → ASR and re-score with the same pre-registered rules. See `benchmarks/ios-audio.md`.
### T2.4 — Accuracy-first LLM quarantine and sentence TTS
- [x] `TutorStage` collects the complete generation on the inference worker. It emits nothing model-derived until shared `finalize_turn` has firewalled and persisted the result.
- [x] After finalization, safe replies are split on 。！？ for sentence-sized synthesis. A late or corrupted sentinel produces zero model text and zero audio.
- [x] Phase 1 is reused rather than reimplemented: HTTP and voice share finalization, observation and persistence.
### T2.5 — VOICEVOX service, synthesis per sentence
- [x] `VoicevoxTTS`, speaker 13, 16 kHz. Verified against the live engine: 0.48 s for 「そうですね。」, consistent with T0.7's 0.63 s.
- [x] **A stub-only test suite was green while the service was broken.** `self.sample_rate` is 0 until the pipeline's StartFrame reaches `start()`, so a standalone call asked for 0 Hz and got an HTTP 500. `tests/test_voicevox_live.py` (slow) now covers the real contract.
### T2.6 — End-to-end latency instrumentation
- [x] The probe measures the assembled pipeline, not just itself. Two defects found by asserting that: `TTSService` consumes `LLMTextFrame`, so `llm_ttft_s` was permanently None, and `first_audio` watched for frames the pipeline does not guarantee, so `voice_to_first_audio_s` was too. **A probe that reports None is indistinguishable from a pipeline that never ran.**
- [x] **Measured through the pipeline: 3.73 s p50 over four turns — G1b MISSED by ~0.5 s.** Not model speed: MLX runs on the event-loop thread (constraint 6), so during generation the loop cannot deliver frames already pushed, and VOICEVOX's audio waits. `benchmarks/voice-loop.md`.
- [x] **G1a fails on every turn.** Segmented whisper gives no interim transcripts, so `transcribing` sits unchanged for ~1.0 s. Not fixable by latency work; three options recorded, one needs a product decision. The test is not being loosened.
- [ ] **DECISION NEEDED — a dedicated single-threaded inference worker with a queue.** Constraint 6 names this as the only correct shape once concurrency is needed. It now is. Expected recovery ~0.6–0.9 s, which is what G1b needs.
- [x] **Historical 2.05 s result invalidated.** It mixed exchange provenance, admitted negative latencies, post-selected clean turns and calculated p95 incorrectly. Preserved in `benchmarks/voice-loop.json`.
- [ ] **G1a/G1b unproven.** Run exactly 50 attributable iPhone turns with `voice_loop_v2.py`; report every ASR rejection and instrument failure.
- [x] **Whisper repetition loops are rejected visibly.** Post-hoc detection emits `asr_rejected`, displays the suspect transcript, plays the curated repair prompt and skips LLM/persistence/observations/FSRS. It does not silently consume the turn.
- [ ] Live v2 measurement from the phone: exactly 50 spoken turns, with no post-selection.

### T2.7 — Browser client, voice-first *(absorbs the deleted T1.9 and T2.8's UI)*
- [x] `web/index.html` — one file, no build step. `getUserMedia` + `AudioWorklet` → 16 kHz mono PCM over the WebSocket; playback queued against an audio cursor; barge-in drops the queue on `interrupt`. The worklet downsamples rather than trusting the 16 kHz constraint, because iOS decides capture parameters for itself.
- [x] Live transcript (user + tutor), grammar panel with `hindi_contrast` and the interference warning, turn-state indicator driven by the server's `state` messages.
- [x] `ClientText` is retained as the supported adapter from Pipecat transcript/reply frames to transport messages.
- [x] Browser UI and transport verified end to end; the automated test browser blocked microphone capture, so the real iPhone capture path remains a physical validation item.
- [ ] Furigana on tutor output. Text input as a fallback mode.
- [ ] **Developed against the Mac's own browser first.** No audio-routing quirk, no provisioning, no Tailscale hop — none of them debugged at the same time as the pipeline.
- [ ] Then the same client from the iPhone as an installed PWA over Tailscale. **Unblocked:** both pre-flight gates passed (`benchmarks/ios-audio.md`).
- **Not doing before October:** the native SwiftUI app. See T4.5 and ARCHITECTURE §2.3.
- [ ] Pitch-accent visualisation is a Phase 3 slot — leave the panel space, ship nothing in it.
- **Acceptance:** the user completes a 10-turn *spoken* exchange from an iPhone.

### T2.8 — Conversational feedback states *(load-bearing, not polish)*
- [x] **Foundation done ahead of order.** `src/ocha/turnstate.py`: the `TurnState` enum, a `TurnTimeline` recorder, and `satisfies_g1a()`. Wired into `run_turn`, so Phase 1 already produces a timeline.
- [x] **G1a is proven violable today**, before any voice component exists: at T0.9's measured 0.75 s LLM stage, the single `THINKING` state runs 1.01 s with no feedback. A test asserts this and fails if the instrument stops detecting it.
- [ ] Transcript appears as ASR resolves; a visible listening state; a visible thinking state.
- **Why this is not polish:** the restructured G1 makes "no dead air beyond ~500 ms without visible or audible feedback" a first-class criterion. At a measured 2.5 s voice-to-first-audio, feedback is what decides whether the app feels broken or merely deliberate. A silent 2.5 s gap reads as a crash; a 2.5 s gap with a live transcript reads as listening.
- **Acceptance:** no state in a real turn leaves the user without feedback for more than 500 ms.

### T2.10 — *(deleted)* Weekly re-deploy workflow
Cut with the native client. Nothing to re-sign — a PWA installs from a URL. See T4.5.

### T2.9 — NFR-6 thermal soak
- [ ] 30-minute sustained session; assert p50 latency degrades by less than 20%.
- **Why here and not Phase 0:** a sustained-session test only means anything once there is a session to sustain. Every Phase 0 figure came from short bursts.
- **Acceptance:** a written result in `benchmarks/thermals.md`.

> **The single biggest failure mode:** emitting any model-derived text or audio before the complete response passes the shared firewall. Full-output quarantine is mandatory even when it misses the latency gate.

- **Gate:** G1a holds over 50 real spoken turns from the phone (no state without feedback for >500 ms), and p50 voice-to-first-audio is within G1b's 3.2 s bound. ~~< 1200 ms~~ — retired with G1, see PRD §7a.

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

### T4.5 — Native iOS app *(post-October, deliberately)*
Deferred from Phase 2 on 31 July 2026, after being started and stopped the same day. The reasons it is worth doing eventually: `AVAudioSession` gives explicit category/route control where WebKit only infers it (the pre-flight found the capture device varies between runs), background audio when the screen locks, and independence from WebKit media policy. ARCHITECTURE §2.3.

The reason it is not Phase 2: a Swift client is a second codebase — `AVAudioEngine`, transport, transcript UI, furigana, barge-in — realistically 3–4 weekends, started before Phase 2 was finished, nine weeks from the deadline. PRD §10's scope breach exactly.

**Free-account constraints to plan around** (they shape the design, so record them now):
- Provisioning profiles expire after **7 days**, silently. A daily-habit SRS tool that dies weekly is worse than no app.
- **10 App IDs per 7-day period**, so bundle identifiers cannot be churned during development.
- **3 devices** per account.
- **Use a separate Apple ID from the outset.** A free personal team cannot be upgraded cleanly, and an account that has been used for 7-day provisioning drags that history along — a later paid upgrade should not be stuck behind it.

The wire format is deliberately trivial (JSON control messages + raw PCM, `speech/wire.py`) so a Swift client is a client, not a port.

### T3.7 — Pitch contour visualisation in the client
- [ ] Trend over time, never an absolute grade (PRD FR-6).

> **Validate early:** alignment-free GOP is published on English and child-speech corpora (speechocean762, CMU Kids). Japanese transfer is untested. Run T3.2 against 10 deliberately mispronounced utterances and inspect before building T3.5 on top of it.

---

## Phase 4 — Polish *(deferred)*

### T4.1 — Nightly batch Gemma 4 audio analysis (qualitative coaching notes only, never a score)
### T4.2 — Progress views, session history
### T4.3 — `launchd` daemonisation of the model server
