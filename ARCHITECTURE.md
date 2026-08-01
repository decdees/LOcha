# Ocha — Real-Time Japanese Tutor Loop
### Architecture v1.0 · July 2026 · $0/month, fully self-hosted

---

## 0. What changed since the last design pass

Three things invalidate parts of the earlier plan:

1. **Model generation turned over.** Qwen3-30B-A3B is a generation behind. As of mid-2026 the relevant open-weight families are **Gemma 4** (released 31 Mar 2026), **Qwen 3.5**, and **Qwen 3.6**. All Apache 2.0.
2. ~~**Memory bandwidth, not memory capacity, is the binding constraint** on an M4 laptop. This forces a Mixture-of-Experts model.~~ **FALSIFIED — see §3.0.** True for a model measured alone; false for the system. Once whisper, VOICEVOX and macOS are co-resident, **capacity binds first**, and the MoE's ~15 GB footprint is what breaks it.
3. **Gemma 4 accepts audio natively** (25 tokens/sec of audio, 30s clips, 16 kHz mono). This does *not* replace the ASR stage, but it enables a genuinely useful offline grader. See §6.3.

---

## 1. Design constraints

| Constraint | Value |
|---|---|
| Budget | $0/month recurring. One-time downloads only. |
| Hardware | MacBook Pro M4, 32 GB unified memory |
| Client | Browser (Mac) in Phase 2, iPhone PWA over Tailscale for shipping; native iOS deferred post-October — §2.3 |
| Target turn latency | ~~< 1.2s p50~~ **G1a: no dead air >500 ms without feedback; G1b: p50 ≤ 3.2 s** (PRD §7a) |
| Blocking schedule risk | Hard external deadline, October 2026 |
| Pedagogical anchor | Falou-style drilling + free conversation, FSRS-scheduled |

**The hard requirement that drives everything:** this is a *pronunciation-sensitive* language app. Correct pitch accent in the audio the learner hears, and reliable measurement of the audio the learner produces, matter more than conversational sparkle. That pushes several choices away from the obvious ones.

---

## 2. Component selection

| Layer | Choice | License | Footprint (4-bit / runtime) |
|---|---|---|---|
| Orchestration | **Pipecat** (BSD-2) | Free | negligible |
| Transport | ~~WebRTC (`SmallWebRTCTransport`)~~ **WebSocket** (`FastAPIWebsocketTransport`), 16 kHz mono PCM — reversed with the client, §2.3 | — | — |
| VAD / turn detection | **silero-vad**; Pipecat Smart Turn deferred | MIT | ~50 MB |
| ASR | ~~kotoba-whisper-v2.0~~ **whisper-large-v3 on MLX** (`mlx-community/whisper-large-v3-mlx`) — **T0.3 REVERSED this**, see §2.1. Wrapped by hand, **not** via `pipecat-ai[mlx-whisper]`: that module imports `faster_whisper` at module scope even for its MLX class, so using it means installing CTranslate2 to run a model we already call directly (T2.3). | Apache 2.0 | ~3 GB |
| LLM | ~~Gemma 4 26B-A4B (MoE)~~ **Qwen3.5-9B, 4-bit** (`mlx-community/Qwen3.5-9B-4bit`) — reversed on correctness (DECISION.md) and again on capacity (§3.0) | Apache 2.0 | ~5 GB |
| TTS | **VOICEVOX Engine**, speaker **id 13 (青山龍星)** — a §6.1 dependency, not a preference | Free (per-character terms) | ~0.3 GB |
| G2P / accent truth | **pyopenjtalk** | Modified BSD | small |
| Segmental scoring | **wav2vec2-xlsr-53-espeak-cv-ft** — alignment-free GOP, 387 phonetic labels | Apache 2.0 | ~1.3 GB fp32 / ~0.7 GB fp16 |
| Pitch extraction | **pyworld** / **parselmouth** | — | small |
| Scheduler | **py-fsrs** | MIT | — |
| Store | **SQLite** (WAL) | Public domain | — |
| API | **FastAPI** + Uvicorn | MIT | — |

### 2.1 ~~Why kotoba-whisper-v2.0~~ — REVERSED BY T0.3

> **The original argument, kept for the record:** *"kotoba-whisper is a Japanese-specialised distillation of Whisper large-v3 — 756M params, roughly 6.3x faster than large-v3, with ~8.4% CER on JSUT and ~11.6% on ReazonSpeech. A generalist audio LLM has to beat a purpose-built distillation on the hardest possible input: accented beginner Japanese. Assume it doesn't until measured."*

**Measured. Both claims fail.** See `benchmarks/asr.md`.

1. **Accuracy.** On this user's voice, whisper-large-v3 scores **2.56% CER on pure Japanese against kotoba-whisper-v2.0's 17.09%** — 6.7x better — and wins the code-switched subset by 19 points. Under the pre-registered Stage 1 gate, both kotoba variants were **disqualified** (5/10 and 4/10 code-switched utterances unusable as LLM input); large-v3 passed at 2/10.

2. **Speed.** The 6.3x figure does not survive contact with the runtime. large-v3 on MLX runs **p50 1.25 s** against kotoba's **1.10 s** on transformers — a 14% difference. The runtime dominates the model choice.

**Why the reasoning inverted:** the distillation is trained on native Japanese. The actual input is a Hindi-L1 absolute beginner speaking accented Japanese and code-switching mid-sentence. That is *out of domain* for a narrow distillation and *in domain* for a multilingual generalist. §2.1 had the direction of the domain gap backwards.

**Smart Turn status.** Current Pipecat releases support Japanese, so language support is not the blocker. Adoption remains deferred until it is validated on a Hindi-L1 absolute-beginner corpus with mid-sentence pauses and code-switching. A support claim is not evidence on this learner distribution.

kotoba-whisper-v2.2 is moot — v2.0 was already beaten.

**Still untested:** the claim about a generalist *audio LLM* (Gemma 4 E4B). This bake-off measured a generalist *ASR*. Open assumption, recorded in DECISION.md.

**Decoding config is load-bearing.** `no_repeat_ngram_size=4` is required: without it large-v3 degenerates on some utterances into hundreds of repetitions (557 insertions, 70 s for one 4.9 s clip).

### 2.2 Why VOICEVOX and not Kokoro / Style-Bert-VITS2

Kokoro-82M is smaller and faster. AivisSpeech (Style-Bert-VITS2 based) is more expressive. Neither is the right call here.

VOICEVOX is built on OpenJTalk, which means its pitch accent comes from an explicit accent dictionary you can inspect and override. For a learner, the TTS output *is* the pronunciation model being internalised — an expressive neural TTS that occasionally flubs 箸 vs 橋 is actively teaching the wrong thing. Deterministic, correctable accent beats naturalness here.

Keep **Kokoro-82M as a fallback** for non-pedagogical speech (UI prompts, encouragement lines) where latency matters more than accent precision.

### 2.3 Client: browser now, native iOS post-October — 31 July 2026

A native SwiftUI client was chosen and then **un-chosen the same day**, before any Swift was written. The argument for it was real: `AVAudioSession` gives explicit category and route control where WebKit only infers them (the pre-flight found the capture device *varies between runs* on the same hardware), plus background audio when the screen locks, plus independence from WebKit media policy — the gates pass on iOS 18.7, which is not a contract for iOS 19.

**The argument against it is scheduling, and it wins.** A Swift client is a second codebase — `AVAudioEngine`, transport, transcript UI, furigana, barge-in — realistically 3–4 weekends, and it was about to begin *before Phase 2 was finished*, nine weeks from a hard deadline. Free provisioning expires every 7 days with no warning, which is corrosive for a tool whose whole value is daily use. Deferred to Phase 4 (TASKS T4.5).

Phase 2 therefore develops against a **browser client on the Mac**: no audio-routing quirk, no provisioning, no Tailscale hop, and none of those debugged at the same time as the pipeline. The **iPhone PWA is the shipping target and is unblocked** — both pre-flight gates passed (`benchmarks/ios-audio.md`), so the same client over Tailscale is the delivery path.

**The phone's audio constraint is not a client choice.** iOS cannot pair the built-in microphone with A2DP output; `setPreferredInput(builtInMic)` forces output to the speaker (Apple QA1799, still current). A headset means HFP duplex in both directions and narrowband audio into the ASR. Unmeasured — T2.3 measures it, and the fallback is phone mic + phone speaker. Going native would not have changed this.

**The transport choice survives the reversal.** WebSocket rather than WebRTC, even with a browser client. `getUserMedia` plus an `AudioWorklet` hands the page PCM directly, and Tailscale (WireGuard) already solves NAT traversal a layer below. What is left of WebRTC's value is jitter buffering and echo cancellation, neither of which pays for SDP negotiation, ICE, and an Opus round-trip on a LAN link. The client streams **16 kHz mono PCM over a WebSocket** into Pipecat's `FastAPIWebsocketTransport`, mounted on the FastAPI app that already serves `POST /turn`. One process, one port, one Tailscale hostname. This is also what makes a later native client cheap: the wire format is 40 lines of JSON-and-PCM, not an SDP negotiation.

`ClientText` is a supported transport adapter: Pipecat's output transport serializes audio and transport-message frames, not transcript/LLM text frames. Converting those frames immediately before output is the intended boundary, not an accidental workaround.

All model weights must already exist in the local Hugging Face cache. Setup may download them explicitly; application startup resolves with `local_files_only=True` and fails rather than downloading or choosing a fallback.

Cost of the choice: no automatic packet-loss concealment, and echo cancellation remains the PWA's capture responsibility. Both are acceptable on a link that is a WireGuard tunnel over LAN; the native Phase 4 client can later use `AVAudioSession` `.voiceChat` mode.

---

## 3. The bandwidth problem — SUPERSEDED. Read §3.0 first.

### 3.0 What this section got wrong

**This section's headline argument is falsified.** It reasons that memory *bandwidth* is the binding constraint, concludes that a Mixture-of-Experts model is therefore mandatory, and recommends Gemma 4 26B-A4B. The reasoning is sound for a model measured **alone**. It does not survive the model being measured **alongside everything else that has to run**.

Measured in T0.9, both models co-resident with whisper-large-v3, VOICEVOX and macOS:

| model | weights | MLX peak | swapouts in an 8-turn burst |
|---|---|---|---|
| Gemma 4 26B-A4B (the MoE this section recommends) | 14.2 GB | 17.95 GB | **~340,000** |
| Qwen3.5-9B (dense, the model this section ranks second) | 5.0 GB | 8.72 GB | **0** |

**Once the full loop is resident, capacity binds before bandwidth does, and the MoE's ~15 GB footprint is what breaks it.** §3 optimised the one constraint it measured and pushed the system into the one it did not.

**The MoE recommendation was wrong for this machine.** Not because the bandwidth arithmetic below is incorrect — it is correct, and T0.1 confirmed its figures — but because it was derived from a single-component measurement and presented as a system-level conclusion.

**What remains unmeasured, stated so this correction is not itself overclaimed:** Gemma was still *faster* end to end (2.49 s vs 2.53–3.03 s) despite the swapping. The latency cost of that swap traffic has not been demonstrated — it would show up in a sustained session, which is T2.9's 30-minute soak. So the honest claim is that the *reasoning* was wrong and the risk is real, not that the swapping has already been shown to hurt. If T2.9 shows no degradation, the MoE becomes a live option again on a machine with more headroom.

The LLM choice was separately reversed to Qwen3.5-9B on **correctness** grounds (see `benchmarks/DECISION.md`); the capacity finding is a second, independent reason the same way.

### 3.1 The bandwidth arithmetic, which is still correct

LLM token generation is memory-bandwidth bound, not compute bound. On Apple Silicon this decides everything **about a model in isolation**.

| Chip | Approx. bandwidth |
|---|---|
| M4 (base) | ~120 GB/s spec — **measured 103.2 GB/s on this machine (86%)**, see `benchmarks/hardware.md` |
| M4 Pro | ~273 GB/s |
| M4 Max | ~410–546 GB/s |

> **T0.1 RESOLVED.** This machine is a **base M4** — 10 CPU (4E+6P), 10-core GPU, 32 GB. Measured GPU read bandwidth **103.2 GB/s ±0.4%**. The low-bandwidth branch below is the live one.

**A 27B dense model at 4-bit reads ~15 GB per token.** At the **measured** 103.2 GB/s that is a ceiling of **6.9 tok/s**. A 40-token tutor reply is ~5.8 s of generation alone. Dead on arrival for voice.

**A 26B-A4B MoE reads only its ~3.8B active params** — roughly 2.2 GB per token at 4-bit. At the measured bandwidth that is a **46.9 tok/s ceiling**. That's the difference between a usable product and a demo.

> **T0.4 MEASURED.** The scattered-read effect is real and now quantified: the MoE achieves **84%** of measured peak bandwidth, the dense Qwen3.5-9B **99%** — a ~15% routing penalty. It does not change the conclusion. Measured **39.4 tok/s** for the MoE against 20.5 for the dense 9B. Note this **beats** the "25–35 realistic" estimate above, which was pessimistic.
>
> **But tok/s is not constant with context, which this section assumes throughout.** gemma falls from 37.7 tok/s at 256 tokens to **14.3 at 8k**. At 8k the MoE advantage is nearly gone. It wins decisively at the lengths this product uses (a 10-turn tutor conversation measured 560 tokens). See `benchmarks/llm.md`.

> **~~Action item~~ RESOLVED by T0.1.** Base M4, 10-core, 32 GB. Measured 103.2 GB/s. **MoE is mandatory**; dense 27B is not viable. §3.2 option 4 is struck, and T0.4 skips `qwen3.5:27b`.

### 3.2 Model shortlist, in preference order

1. **Gemma 4 26B-A4B** — best active-parameter efficiency available; 3.8B active. Default pick on base M4.
2. **Qwen 3.5 9B** — dense, small, and Qwen 3.5 has the strongest Japanese of the open families (201 languages, top-tier JA/KO/ZH). Fastest to first token. Weaker at grammar explanation.
3. **Qwen 3.6 35B-A3B** — 3B active, excellent speed, but ~19 GB at 4-bit leaves almost nothing for whisper + VOICEVOX + macOS. Only if you drop to a smaller ASR.
4. ~~**Qwen 3.5 27B dense** — M4 Pro only. Best Japanese quality of the four.~~ **STRUCK by T0.1** — this machine is a base M4 at a measured 103.2 GB/s, giving a 6.9 tok/s ceiling. Not viable.

> **Ordering superseded by §3.0.** This list ranks by throughput, which is the wrong axis once the whole loop is resident. Item 2 (Qwen 3.5 9B) is the shipped choice: it is slower and it is the only one that leaves the system enough headroom not to swap.

**Known gotcha:** Ollama had an `unknown model architecture: 'qwen35moe'` bug (filed Mar 2026) affecting Qwen 3.5 MoE variants with separate vision projectors. Not directly your path, but a signal — verify your exact artifact loads before building around it. LM Studio moved to an MLX backend and is a lower-friction alternative for Mac.

---

## 4. Memory budget (base M4, Gemma 4 26B-A4B)

```
Gemma 4 26B-A4B, 4-bit MLX          ~14.5 GB
  + KV cache @ 8k ctx                ~1.0 GB
kotoba-whisper-v2.0 (fp16)           ~1.5 GB
VOICEVOX engine + voice              ~1.0 GB
silero-vad                           ~0.1 GB
wav2vec2 GOP (on demand, batch only) ~0.7 GB fp16
FastAPI + Python runtime             ~0.8 GB
macOS + browser + editor             ~7.0 GB
─────────────────────────────────────────────
Total                                ~26.6 GB
Headroom                              ~5.4 GB
```

The GOP model replaces MFA (§9 risk 5) and is slightly larger. Not a problem: §6 runs scoring asynchronously outside the loop, so it loads for batch work and need not be co-resident with a live turn. Measure the loop-resident subset — whisper + LLM + VOICEVOX — in T0.7.

> **T0.4 MEASURED.** Weights **14.20 GB** — the ~14.5 GB estimate is accurate. KV cache at 8k is **1.94 GB, not ~1.0** (1.9x low), so model+KV at 8k is **16.14 GB**. That costs ~0.9 GB of the headroom below. Moot at the context cap now recommended (~2k → KV ≈ 0.8 GB).

Workable. Two rules:

- **Keep everything warm.** A cold MLX load is 15–25s and destroys the illusion of conversation. Run the model server as a `launchd` daemon, not on demand.
- **Cap context at ~2k, not 8k.** T0.4 revised this. KV growth is the smaller problem; *prefill and decode* are the larger ones. At 8k, TTFT is 32.6 s and decode drops to 14.3 tok/s — slower than the dense model. At 2k the MoE still runs ~36 tok/s. A 10-turn tutor conversation measured 560 tokens, so 2k is ample.
- **Do not retain mutable MLX prompt caches.** Every turn renders the system prompt, at most four explicit role-tagged exchanges, and the current user turn. Oldest complete exchanges are dropped until the prompt is at most 2,048 tokens.

---

## 5. The conversation loop

```
Browser (getUserMedia + AudioWorklet) — Mac in Phase 2, iPhone PWA to ship
   │ WebSocket / 16 kHz mono PCM, over Tailscale
   ▼
Pipecat pipeline ── FastAPIWebsocketTransport
   │
   ├─ TurnStateProbe ──────► timeline for G1a  (tap, passes frames through)
   │
   ├─ silero-vad ──────────► endpoint detected (~150ms after speech end)
   │
   ├─ whisper-large-v3 ────► transcript          (1250 ms, T0.3 measured)
   │
   ├─ Context Builder ─────► system prompt + FSRS steering + explicit history
   │
   ├─ Qwen3.5-9B ──────────► complete reply quarantined on inference worker
   │
   ├─ Sentence Chunker ────► splits on 。！？ boundaries
   │
   └─ VOICEVOX ────────────► first audio packet   (~200ms after first sentence)
        │
        ▼
   Browser playback ─► headset
```

### 5.1 Latency budget

| Stage | Target | Notes |
|---|---|---|
| VAD endpoint | **600 ms configured** | Accuracy-first: 200 ms cut off 6/8 recorded beginner utterances. Smart Turn remains deferred pending the target corpus. |
| ASR | ~~250 ms~~ **1250 ms** | **T0.3 MEASURED**, large-v3 on MLX. The 250 ms assumed kotoba, which T0.3 disqualified. Choosing kotoba to save 150 ms would cost 6.7x the character errors, and a wrong transcript costs a whole turn. |
| LLM stage | **UNPROVEN in v2** | Mutable prompt-cache reuse is removed. The complete output is quarantined before firewall finalization, so historical TTFT is not the user-visible metric. |
| First sentence generated | ~~250 ms~~ **400 ms** | **T0.4 MEASURED.** ~15 tokens at 37.7 tok/s. (The old row's arithmetic was inconsistent: 15 tokens at 30 tok/s is 500 ms, not 250.) |
| VOICEVOX synthesis | ~~200 ms~~ **690 ms standalone / 1170–1440 ms in the pipeline** | **T2.6 MEASURED.** The gap between the two is the blocked event loop, not VOICEVOX. |
| Network (Tailscale, LAN) | 30 ms | WAN adds 50–150 ms |
| **Voice-to-first-audio** | **UNPROVEN** | The historical 2.05 s result is invalid because attribution was inferred from arrival order and negative measurements were admitted. The v2 measure is final client-sent speech sample to scheduled first `tutor` audio for the same exchange. Accuracy-first full-output quarantine remains in place even if the latency gate is missed. |

### 5.2 The three rules that produce that number

1. **Quarantine the complete LLM response.** No model-derived text or audio is emitted until the shared firewall/finalization function has examined the entire reply. This correctness requirement intentionally overrides streaming latency.
2. **After finalization, chunk TTS by sentence.** Sentence-sized synthesis remains useful, but it begins only from the safe `TurnResult.reply`.
3. **Cap reply length in the system prompt.** One to two sentences. This is both more natural for a tutor and ~3x faster. Enforce it with `max_tokens` too — prompts get ignored.

~~**Use WebRTC, not WebSockets.**~~ **Reversed — see §2.3.** Behind Tailscale, WebRTC's NAT traversal is redundant and its Opus round-trip is pure cost because the PWA already captures PCM. The choppy-audio caveat referred to `SmallWebRTCTransport` in the 0.0.x series and is doubly moot — Pipecat is at 1.6.0 and that transport is no longer used.

The three rules above are transport-agnostic and unchanged.

---

## 6. Pronunciation assessment (the part no LLM does)

This runs **outside** the latency-critical path. The learner speaks, the conversation continues immediately, and scoring lands asynchronously in the review panel.

### 6.1 Deterministic pipeline

```
User audio (16 kHz mono)
   │
   ├─ Reference transcript (kotoba-whisper) ──► score free speech as if scripted
   │      ⚠ A bad transcript silently corrupts every downstream pronunciation
   │        score. This is the second reason FR-2's transcript display is
   │        load-bearing rather than cosmetic.
   │
   ├─ wav2vec2-xlsr-53 GOP(audio, transcript) ─► alignment-free segmental score
   │      No forced alignment, therefore no alignment error to contaminate it.
   │      ⚠ CONTINGENT ON T3.2a: this model emits espeak IPA; pyopenjtalk emits
   │        a different, non-IPA set. If no clean mapping exists, this stage is
   │        REPLACED, not patched. See §9 risk 9.
   │      └─► frame-level CTC posteriors ──┐  (same pass, reused below)
   │                                        │
   ├─ pyworld.harvest(audio) ──────────────►│  F0 contour
   │                                        │
   └─ Comparative scorer (onsei method) ────┘  student vs REFERENCE RECORDING
        1. crop + segment ── boundaries from the CTC posteriors above.
                             No Julius, no second segmentation mechanism.
        2. DTW-align student to reference on those phonemes
        3. apply that alignment to both pitch signals
        4. normalize, compute mean distance
        ├─ Accent:  mean DTW pitch distance vs reference
        └─ Rhythm:  mora timing regularity vs reference
```

Three numbers per utterance: **segmental**, **pitch accent**, **rhythm**. Deterministic, explainable, and — critically — *stable across sessions*, which an LLM's opinion is not. FSRS needs a stable signal to schedule against.

**Why comparative, not theoretical.** Scoring measured F0 against `pyopenjtalk`'s expected mora-level H/L pattern is the wrong grading target. Two independent sources: `onsei` (MIT) documents that theoretical sentence accent patterns diverge from native production (emphasis, emotion, slurring); older Japanese CAPT research finds that system-designated F0 bands do not correspond to native speakers' acceptance thresholds. `pyopenjtalk` **stays** — as the accent-type reference for display and for sanity-checking the reference recording, not as the thing being graded against.

**The reference recording is VOICEVOX output.** In a shadowing drill the tutor's line has already been synthesized, so a paired accent-correct reference exists at zero cost and with no corpus licensing. **Limitation, stated plainly and not in a footnote:** VOICEVOX is synthetic, so this scores similarity-to-TTS, not similarity-to-native. Its H/L pattern is OpenJTalk-correct; its prosody is not native. T3.6 bounds this.

**One segmentation mechanism, not two.** Frame-level CTC posteriors from the GOP pass supply the phoneme boundaries the DTW needs, so `onsei`'s Julius segmentation-kit dependency (Perl + C + HTK models) is not carried. Segmentation itself is *not* dropped — onsei's author moved from intensity-based DTW to phoneme segmentation deliberately after mixed results, so only its source changes. Contingent on §9 risk 9: if T3.2a finds no usable phone-set mapping, CTC posteriors may be unavailable in a workable inventory and Julius returns to the table.

### 6.2 Why pitch accent gets its own score

It's the most-neglected dimension in commercial apps and the one most responsible for sounding foreign despite correct grammar.

**"Nobody else is doing this at $0" was wrong.** JPitch (jpitch.org) does per-mora H/L grading against UniDic patterns, free and in-browser. `onsei` does sentence-level pitch assessment, open source. The accurate and narrower claim: Ocha's novelty is **integrating accent measurement into an SRS-scheduled free-conversation loop.** JPitch is a dictionary drill tool; `onsei` is an analyzer; neither is a tutor. Stated narrowly on purpose — do not overstate this again.

### 6.3 Gemma 4 audio as a second opinion (optional, phase 4)

Gemma 4 E4B accepts audio directly and does multilingual ASR and speech translation. Cost is 25 tokens per second of audio, 30s max clip, 16 kHz mono float32 in [-1, 1].

Run it **nightly, in batch, over the day's recordings** — never in the loop. It hears prosody and hesitation that transcription discards, and can produce qualitative notes ("you hesitated before every counter word") that the deterministic scorer structurally cannot. Treat its output as a coaching note, never as a score.

Resample with a Fourier method (`scipy.signal.resample` or `librosa` with `res_type='scipy'`) — the docs are explicit about this.

---

## 7. FSRS ↔ LLM integration

This is the "practice what you've already learned" mechanic, and it's the part that makes Ocha more than a chatbot.

### 7.1 Context Builder

Before every turn:

```python
due     = fsrs.due_items(limit=5)        # cards at/past retrievability threshold
known   = fsrs.known_items(min_reps=3)   # safe vocabulary/grammar pool
weak    = fsrs.lowest_stability(limit=3) # struggling items
```

These go into the system prompt as strong **steering, not deterministic enforcement**:

```
You are a Japanese conversation partner. Reply in 1–2 short sentences.

VOCABULARY: Use only words from KNOWN. If you must introduce a new word,
introduce exactly one and gloss it in English in parentheses.
KNOWN: {known}

TARGET: Steer the conversation so the learner naturally needs these:
{due}

REGISTER: Always use polite です/ます form. Never mix polite and plain forms.

AVOID: Do not use grammar beyond {level}.

Never break character to explain grammar. If asked a grammar question,
respond with exactly: [GRAMMAR_QUERY]
```

> **T0.5 MEASURED — the REGISTER line above is required, not decorative.** Without it, Gemma 4 mixed polite and plain within a single reply on 6 of 15 turns (`いいですね。あなたは今日、何を食べる？`), which models inconsistent politeness to a learner who cannot yet detect it. Adding the line took register compliance from 9/15 to 14/15 and vocabulary from 14/15 to 15/15. See `benchmarks/generation-probe.md`.
>
> **`enable_thinking=False` is a hard requirement of the LLM service.** Both candidates are reasoning models and emit a thinking channel by default. Beyond wrecking the latency budget, Gemma answered a grammar question with the sentinel *plus 399 characters of grammar explanation* — exactly what FR-5 forbids.

### 7.2 The correctness firewall

**Generation for fluency, lookup for correctness.**

The `[GRAMMAR_QUERY]` sentinel is the mechanism. When the model emits it, the app does *not* let the model answer.

> The firewall suppresses the **entire completed output whenever the contiguous token `GRAMMAR_QUERY` is present**, including damaged brackets and dirty surrounding text. It does not claim to structurally detect arbitrary marker-free grammar explanations. Curated package data is the only grammar-answer source.

A 4-bit local model will confidently give you a wrong explanation of は vs が, and you will not have the Japanese to detect it. That is the single highest-risk failure mode in this entire system. The firewall is not optional.

### 7.3 Observing a conversational turn

Free conversation can establish only that a target surface or lemma occurred. It stores one `mentioned` or `mentioned_after_prompt` observation per turn/item and never calls `record_review()`. Malformed Japanese such as `ご飯を食べるです` may contain `食べる`; that is not proof of correct recall. Only a future explicit validated drill may produce an FSRS rating.

---

## 8. Build phases

Sequenced so that each phase is independently useful if you stop there.

**Phase 0 — Bake-off (1 evening).** Record 20 utterances of your own Japanese, 10 pure and 10 code-switched. ASR bake-off is **kotoba-whisper-v2.0, kotoba-whisper-bilingual-v1.0, and whisper-large-v3** — see TASKS.md T0.3. Gemma 4 E4B audio is **excluded**, which leaves §2.1's specialised-vs-generalist justification untested; DECISION.md records that as an open assumption. Measure `mlx_lm` tok/s for Gemma 4 26B-A4B and Qwen 3.5 9B, then run the T0.7 contention test with all three loop services resident. *This phase decides §2, §3, §4, and §5.1.* Do not skip it.

**Phase 1 — Text loop (1 weekend).** FastAPI + MLX + SQLite + py-fsrs. Typed input, typed output, no audio. Proves the Context Builder and grammar firewall.

**Phase 2 — Voice loop (2–3 weekends).** Pipecat over WebSocket, VAD, whisper, VOICEVOX, sentence chunking, plus the browser client (§2.3). Target: G1a always, G1b p50 ≤ 3.2 s. This is where the latency work lives, and the client is now part of the phase rather than a thin wrapper on it.

**Phase 3 — Pronunciation (post-October 2026).** Alignment-free GOP + pyworld + comparative DTW accent scorer against a VOICEVOX reference. Feed scores into FSRS grading, with the accent cap disabled until T3.6 validates the reference. Deferred as a deliberate trade — see PRD §10.

**Phase 4 — Client polish + Gemma audio grader (ongoing).** Offline batch analysis, progress views, streak mechanics.

### 8.1 Schedule reality check

It is 30 July 2026, and the deadline is in October. Phases 0–2 is realistically 3–4 focused weekends, and it will expand — every voice pipeline does.

The honest options:

- ~~**Ship Phase 0 + 1 now** (two weekends), then freeze.~~ **Rejected — see PRD §10.** A text-only tutor is a rebuild of the thing this product rejects, so Phase 1 is not a shippable stopping point.
- **Build Phases 0–2 now** and defer Phase 3 (pitch accent) past October. This is the chosen path: the scope cut is at the feature level, not the phase level.

Phase 1 alone is genuinely useful and preserves the option. Phase 2 is where the project starts eating unbounded time.

---

## 9. Open risks and unvalidated assumptions

| # | Assumption | How to falsify | Severity |
|---|---|---|---|
| 1 | ~~Machine is M4 Pro-class bandwidth~~ | **RESOLVED, T0.1.** Base M4, measured 103.2 GB/s (86% of the ~120 spec). MoE mandatory; dense 27B struck. See `benchmarks/hardware.md`. | ~~High~~ — closed |
| 2 | Gemma 4 26B-A4B is competent at Japanese grammar | 20 hand-written は/が, は/も, transitivity prompts, checked against a reference | **High** |
| 3 | ~~kotoba-whisper handles *your* accent~~ | **RESOLVED, T0.3: it does not.** 17.09% CER on pure Japanese, disqualified at Stage 1. Replaced by whisper-large-v3 on MLX (2.56%). | ~~Medium~~ — closed |
| 4 | ~~Pipecat's Mac WebRTC path is stable~~ **MOOT, §2.3.** Replaced by: *HFP-duplex headset audio is good enough for whisper.* iOS cannot pair the built-in mic with A2DP output, so a headset means narrowband capture. The corpus was recorded on the MacBook's mic; production CER on HFP audio is unmeasured. Falsify in T2.3 by re-recording ≥10 corpus utterances through the app on both capture paths. | **Medium-High** |
| 5 | Alignment-free GOP transfers to Japanese | Published results are English and child-speech corpora (speechocean762, CMU Kids). Japanese transfer is **untested**. Validate on 10 deliberately mispronounced utterances before trusting any segmental score. | Medium |
| 6 | 26 GB budget survives real sessions | T0.7 contention test, then `footprint` during a 20-min session | Medium |
| 7 | Sustained thermals on a laptop | 30-min session, watch for throttling | Low-Medium |
| 8 | VOICEVOX is an adequate accent reference | T3.6: ~10 native recordings, check rank correlation against VOICEVOX scoring | Medium |
| 9 | **`pyopenjtalk` phones map to espeak IPA with acceptable loss** | T3.2a. The GOP model emits espeak IPA (387 labels); `pyopenjtalk` emits a different non-IPA Japanese set; espeak's own Japanese G2P resolves kanji readings badly. If no clean mapping exists, alignment-free GOP must be **replaced, not patched** — this invalidates §6.1's segmental score and, via the segmentation collapse, T3.4's boundaries too. | **High** — gates the entire Phase 3 segmental design |
| 10 | The Dec-2021 GOP checkpoint runs on current tooling | `lastModified` 2021-12-10, untouched ~4.5 years. `Wav2Vec2Processor` TypeError reported on `transformers` 4.46. Pin `transformers`; add `espeak-ng` + `phonemizer`. | Medium |

**Risk 9 gates Phase 3.** Everything in §6.1 downstream of the GOP stage assumes phoneme posteriors in an inventory that the Japanese reference sequence can be expressed in. That assumption has not been checked. T3.2a checks it and reports before anything is implemented; a lossy mapping shipped to preserve the plan is the failure mode to avoid.

**Phase 3 dependencies** (post-October, deliberately *not* in the Phase 0 environment): `espeak-ng` (system, via brew), `phonemizer`, and a pinned `transformers`.

---

## 10. Things deliberately excluded

- **Cloud anything.** No ASR, TTS, or LLM APIs. Hard constraint.
- **Forced alignment.** Removed deliberately. Alignment error is a known contaminant of pronunciation scoring, and MFA is trained on native speech — it degrades in exactly the cases we most want to measure. Alignment-free GOP removes the dependency rather than mitigating it.
- **Julius segmentation-kit.** `onsei` uses it for phoneme boundaries. Perl + C + HTK models, Ubuntu-tested. CTC posteriors from the GOP pass already provide boundaries, so this dependency is not carried. See §6.1.
- **Fine-tuning.** Tempting, expensive in time, and the Context Builder gets you most of the way.
- **Speaker diarization.** Single speaker. kotoba-whisper-v2.2's diarization stack is pure overhead here.
- **Voice cloning TTS** (Fish Speech, Chatterbox, Irodori-TTS). Expressive, but accent-unreliable. Wrong trade for a learner.
- **Multi-user.** One user. SQLite, no auth beyond Tailscale.

---

## Sources consulted (July 2026)

- Gemma audio capabilities — https://ai.google.dev/gemma/docs/capabilities/audio
- kotoba-whisper v2.0 / v2.2 — https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0
- Pipecat — https://github.com/pipecat-ai/pipecat
- Model landscape figures drawn from mid-2026 comparison write-ups; treat all tok/s and benchmark numbers as hypotheses to verify in Phase 0.
