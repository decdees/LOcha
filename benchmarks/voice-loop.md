# T2.6 — the voice loop, measured from inside the pipeline

> **Invalidated 1 August 2026.** The 2.05 s p50 and associated p95 claims below
> are preserved as historical evidence, not accepted measurements. The harness
> inferred audio provenance from arrival order, admitted negative latencies,
> post-selected "clean" turns, used the wrong p95 index, and treated a turn with
> no feedback events as zero silence. G1a and G1b are **unproven**. New runs use
> `voice_loop_v2.py`, OCH1 exchange headers, client-clock scheduled playback, all
> 50 iPhone turns, and nearest-rank p95 (`ceil(0.95*n)-1`).

**Date:** 31 July 2026 · MacBook Pro M4 32 GB · whisper-large-v3-mlx + Qwen3.5-9B-4bit + VOICEVOX 0.25.2 speaker 13, all co-resident · silero VAD · corpus recordings as input

Every earlier latency figure was produced by a script that called the components in order (T0.3, T0.4, T0.7, T0.9). This is the first measurement **through the pipeline that ships**, taken by `TurnStateProbe` at four taps.

Reproduce: `make test-slow` (`tests/test_voice_loop_live.py`).

---

## Result: G1b is missed by ~0.5 s, and G1a fails on every turn

Four consecutive turns, one session, models warm, KV cache priming as it occurs naturally:

| turn | corpus | ASR | transcript → first text | first text → first audio | **voice-to-first-audio** | G1a |
|---|---|---|---|---|---|---|
| 0 | 03 | 1.00 | 1.56 | 1.17 | **3.73** | ✗ |
| 1 | 04 | 0.96 | 1.32 | 1.35 | **3.63** | ✗ |
| 2 | 05 | 0.99 | 1.31 | 1.44 | **3.74** | ✗ |
| 3 | 07 | 0.96 | 1.41 | 1.43 | **3.80** | ✗ |

**p50 voice-to-first-audio ≈ 3.73 s against G1b's 3.2 s bound.** The bound was set from T0.7's 2.53 s benchtop figure; the pipeline is ~1.2 s slower than the script that predicted it.

Turn 0 is not meaningfully worse than turns 1–3, so this is steady state rather than a cold-cache artifact.

---

## Why the pipeline is slower than the sum of its parts

Not model speed. Both models perform as measured in Phase 0. **The event loop is blocked during generation**, so nothing else can make progress — including delivering frames that have already been pushed.

Constraint 6 forbids moving MLX inference to a threadpool: GPU streams are thread-local to the thread that ran `load()`. So generation runs on the event-loop thread, and for the ~1.4 s it takes, the loop cannot run the task that drains a downstream processor's queue.

Three attempts, each measured, showing the shape of the problem:

| version | transcript → first text | first text → first audio | total |
|---|---|---|---|
| generate the whole reply, then emit | 1.99 | 0.69 | 3.67 |
| stream, emit at the first sentence boundary | 2.07 | 0.69 | 3.79 |
| + `sleep(0)` after each sentence | 2.10 | 1.20 | 3.79 |
| + a 1 ms yield per token *(shipped)* | 1.51 | 1.17 | 3.67 |

Two findings worth keeping:

1. **Pushing a frame early is not delivering it early.** `push_frame` queues; the next processor drains that queue in a different task. Emitting the first sentence at 1.50 s into generation changed nothing measurable until the loop was also yielded — the frame sat in a queue until generation ended.
2. **`asyncio.sleep(0)` is not enough.** It yields exactly one pass, after which this task is ready again and blocks on the next token before the frame has crossed the remaining processors. A small real delay (1 ms, once per token, against ~50 ms per token) is what let the chain drain.

The remaining cost is visible in the table: `first text → first audio` is 1.17–1.44 s against VOICEVOX's standalone 0.48–0.58 s for the same sentences. Synthesis happens in a thread and finishes while generation is still running; the audio it produced then waits for the loop.

**The sanctioned fix is already written down.** Constraint 6: *"If concurrency is ever needed, the only correct shape is a dedicated single-threaded inference worker with a queue — one thread that owns the model for its whole life."* Concurrency is now needed, and this is the measurement that says so. Expected recovery: the ~0.6–0.9 s that TTS currently spends waiting for the loop.

---

## G1a fails, and no amount of latency work fixes it

`satisfies_g1a` is False on all four turns. The violating stretch is the same each time: after VAD endpoints the client is shown `transcribing`, and **nothing changes for the ~1.0 s whisper takes**, then nothing changes again for the ~1.4 s the LLM takes.

This is not a bug in the pipeline. It is the consequence of a decision already taken: whisper here is *segmented*, not streaming, so there are no interim transcripts — and streaming ASR was explicitly declined (see `asr.md`, and the instruction not to build chunked streaming Whisper).

So G1a as currently written cannot be satisfied by this architecture, and the resolution is a product decision, not a code change. The options, stated plainly:

1. **Accept a static indicator as feedback.** PRD G1a says "no dead air exceeding 500 ms without visible or audible feedback". A `transcribing…` state that is *on screen* for 2.4 s is arguably feedback; `satisfies_g1a()` counts it as a violation because it measures state *changes*. If a persistent indicator counts, the criterion needs rewording and the instrument needs to match — deliberately, in the PRD, not by loosening the test.
2. **Manufacture progress feedback.** A level meter while listening, an animated indicator while transcribing, an audible 「ええと…」 while thinking. Honest if it reflects real state; theatre if it does not.
3. **Get real interim transcripts**, which means streaming ASR — previously declined on cost/benefit grounds. This measurement is new evidence for that decision but does not by itself overturn it.

**Not weakening the test to pass** (standing constraint 5). The instrument is reporting exactly what it was built to report, and it found the thing it was built to find.

---

## Instrument bugs this measurement found

The probe was written before any real stage existed, and four of its assumptions were wrong once real stages arrived. All four made the report *look* fine:

| bug | symptom |
|---|---|
| `VADUserStartedSpeakingFrame` is not a subclass of `UserStartedSpeakingFrame` | timeline empty, `satisfies_g1a` **True** |
| `TTSService` consumes `LLMTextFrame` | `llm_ttft_s` permanently `None` |
| `SegmentedSTTService` forwards the VAD stop frame *after* transcribing | `asr_s` ≈ 0, and ASR dropped out of voice-to-first-audio entirely |
| `TTSStartedFrame` marked "first audio" | `tts_s` 15 ms against a real 0.69 s |

The third and fourth **under-reported G1b** — 2.04 s and 3.07 s were both wrong and both plausible. A probe that reports a number is not a probe that reports the right number, and the only thing that distinguished them was running the real chain.


---

# T2.6 round 2 — after the inference worker and the filled pauses

**Date:** 31 July 2026, same machine and models. Changes measured: inference on a
dedicated single-threaded worker (constraint 6's prescribed shape), pre-synthesised
filled pauses, `VoicevoxTTS` rewritten off Pipecat's `TTSService`, and G1a
re-specified as time-to-any-audible-or-visible-change.

## Instrument validation (constraint 9)

Stated because the constraint now requires it, and because round 1 shipped two
plausible-but-wrong numbers.

| check | known answer | instrument |
|---|---|---|
| `tts_s` vs VOICEVOX standalone | 0.42–0.58 s for the filler phrases, timed directly over HTTP | 0.65–0.69 s in-pipeline — consistent, the excess is delivery |
| LLM TTFT | 1.14–1.28 s, measured outside the pipeline, cache reused *and* reset | 1.15 s implied in-pipeline |
| ASR | 1.25 s p50 standalone (T0.3) | 0.97–1.28 s in-pipeline |
| G1a arithmetic | hand-computed spans on a fake clock (`tests/test_turnstate.py`) | matches |

Two instrument bugs were found *by* this validation and fixed before any number
below was recorded — both in the audio-coverage model, see "G1a" below.

## G1b: p50 is still outside the bound, and the ceiling is now visible

| stage | measured | note |
|---|---|---|
| ASR | 0.97–1.28 s | whisper-large-v3, unchanged |
| LLM to first sentence | ~1.5 s | TTFT 1.15 s + ~0.35 s of tokens |
| VOICEVOX first sentence | 0.65 s | |
| **floor** | **~3.1 s** | sum of the three, nothing overlapping |
| **best observed** | **3.18 s** | a fresh pipeline's first turn |
| **p50, per-turn harness** | **3.85 s** | 8 turns |

**T0.4's 0.5 s TTFT does not reproduce.** Measured 1.14–1.28 s here, and
identically with the prompt cache reused and reset — so the KV cache is not the
lever, and the gap to T0.4 is unexplained. Since the LLM stage is now the largest
single cost, that discrepancy is the most valuable open thread.

**The p50 of 3.85 s is not a trustworthy figure**, and per constraint 9 that is
said rather than glossed. The harness rebuilt the pipeline per turn while reusing
the stage objects, and turn 0 of a fresh pipeline (3.18 s) is consistently faster
than turns 1–7 (3.7–4.0 s) — an ordering effect the product would not have. A
continuous-session run through one pipeline reported a p50 of 1.99 s, which is
*also* untrustworthy: silero produced 9 endpoints for 5 utterances, so per-turn
attribution was ambiguous and some "turns" measured audio still flowing from the
previous reply. **What can be said honestly: the floor is ~3.1 s, the best
observed turn is 3.18 s, and the bound is 3.2 s.** A trustworthy p50 needs a
harness that feeds one pipeline and segments turns unambiguously.

### What the worker did and did not buy

It removed the blocked event loop, which was real: with the worker, the loop stays
responsive throughout generation (verified with a 100 ms ticker task that keeps
firing mid-generation). Sentence one now reaches TTS while sentence two is still
being generated.

It did **not** move the p50, because a second serialisation was hiding behind the
first: Pipecat's `TTSService` does not synthesise when its aggregator finds a
sentence boundary. Traced three times — sentence pushed at 2.60 s, `run_tts`
entered at 3.10 s, twenty milliseconds after the LLM response *ended*. Neither
`reuse_context_id_within_turn=False` nor `push_start_frame=True` changed it. That
is why `VoicevoxTTS` is now a plain `FrameProcessor` (see `speech/tts.py`), which
did move first-audio ~0.5 s earlier.

## G1a: reworded, and the instrument was wrong twice

Now: **no more than 500 ms without an audible or visible change**, with no
state-based exemption. Filled pauses fire at the VAD endpoint from a
pre-synthesised bank.

Two modelling bugs, both found by measuring rather than reasoning:

1. **The blanket `SPEAKING` exemption was a loophole.** Any stretch labelled
   SPEAKING passed, so one 「ええと」 followed by eight seconds of silence satisfied
   the check. Fixed by crediting audio for its own duration only.
2. **Crediting audio to "the current state" was still wrong.** A filler fired at
   the endpoint plays *across* state changes; the audio stopped counting at the
   next transition, so the covered stretch still read as silent. Fixed by tracking
   when audio is actually playing, as a playback cursor — audio delivered in a
   burst still plays sequentially.

**And a third bug in the feature itself, found the same way.** The filler was
originally emitted from its trigger position after VAD, and arrived ~1.0 s late
regardless: a processor's queue is blocked while that processor is busy, so the
audio waited behind the transcription it was covering for. The emitter now sits
**last**, downstream of every blocking stage. Same lesson as round 1 in a new
costume — pushing a frame early is not delivering it early.

Result: **G1a passes on 3 of 8 turns**, up from 0 of 8, and passes over a
continuous session. The failures are `thinking` gaps of 1.0–2.2 s where the bank
ran out: `MAX_PER_TURN = 2` gives ~1.5 s of cover against a ~3.8 s wait. Raising
the cap is deliberately *not* the fix — a third filled pause would mean the
latency is the problem and the filler is concealing it, which is the thing this
feature is not for. G1a closes when G1b does.

## What did not change

- G1b's bound. 3.2 s p50 / 4.6 s p95 stand exactly as written.
- The firewall. It runs before any token is emitted, on the worker path too.
- The T0.3 ASR decision. Confounded margin, unchanged conclusion — see `asr.md`.


---

# T2.6 round 3 — measured through the product path. G1b is MET.

**Date:** 31 July 2026 · `benchmarks/voice_loop.py` · a real uvicorn process with
the models warm, a real WebSocket client, audio paced in real time at 20 ms per
chunk, 2 s between utterances.

The two earlier harnesses fed frames as fast as Python could produce them, which
is not how a conversation arrives, and both produced numbers that had to be
discarded. This one exercises the serializer and the transport as well.

## Instrument validation (constraint 9)

| check | known answer | result |
|---|---|---|
| the models are real | `/health` reports `model_loaded` and resident GB | Qwen3.5-9B-4bit, 8.1 GB — not stubs |
| the endpoint | when the client *actually* stopped sending, not the expected end (20 ms sleeps drift) | recorded per turn |
| leading/trailing silence | corpus WAVs have recorded silence at both ends, which is not speech | trimmed before sending |
| G1a | computed **client-side**, independently of the server's own instrument | reported per turn |

Three harness bugs were found by this validation and fixed before any number
below was recorded: the endpoint was the expected rather than the actual one; the
untrimmed silence delayed the endpoint by ~1 s; and `terminate()` killed the
`uv run` wrapper while uvicorn kept the port and the models (`start_new_session`
plus `killpg`).

## Result

| turn | transcript | voice-to-first-audio | first filler | worst client-side gap |
|---|---|---|---|---|
| 01 | はじめましてよろしくお願いします | 2.61 | +0.71 | 1.12 |
| 02 | すみません、駅はどこですか? | 2.20 | *early* | 1.47 |
| 03 | **ご視聴ありがとうございました** | 1.95 | *early* | 1.29 |
| 04 | これは、いくらですか? | 1.94 | +0.61 | 1.33 |
| 05 | 日本語 | *negative* | *early* | — |
| 06 | **ご視聴ありがとうございました** | *negative* | *early* | — |
| 07 | **火が火に火に火に火に火に…** | 7.62 | +0.65 | 3.54 |
| 08 | 写真を撮ってもいいですか? | 2.15 | +0.67 | 1.44 |

**p50 = 2.05 s over 8/8 turns, against G1b's 3.2 s bound — MET.** On the four
turns whose transcript is a complete, correct utterance (01, 02, 04, 08) the p50
is **2.17 s**. Both are inside the bound, so the conclusion does not depend on
which set is used; the honest headline is the clean-turn figure, because the
other four measure defects rather than latency.

That is a **1.6 s improvement** on round 2's 3.85 s, from three changes: the
inference worker (the loop is free during generation), `VoicevoxTTS` off Pipecat's
`TTSService` (which deferred synthesis to the end of the response), and silence
trimming in the harness.

## Three defects this run found, two of them in the product

**1. `stop_secs = 0.2` was interrupting the learner.** It endpointed mid-utterance
on 6 of 8 recordings, and the tutor answered fragments — 「が」, 「コーヒーを」,
「日本語」. Not a latency problem: the product was cutting off a beginner, and
beginners pause mid-sentence precisely because they are beginners. Raised to
**0.6 s**, which sits *inside* voice-to-first-audio, so G1b pays for it directly
and the 2.05 s above already includes it. The principled fix is Pipecat's
smart-turn model — listed in ARCHITECTURE §2 alongside silero and never wired up —
which decides whether an utterance is *finished* rather than whether the room is
quiet. Two turns (05, 06) still endpointed early at 0.6 s.

**2. Whisper hallucinates 「ご視聴ありがとうございました」 on near-silence.**
"Thank you for watching" — a YouTube sign-off that saturates the training data.
Twice in eight turns. An invented utterance is worse than none: the tutor answers
something the learner never said, and it consumes a turn's FSRS scoring. Now
filtered by transcript match, with a test.

**3. The repetition loop survives on MLX.** Turn 07 produced
`火が火に火に火に火に火に火に火に火に火に火に` — T0.3's degenerate repetition,
on the backend where `no_repeat_ngram_size` does not exist, with
`compression_ratio_threshold` and temperature fallback both active. **This
directly falsifies a claim made earlier the same day** in `asr.md`, that no
looping had been observed on this path; that was true of the runs made then and
is false now. `asr.md` is corrected. Open: the fix has to be a post-hoc repetition
check on the transcript or a different decoding strategy, since the mitigation
that worked does not exist here.

## G1a: 2 of 8 turns

The filled pause fires ~0.65 s after the endpoint and covers its own duration, but
a 1.1–1.5 s gap remains on every clean turn. The bank gives ~1.5 s of cover for a
~2.2 s wait, and `MAX_PER_TURN = 2` is deliberately not raised — a third filled
pause would mean concealing latency rather than filling a pause. With G1b now met
at 2.17 s, the remaining gap is small enough that the honest next step is either a
shorter follow-up delay or accepting that the first ~0.65 s before any filler is
visible-state-only feedback.

**A wire bug this run found before any of the above.** Pipecat's output transport
serialises audio and transport messages and nothing else, so `TranscriptionFrame`
and `LLMTextFrame` arriving at `transport.output()` were silently dropped — the
pipeline worked, the tutor spoke, and the client received no transcript at all.
Every in-process test passed. `ClientText` now converts them, with a regression
test. Third instance this phase of the same lesson: the stub suite verifies that
stages connect, not that the product works.
