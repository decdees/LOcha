# T2.6 — the voice loop, measured from inside the pipeline

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
