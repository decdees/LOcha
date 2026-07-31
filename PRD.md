# Ocha — Product Requirements Document
**Version:** 1.0
**Date:** 30 July 2026
**Status:** Approved for Phase 0–1

---

## 1. Summary

Ocha is a fully self-hosted, speech-to-speech Japanese conversation tutor. It runs entirely on a single MacBook Pro M4 (32 GB), is accessed from an iPhone PWA over Tailscale, and costs $0/month in recurring fees.

It differs from commercial apps (Falou, Duolingo, Speak) on two axes. The **long-term** differentiator is that it measures Japanese pitch accent deterministically and schedules practice against that measurement — deferred to post-October 2026 (§10). What ships before then is spoken production practice against a vocabulary pool the learner has actually earned, with grammar served from curated reference rather than generated, at $0/month. That is not the full thesis, and §10 states the trade plainly.

---

## 2. Problem statement

The target learner is an absolute beginner in Japanese, aiming at travel and conversational survival. Existing apps have three problems:

1. **Recurring cost.** Comparable apps charge a recurring annual subscription. Hard budget constraint of $0.
2. **Pitch accent is ignored *by tutors*.** Commercial conversation apps score "pronunciation" as a single opaque number, almost always segmental-only. Free tools that do measure it exist — JPitch (per-mora H/L against UniDic), `onsei` (sentence-level, open source) — but they are a dictionary drill and an analyzer respectively. Neither schedules practice or holds a conversation. **The gap is integration, not measurement.**
3. **No control over scheduling.** Closed SRS, no ability to tie spaced repetition to production quality rather than recognition.

---

## 3. Goals

| # | Goal | Measure |
|---|---|---|
| G1a | The conversation never *feels* broken | **No dead air exceeding 500 ms without visible or audible feedback** — transcript appearing, listening state, thinking state. **FAILING as of T2.6, and not for want of latency work.** Segmented whisper produces no interim transcripts, so `transcribing` is on screen unchanged for ~1.0 s and `thinking` for ~1.4 s. Whether a *static* indicator counts as feedback is a wording question this criterion has not answered; three options are set out in `benchmarks/voice-loop.md` and one of them needs deciding. The test is not being loosened to pass. |
| G1b | Turn latency stays within a measured bound | **p50 voice-to-first-audio ≤ 3.2 s; p95 ≤ 4.6 s** — a regression bound. **NOT HELD as of T2.6: measured 3.73 s p50 through the real pipeline** (`benchmarks/voice-loop.md`). The bound was derived from a benchtop script; the pipeline is ~1.2 s slower because MLX generation blocks the event loop and pushed frames cannot be delivered. Recovery is a dedicated single-threaded inference worker (standing constraint 6's own prescription), not a looser bound. |
| G2 | Never teach incorrect grammar | 100% of grammar explanations served from curated reference, 0% from model generation |
| G3 | Measure pitch accent per utterance | Three stable scores per turn: segmental, accent, rhythm. **Deferred post-October 2026 — see §10.** |
| G4 | Schedule practice from production quality | FSRS rating derived from usage + pronunciation, not self-report |
| G5 | Zero recurring cost | No cloud API in any code path |

## 4. Non-goals

- Multi-user, authentication, or account management. One user, Tailscale is the perimeter.
- Kanji writing practice, reading comprehension of long text, JLPT prep.
- **Native mobile app before October. PWA only.** ~~Reversed 31 July 2026~~ — **and reversed back the same day, deliberately.** The native pivot lasted one commit. It is restored as a non-goal because a SwiftUI client is a second codebase (`AVAudioEngine`, transport, transcript UI, furigana, barge-in), realistically 3–4 weekends, and it was about to be started *before Phase 2 was finished*, nine weeks from the deadline. That is precisely the scope breach §10 exists to prevent. Free provisioning also expires every 7 days with no warning, which is corrosive for a daily-habit SRS tool. Native iOS is now a **Phase 4, post-October** item (TASKS T4.5).
- Fine-tuning any model.
- Offline operation on the phone. The Mac must be reachable.
- Voice cloning or expressive/emotional TTS.

---

## 5. Users

Single user, self-hosting. Design parameters that drive functional requirements:

- **Japanese level:** absolute beginner. Drives the vocabulary constraint and the one-new-word-per-turn rule (FR-3).
- **L1 is Hindi.** Drives FR-7's `hindi_contrast` and `interference_warning` fields. Hindi and Japanese share SOV order, postpositions, pro-drop, and head-final relative clauses — real positive transfer. But Hindi ने (ergative) is *not* Japanese が, and Hindi's weight-based stress interferes with Japanese pitch accent. These are the traps the reference must flag.
- **Operator is technical.** No hand-holding in setup or ops docs.

**Interface language is English.** Hindi appears only as an optional contrast note on grammar entries (see FR-7).

---

## 6. Functional requirements

### FR-1 — Speech input
- Continuous listening with VAD-based endpointing. No push-to-talk.
- Detects end of utterance within 200 ms of speech ending.
- Barge-in supported: user speaking interrupts TTS playback.

### FR-2 — Speech recognition
- Japanese ASR, optimised for the target language rather than generalist.
- Must degrade gracefully on Japanese/English code-switching (beginner reality: "えーと, how do you say receipt?").
- Transcript displayed to user for every turn. Non-negotiable — it is the only way to catch ASR error.

### FR-3 — Conversation
- LLM replies in Japanese, 1–2 sentences maximum, enforced by both prompt and `max_tokens`.
- Vocabulary constrained to the user's known-item pool. At most one new word per turn, glossed in English.
- Never breaks character to explain grammar (see FR-5).

### FR-4 — Speech output
- Japanese TTS with deterministic, inspectable pitch accent.
- Streaming: synthesis begins on the first complete sentence, not the full response.

### FR-5 — Grammar firewall *(critical)*
- The model emits the literal sentinel `[GRAMMAR_QUERY]` when asked a grammar question.
- The application intercepts this and serves an explanation from a curated local JSON reference keyed to FSRS item IDs.
- **The model's own grammar explanations are never shown to the user under any circumstance.**
- If no reference entry exists, show "not yet documented" and log it for manual authoring. Do not fall back to generation.

### FR-6 — Pronunciation assessment

> **Deferred to post-October 2026.** Specified here in full because Phase 2's schema and the FR-8 accent cap are designed around it. Not built before the October 2026 deadline.

- Runs asynchronously, outside the conversation loop. Never blocks a turn.
- Produces three independent scores per utterance:
  - **Segmental** — alignment-free GOP (wav2vec2-xlsr-53 phoneme posteriors). No forced aligner.
  - **Accent** — DTW-aligned F0 distance vs a **reference recording**, not vs a theoretical H/L pattern. `pyopenjtalk` supplies the accent type for display, not the grading target.
  - **Rhythm** — mora timing regularity vs reference
- Scores are deterministic: identical audio produces identical scores.
- **Presentation:** all three are shown as a **trend over time, never as an absolute grade.** Published GOP-to-human correlation is ~0.62–0.64 against human-to-human agreement of ~0.57 — automatic scoring agrees with expert humans about as well as humans agree with each other. That is the realistic ceiling, and an absolute grade would overstate it.

### FR-7 — Grammar reference
- Local JSON, hand-authored, keyed to FSRS item IDs.
- Schema per entry:
  ```json
  {
    "id": "particle_wa_ga",
    "en": "は marks topic; が marks new or identified subject.",
    "examples": ["..."],
    "hindi_contrast": "Closer to तो vs plain subject. NOT ने — ने is ergative and unrelated.",
    "interference_warning": true,
    "source": "Tae Kim §3.1"
  }
  ```
- `hindi_contrast` is optional and only present where the analogy genuinely holds.
- `interference_warning: true` surfaces the entry prominently — these are the traps where Hindi intuition actively misleads.

### FR-8 — Scheduling
- FSRS via `py-fsrs`. Ratings derived, never self-reported.
- Derivation table:

  | Signal | Rating |
  |---|---|
  | Target item used correctly, unprompted | Good / Easy |
  | Used after a hint | Hard |
  | Avoided or substituted | Again |
  | Accent score below threshold | **PROVISIONAL — disabled by default.** Must not gate scheduling until validated by T3.6. Stubbed and inert through Phases 1–2; the code path and its test exist, the score does not. |

### FR-9 — Client
- **Browser client, served by the Mac and run on the Mac** for Phase 2. Streams 16 kHz mono PCM over a WebSocket to `/ws`. No audio-routing quirk, no provisioning, no Tailscale hop — the three things that would otherwise be debugged simultaneously with the pipeline.
- The **iPhone PWA is the shipping target** and is unblocked: both T2.1 pre-flight gates passed (`benchmarks/ios-audio.md`) — `getUserMedia` works in standalone home-screen mode and audio output stays on the headset with the mic live. Moving from the Mac browser to the phone is the same client over Tailscale.
- A **native SwiftUI app is Phase 4, post-October** (TASKS T4.5), not before.
- Displays: live transcript (user + tutor), furigana on tutor output, pitch accent visualisation after each turn, grammar panel on `[GRAMMAR_QUERY]`, and an explicit turn-state indicator (listening / transcribing / thinking / speaking) — G1a is a client requirement as much as a server one.
- Text input mode available as a fallback *within* the voice-first app, for noisy environments and ASR failures. It is not a separate interface and not a Phase 1 deliverable — Phase 1 exposes `POST /turn` and no client.
- **Audio routing remains a known constraint of the phone, whatever the client is.** iOS cannot combine the built-in microphone with A2DP output; requesting the built-in mic forces output to the speaker. With a headset connected the session is HFP duplex in both directions, at narrowband quality into the ASR. Unmeasured — T2.3 measures it, and the fallback if CER degrades is phone mic + phone speaker.

---

## 7. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Total resident memory across all services < 27 GB during a session |
| NFR-2 | All models loaded and warm at all times; no cold-load in the request path |
| NFR-3 | No outbound network calls except Tailscale transport. Enforced by test. |
| NFR-4 | All state in a single SQLite file (WAL mode), trivially backup-able |
| NFR-5 | Every service startable via one `make dev`; model server runs under `launchd` |
| NFR-6 | Sustained 30-minute session without thermal throttling degrading latency >20% |

---

## 7a. G1 restructured — why one number was the wrong shape

The original G1 (p50 < 1200 ms) was written before anything was measured and had no grounding in learner perception. Phase 0 measured **p50 3.03 s** for the shipped configuration on a clean machine (T0.9), so the target was not merely missed but never derived from anything.

Replacing 1200 ms with a different single number would repeat the mistake. G1 is therefore split, because the original conflated two questions that have different answers and different fixes:

**G1a — does it feel broken?** This is the one that decides whether the product is usable, and it is *not* a function of total latency. A silent 3-second gap reads as a crash; 3 seconds with the transcript appearing as you speak, then a visible thinking state, reads as a tutor considering their answer. **G1a is achievable today at the measured 3.03 s**, and it makes the feedback states (T2.8) load-bearing rather than polish.

**G1b — is it getting worse?** A ceiling grounded in measurement:

| | value | basis |
|---|---|---|
| p50 ≤ **3.2 s** | measured 3.03 s + 0.18 s (VAD, network) | T0.9, clean boot, shipped config |
| p95 ≤ **4.6 s** | measured 4.37 s + 0.18 s | T0.9 |

This is a **regression bound, not an aspiration** — it says "do not get worse", and it is the only kind of number Phase 0 can honestly justify. The engineering target of ≤ 2.2 s depends on streaming ASR, which is unbuilt and, after Qwen3-ASR was disqualified in T0.8, no longer available off the shelf. It becomes a commitment when it is measured, not before.

---

## 8. Success criteria

**Phase 0 is successful if** it produces a written benchmark report with measured CER per ASR candidate and measured tok/s per LLM candidate on this specific machine. No code beyond benchmark scripts.

**Phase 1 is successful if** a scripted 10-turn exchange against `POST /turn` (curl or a test harness — there is no client) keeps vocabulary within the known pool, updates FSRS state correctly, and routes every grammar question through the firewall. Phase 1 is a build-order step, not a shippable state.

**Phase 2 is successful if** G1a holds over 50 real turns — no dead air beyond 500 ms without feedback — and p50 voice-to-first-audio stays within the G1b bound. The original "under 1200 ms" criterion is retired: T0.9 measured 3.03 s on the shipped configuration and no combination of built components reaches 1200 ms on this hardware.

**Phase 3 is successful if** accent scores separate deliberately-correct from deliberately-wrong pitch patterns on a hand-built 20-utterance test set. *(Post-October 2026.)*

---

## 9. Key risks

| Risk | Impact | Mitigation |
|---|---|---|
| Base M4 bandwidth forces a weaker model | Conversation quality drops | Phase 0 measures this before any architecture commitment |
| Phoneme inventories don't reconcile (`pyopenjtalk` vs espeak IPA) | FR-6's segmental score unimplementable as specified | T3.2a reports a go/no-go before any scorer is written. If no clean mapping exists, alignment-free GOP is replaced, not patched. See ARCHITECTURE §9 risk 9. |
| Alignment-free GOP doesn't transfer to Japanese | Segmental score is noise | Published on English and child-speech corpora only. Validate on 10 deliberately mispronounced utterances before trusting it. |
| Local model's Japanese is inadequate | Core value prop fails | Phase 0 grammar probe set; firewall limits blast radius |
| **Hard external deadline, October 2026** | Project stalls at partial state | Scope cut at the *feature* level, not the phase level: Phases 0–2 (voice conversation) ship; Phase 3 (pitch accent) defers. Phase 1 is not a stopping point — a text-only Ocha is not a smaller Ocha. |

---

## 10. Scope decision

**Phases 0, 1, and 2 are in scope. Phase 3 is deferred to post-October 2026.**

### Why Phase 1 is not a milestone

The product thesis is that text-based Japanese apps are ineffective and that spoken production practice is the entire reason Ocha exists. A text-only Ocha is therefore not a smaller version of the product — it is a rebuild of the thing being rejected. Phase 1 is first in build *order* for one reason: the Context Builder, FSRS rating derivation, and grammar firewall are pure logic, and debugging pure logic through a streaming audio pipeline costs far more than debugging it over curl. It is a step. It does not ship.

### The deliberate trade

Voice conversation and pitch-accent scoring cannot both ship before a hard external deadline in October 2026. Voice wins, because without it there is no product. Consequences, accepted with eyes open:

- Until Phase 3 lands, Ocha's stated differentiator is not shipping. The honest interim claim is "free, self-hosted, vocabulary-constrained spoken practice," not "the app that measures pitch accent."
- The FR-8 accent cap stays inert, so FSRS ratings derive from usage alone. A word produced with wrong accent will be scheduled as known. This is a known, accepted inaccuracy in the scheduler, not an oversight.
- Phase 2 must not foreclose Phase 3: `utterances` retains raw 16 kHz audio and the `pronunciation_scores` table ships empty in Phase 1 rather than being added later.

This is a trade, not an omission.
