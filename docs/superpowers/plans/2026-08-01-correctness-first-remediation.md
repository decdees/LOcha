# Ocha Correctness-First Remediation Implementation Plan

> **For agentic workers:** Execute with test-driven development and atomic local
> commits. Do not modify `CLAUDE.md`, rewrite historical benchmark artifacts,
> push, or publish.

**Goal:** Restore safety and measurement guarantees before latency optimisation.

**Architecture:** Complete replies cross one final firewall before emission;
local-only MLX inference runs on one dedicated thread; suspicious ASR produces a
curated retry; conversation produces observations rather than FSRS ratings; and
an exchange-identified wire protocol makes client-side timing attributable.

**Tech stack:** Python 3.12, FastAPI, Pipecat, MLX/MLX-LM, SQLite, pytest, vanilla
browser JavaScript, VOICEVOX.

## Tasks

1. Package the curated grammar reference and resolve model repositories from the
   local cache only.
2. Buffer complete model output, finalise it through one firewall path, and add
   late/chunked-sentinel regressions.
3. Remove prompt-cache reuse; add bounded role-tagged history; enforce MLX worker
   ownership and fail-closed lifecycle behaviour.
4. Make suspicious ASR produce an explicit repair path rather than disappearing.
5. Persist vocabulary observations without deriving or applying FSRS ratings.
6. Add exchange IDs, typed audio headers, per-exchange telemetry, and PWA support.
7. Build a v2 known-answer measurement library and harness without modifying the
   old benchmark JSON.
8. Reconcile README, PRD, architecture, tasks, and benchmark status claims.
9. Run focused, full, slow, offline, sustained-session, capture-path, and 50-turn
   iPhone verification as available; never infer a passed gate from missing data.
