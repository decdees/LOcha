# Ocha Correctness-First Remediation Design

## Goal

Restore Ocha's safety, Japanese-learning, and measurement guarantees before
optimising latency. Accuracy wins every trade-off in this cycle.

## Decisions

- Quarantine the complete LLM reply and apply the grammar firewall before any
  tutor text or audio is emitted.
- Remove mutable prompt-cache reuse. Conversation history is explicit,
  role-tagged, bounded to four exchanges, and trimmed to the 2,048-token limit.
- Load MLX models only from the local Hugging Face cache, and make the dedicated
  inference worker the sole owner of every MLX call, including status queries.
- Replace silent ASR drops with a visible and audible deterministic retry. A
  rejected transcript reaches neither the tutor nor persistence or scheduling.
- Record free-conversation vocabulary mentions as observations. They are not
  evidence of grammatical correctness and therefore do not mutate FSRS.
- Give every voice exchange a UUID and every outbound audio chunk an explicit
  kind so latency and feedback measurements never infer provenance from order.
- Treat the previous 2.05-second result as invalid. G1a and G1b remain unproven
  until one attribution-clean 50-turn iPhone run satisfies the original bounds.

## Boundaries

Native iOS, Smart Turn integration, pitch scoring, deterministic vocabulary
enforcement, and graded drills are deferred. Runtime has no network fallback;
model provisioning is an explicit setup operation. Historical benchmark files
remain immutable.

## Acceptance

The firewall blocks a contiguous `GRAMMAR_QUERY` anywhere in complete model
output; grammar data ships in the wheel; suspicious ASR fails visibly; free
conversation leaves FSRS unchanged; every MLX call uses one worker thread; and
the v2 instrument passes known-answer tests before it may report a product gate.
