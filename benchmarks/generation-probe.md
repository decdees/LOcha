# T0.5 — generation probe

**Date:** 30 July 2026 · base M4
**Replaces** the original grammar-explanation probe. FR-5 firewalls the model's explanations, so explanation quality cannot change any decision. What the product depends on is whether the model *produces* acceptable Japanese under constraint — and all four checks below are decidable without Japanese grammar expertise.

---

## Results

Prompt = ARCHITECTURE §7.1's template verbatim, frozen 40-item `KNOWN` list, `TARGET = 食べる`. 15 conversational turns + 5 grammar questions. Greedy decoding.

| model | vocabulary | register | length | sentinel |
|---|---|---|---|---|
| gemma-4-26b-a4b | 14/15 | **9/15** | 15/15 | **5/5** |
| gemma-4-26b-a4b *+ register line* | **15/15** | **14/15** | 15/15 | **5/5** |
| Qwen3.5-9B | **15/15** | **15/15** | **15/15** | **5/5** |

**The firewall trigger is reliable.** Both models emitted a clean, bare `[GRAMMAR_QUERY]` on 5/5 grammar questions, with no explanatory text alongside. That is the single most load-bearing result here — T1.5's firewall depends on it.

---

## ARCHITECTURE §7.1's template is missing a register constraint

gemma failed register on 6/15 turns, always the same shape:

> `いいですね。あなたは今日、何を食べる？` — polite `いいですね` followed by plain-form `食べる`

For an absolute beginner this is actively harmful: it models inconsistent politeness to someone who cannot yet detect it.

**But it is a prompt bug, not a model defect.** §7.1's template constrains vocabulary, target items, length and grammar level — and says nothing about politeness register. Adding one line:

```
REGISTER: Always use polite です/ます form. Never mix polite and plain forms.
```

takes gemma from **9/15 to 14/15** on register and **14/15 to 15/15** on vocabulary. **This line is required in T1.6's Context Builder.**

---

## The naturalness blind spot, demonstrated rather than hypothesised

The four checks verify constraint *compliance*, not that the Japanese is *correct*. That limitation was stated up front; this run produced a concrete instance of it.

With the register line added, gemma produced:

> `今日は何を食べるですか？`
> `そこで何を食べる（eat）ですか？`

`食べる` + `ですか` is a conjugation error — the polite interrogative is `食べますか`. **Both replies scored as PASS on register**, because the checker sees a sentence ending in `ですか` and correctly classifies it as polite. It has no way to know the verb form preceding it is wrong.

Qwen3.5-9B's replies over the same turns did not show this (`何を食べましたか？`, `駅には何を食べに行きますか？`).

This is exactly the failure mode the product cannot afford: a tutor teaching a beginner a wrong conjugation, where the learner is by definition unable to catch it. The grammar firewall does **not** protect against it — FR-5 covers *explanations*, and this is *production*.

*Confidence note: `食べるですか` is a basic conjugation error and I state it with high confidence, but I am not a Japanese-competent reviewer. It should be verified by one, along with the retained reply samples.*

---

## Does this change the LLM choice?

**No — gemma-4-26b-a4b stands, but conditionally.**

| | gemma-4-26b-a4b | Qwen3.5-9B |
|---|---|---|
| compliance (with register line) | 49/50 | **50/50** |
| observed grammatical errors | `食べるですか` | none observed |
| throughput | **39.4 tok/s** | 20.5 tok/s |

Qwen3.5-9B is the better-behaved model on this probe and produced cleaner Japanese. gemma is **1.9× faster**, and T0.3/T0.4 leave the latency budget already 2150 ms over a 1200 ms target — halving generation throughput is not affordable.

The decision rests on gemma's compliance being fixable in the prompt (demonstrated) while its throughput advantage is not recoverable any other way. **If a native reviewer finds gemma's Japanese materially worse than Qwen's, that reverses this** — the tutor's output quality outranks its speed, because a fast tutor teaching wrong forms is worse than a slow one teaching right ones. Recorded as the trigger for revisiting.

---

## Harness bugs found — the results before fixing them were 0/15 and 0/5

Both models initially scored **0/15 vocabulary and 0/5 sentinel**. Two models failing identically is a harness signature, not a finding.

1. **Both candidates are reasoning models.** By default they emit a thinking channel — `<|channel>thought` for gemma, `Thinking Process:` for Qwen — before the reply, and `max_tokens=120` truncated the output before any actual answer. Fixed with `enable_thinking=False` in `apply_chat_template`.

   **This matters far beyond the probe.** With thinking enabled, gemma answered a grammar question with the sentinel *plus 399 characters of grammar explanation*. That is precisely what FR-5 forbids. Two consequences for Phase 1:
   - `enable_thinking=False` is a **hard requirement** of the LLM service, not a tuning knob.
   - **T1.5's firewall must not simply test `"[GRAMMAR_QUERY]" in output`.** It must assert the sentinel is the *entire* payload, or a reasoning trace containing both the sentinel and an explanation would pass the check while leaking exactly what the firewall exists to block.

2. **The length checker counted English glosses as sentences.** FR-3 explicitly permits a gloss for a new word, so `駅に行きますか。(Do you go to the station?)` is one compliant sentence — the checker scored it 3 and failed it. That understated Qwen3.5-9B at 9/15 when it in fact passes 15/15. Fixed by stripping glosses before counting; two fixtures added.

---

## Limitations

- **Naturalness is unmeasured**, and the `食べるですか` case above shows the gap is real, not theoretical. All 40 replies are retained in `generation-probe-replies.json` for review by a Japanese-competent reader.
- **n = 15 conversational + 5 grammar, one prompt configuration, greedy decoding.** No sampling variance, no prompt-sensitivity sweep. The register finding is a single A/B.
- **`KNOWN` is a synthetic 40-item list**, not real FSRS state. Vocabulary adherence against a realistic evolving pool is untested.
- **The vocabulary checker is lenient by construction** — it ignores single-character tokens to avoid false positives on particles and inflection fragments, so a genuinely out-of-vocabulary single-kanji word would pass.

## Reproduce

```bash
uv run python benchmarks/generation_probe.py --self-check   # calibrate the 4 checkers
uv run python benchmarks/generation_probe.py
```
