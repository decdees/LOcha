# T0.2 — recording checklist

**~15 minutes.** T0.3 is blocked until this is done; T0.4, T0.5, and T0.7 are not.

---

## Before you start

- [ ] Quiet room. Close the window, kill the fan. Room noise raises CER on every model equally, which flattens exactly the differences T0.3 is trying to measure.
- [ ] Use the **built-in MacBook Pro Microphone**. Not AirPods, not the iPhone mic, not an Aggregate Device.
- [ ] Sit ~20 cm from the machine. Consistent distance across all 20 takes matters more than the exact distance.
- [ ] Confirm the mic works before recording 20 files:

```bash
uv run python benchmarks/corpus/record.py --test-mic
```

You want `✓ peak 0.1–0.7, active 40%+`. If you see `peak 0.000`, the wrong input device is selected — run with `--list` and pass `--device N`.

---

## Recording

```bash
uv run python benchmarks/corpus/record.py
```

Each take: the sentence is printed, Enter starts, Enter stops. The script rejects anything that isn't plausibly speech and offers a retry. Progress is saved after every take, so you can stop and resume — already-verified takes are skipped.

### Rule zero — read the `SAY:` line, nothing else

Each prompt now looks like this:

```
── 05 (pure_ja)
   SAY:     nihongo ga sukoshi wakarimasu
   script:  日本語が少しわかります。
```

**Read the `SAY:` line out loud. That's it.** The `script:` line is there for context and is the CER reference — you don't need to read it, and you shouldn't.

This is rule zero because the first corpus was invalidated by exactly this. The old prompt showed the Japanese sentence with its English translation underneath, no romaji. Given kanji you can't read and English you can, reading the English was the only sensible response — and 11 of 20 takes came back in English. That was a tooling failure, not yours. The gloss is gone and the romaji is now the primary line.

### The one rule that matters

**Say it naturally. Do not over-enunciate.**

This is the easiest way to invalidate the whole benchmark. Careful, over-articulated speech is *easier* for every ASR model, so all three score well, the differences between them collapse, and you pick a model on noise. The corpus needs to sound like you actually talk when you're mid-conversation and slightly unsure — because that is the input the product will really get.

Specifically:
- Normal conversational pace. Don't slow down for the microphone.
- Leave the hesitation in. `えーと` and `えっと` are in the corpus deliberately — distilled two-layer decoders drop or loop on fillers, and that is a finding worth having.
- Don't retry a take just because you sounded unsure. Retry only if the script rejects it, or if you fumbled the words.

### The second rule

**If you deviate from the printed sentence, say so.** After each take:

```
Read exactly as written? [Y/n/r=redo]
```

Answer `n` and type what you actually said. CER against a sentence you didn't say measures nothing — it's the one error that silently corrupts every number in T0.3 while looking completely normal.

### On the code-switched half (items 11–20)

Read the English words *in English*, in your normal accent. Don't Japanify them into katakana pronunciation — the point is to test what happens when a Japanese-only ASR meets actual English, which is the reality of beginner speech.

Items 13, 18, and 20 are deliberately awkward English/Japanese hybrids. That is intentional. Read them as written.

---

## When you're done

The script prints `20/20 verified`. Then:

```bash
uv run python benchmarks/corpus/record.py --verify
```

That runs whisper over every take and flags any recorded in the wrong language. Expect `0 of 20`. Then:

```bash
ls benchmarks/corpus/*.wav | wc -l          # expect 20
uv run python -c "
import json,glob
d=json.load(open('benchmarks/corpus/transcripts.json'))
print(sum(1 for u in d['utterances'] if u['verified']), 'verified /', len(d['utterances']))
print(len(glob.glob('benchmarks/corpus/*.wav')), 'wav files')"
```

Tell me when it's done and I'll run T0.3.

---

## What the script will not let you do

Recorded here because these are the failure modes that produce a *plausible-looking* bad benchmark rather than an obvious one:

| Failure | Why it's dangerous | Guard |
|---|---|---|
| Recording the wrong input device | 20 silent WAVs → ~100% CER on every model, which reads as a model finding | Device pinned to the built-in mic; every take level-checked |
| A take that's silent except one pop | Passes a naive loudness check | Presence measured as the fraction of 20 ms frames carrying energy, not overall volume |
| Ground truth ≠ what was said | Corrupts every CER number invisibly | Per-take confirmation prompt, correction written back to `transcripts.json` |
| Reading the wrong line | 11/20 of the first corpus came back in English | Romaji is the only line you're asked to say; the English gloss is never displayed; `--verify` re-checks the language of every take afterwards |
| Take much longer than the sentence | Suggests extra content was spoken | Warns if voiced speech exceeds ~1.9x the expected length for that line |
| Over-enunciating | Flatters all models equally, collapsing the differences being measured | Nothing automatic can catch this. It's on you. |
