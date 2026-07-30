"""T0.4 follow-up 2 — does prompt-cache reuse rescue the latency budget?

llm_context.py measured TTFT of 1.6 s at 256 tokens and 32.6 s at 8k, against
ARCHITECTURE §5.1's 200 ms budget. If every turn re-prefills the whole
conversation, the voice loop's 1.2 s target is unreachable and §5.1 is fiction.

The mitigation is KV-cache reuse: keep the cache across turns so turn N only
prefills the NEW tokens rather than the whole prefix. This measures whether that
works and how much it buys, simulating a 10-turn conversation both ways.

If reuse works, it is an architectural REQUIREMENT for Phase 2, not an
optimisation -- and it belongs in DECISION.md as such.
"""

from __future__ import annotations

import json
import pathlib
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

OUT = pathlib.Path(__file__).parent / "llm-promptcache.json"
REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"

SYSTEM = "あなたは日本語の会話パートナーです。1、2文で短く返事してください。" * 12  # ~500 tok
TURNS = [
    "こんにちは。",
    "週末は何をしましたか。",
    "私は映画を見ました。",
    "その映画は面白かったですか。",
    "はい、とても面白かったです。",
    "どんな映画が好きですか。",
    "アクション映画が好きです。",
    "そうですか。他には何をしましたか。",
    "友達と食事をしました。",
    "いいですね。何を食べましたか。",
]


def first_token_s(model, tok, prompt, cache=None) -> tuple[float, str]:
    t = time.perf_counter()
    ttft, out = None, []
    for r in stream_generate(model, tok, prompt, max_tokens=24, prompt_cache=cache):
        if ttft is None:
            ttft = time.perf_counter() - t
        out.append(r.text)
    return ttft or 0.0, "".join(out)


def main() -> None:
    model, tok = load(REPO)
    print(f"system prompt: {len(tok.encode(SYSTEM))} tokens\n")

    # A) naive: rebuild and re-prefill the entire conversation every turn
    print("A) no cache reuse -- re-prefill whole conversation each turn")
    naive = []
    convo = SYSTEM
    for i, u in enumerate(TURNS, 1):
        convo += f"\nユーザー: {u}\nチューター: "
        ttft, reply = first_token_s(model, tok, convo)
        n = len(tok.encode(convo))
        naive.append({"turn": i, "ctx": n, "ttft_s": round(ttft, 3)})
        convo += reply.strip()
        print(f"   turn {i:2}  ctx {n:5} tok   TTFT {ttft:6.2f}s")

    # B) cache reuse: only the new tokens are prefilled each turn
    print("\nB) prompt-cache reuse -- only new tokens prefilled")
    cache = make_prompt_cache(model)
    cached = []
    _, _ = first_token_s(model, tok, SYSTEM, cache)  # prime with the system prompt
    for i, u in enumerate(TURNS, 1):
        piece = f"\nユーザー: {u}\nチューター: "
        ttft, reply = first_token_s(model, tok, piece, cache)
        cached.append({"turn": i, "new_tok": len(tok.encode(piece)), "ttft_s": round(ttft, 3)})
        print(f"   turn {i:2}  new {len(tok.encode(piece)):3} tok   TTFT {ttft:6.2f}s")

    OUT.write_text(json.dumps({"naive": naive, "cached": cached}, indent=2) + "\n")

    nm = sorted(r["ttft_s"] for r in naive)
    cm = sorted(r["ttft_s"] for r in cached)
    print("\n" + "=" * 62)
    print(f"{'':22}{'p50':>8}{'p95':>8}{'worst':>8}")
    print(f"{'no reuse':22}{nm[len(nm)//2]:8.2f}{nm[int(len(nm)*.95)-1]:8.2f}{nm[-1]:8.2f}")
    print(f"{'cache reuse':22}{cm[len(cm)//2]:8.2f}{cm[int(len(cm)*.95)-1]:8.2f}{cm[-1]:8.2f}")
    print(f"\n§5.1 budgets 200 ms for LLM TTFT.")
    print(f"speedup at p50: {nm[len(nm)//2]/max(cm[len(cm)//2],1e-6):.1f}x")


if __name__ == "__main__":
    main()
