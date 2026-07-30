"""T0.1 — measured GPU memory bandwidth on this machine.

Why this and not a STREAM-style CPU benchmark: MLX decode runs on the GPU, and
ARCHITECTURE.md §3's entire dense-vs-MoE argument turns on achieved *GPU* read
bandwidth during memory-bound work. Peak spec numbers overstate that by 15-40%.

Two kernels, both memory-bound:
  read   -- mx.sum(a):    reads N bytes
  triad  -- c = a + b:    reads 2N, writes N

MLX is lazily evaluated, so every timed region ends in mx.eval() or the timer
measures graph construction rather than memory traffic.
"""

import argparse
import time

import mlx.core as mx

GIB = 1024**3


def _time(fn, reps: int) -> list[float]:
    fn()  # warmup: first call pays kernel compilation and page-in
    mx.eval(mx.zeros(1))
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def measure(size_gib: float, reps: int) -> dict[str, tuple[float, float]]:
    n = int(size_gib * GIB / 4)  # float32
    nbytes = n * 4
    a = mx.random.uniform(shape=(n,), dtype=mx.float32)
    b = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(a, b)  # materialize before timing, not during

    def read() -> None:
        mx.eval(mx.sum(a))

    def triad() -> None:
        mx.eval(a + b)

    results = {}
    for name, fn, traffic in (("read", read, nbytes), ("triad", triad, 3 * nbytes)):
        ts = _time(fn, reps)
        best = min(ts)
        spread = (max(ts) - best) / best * 100
        results[name] = (traffic / best / 1e9, spread)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size-gib", type=float, default=2.0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--spec-gbs", type=float, default=120.0, help="ARCHITECTURE.md §3 figure")
    args = p.parse_args()

    r = measure(args.size_gib, args.reps)
    print(f"array {args.size_gib} GiB float32, best of {args.reps}\n")
    for name, (gbs, spread) in r.items():
        print(f"  {name:6s} {gbs:7.1f} GB/s   (run spread {spread:.1f}%)")

    achieved = r["read"][0]
    print(f"\n  vs §3 spec {args.spec_gbs:.0f} GB/s: {achieved / args.spec_gbs * 100:.0f}% achieved")

    # §3's argument, recomputed from the measured number instead of the spec.
    print("\n  token ceilings at measured read bandwidth:")
    for label, gb_per_tok in (("27B dense @4bit", 15.0), ("26B-A4B MoE @4bit", 2.2)):
        print(f"    {label:20s} {achieved / gb_per_tok:6.1f} tok/s")


def _self_check() -> None:
    """Bandwidth must not scale with array size; if it does, we're timing overhead."""
    small = measure(0.5, 3)["read"][0]
    large = measure(2.0, 3)["read"][0]
    ratio = large / small
    assert 0.5 < ratio < 2.0, f"bandwidth scales with size ({ratio:.2f}x) -- timing overhead"
    print(f"self-check ok: 0.5 GiB {small:.1f} GB/s vs 2 GiB {large:.1f} GB/s ({ratio:.2f}x)")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
