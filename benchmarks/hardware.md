# T0.1 — Hardware identification and measured memory bandwidth

**Date:** 30 July 2026
**Acceptance:** states the chip variant and measured bandwidth.

---

## Chip variant

| Property | Value | Source |
|---|---|---|
| Chip | **Apple M4 (base)** | `sysctl -n machdep.cpu.brand_string` → `Apple M4` |
| CPU cores | 10 — 4 efficiency + 6 performance | `hw.perflevel0.logicalcpu` = 4, `hw.perflevel1.logicalcpu` = 6 |
| GPU cores | 10 | `system_profiler SPDisplaysDataType` |
| Unified memory | 32 GB (34,359,738,368 bytes) | `hw.memsize` |
| macOS | 26.5.2 (build 25F84) | `sw_vers` |

**This is the base M4, not M4 Pro or M4 Max.** `ARCHITECTURE.md` §9 Risk #1 — *"machine is M4 Pro-class bandwidth"* — is **falsified**. Consequences, applied:

- §3's low-bandwidth branch is the live one: **MoE is mandatory**, dense 27B is not viable.
- §3.1 shortlist option 4 (Qwen 3.5 27B dense, "M4 Pro only") is dead.
- T0.4's conditional third candidate `qwen3.5:27b` is **skipped** by that task's own stated condition.

---

## Measured bandwidth

Measured with `benchmarks/bandwidth.py` — MLX, GPU-side, 2 GiB float32 array, best-of-5 per invocation.

**Why GPU-side and memory-bound specifically:** MLX decode runs on the GPU, and §3's whole dense-vs-MoE argument turns on achieved read bandwidth during memory-bound work. A CPU STREAM benchmark or a peak-synthetic figure would answer a different question. MLX is lazily evaluated, so each timed region ends in `mx.eval()` — otherwise the timer measures graph construction, not memory traffic.

| Kernel | Traffic | Measured |
|---|---|---|
| `read` — `mx.sum(a)` | reads N | **103.2 GB/s** (median of 5 invocations) |
| `triad` — `a + b` | reads 2N, writes N | ~97–98 GB/s |

**Read bandwidth across 5 consecutive invocations:** 102.5, 102.6, 103.2, 103.3, 103.4 GB/s — spread **±0.4%**, well inside the ≤5% stability bar.

An earlier set taken under heavier concurrent load gave 97.6–106.6 GB/s (±4.5%), with one within-run spread outlier of 127% on `triad`. Reported figures are best-of-N, so the outlier does not propagate, but the wider range is recorded rather than hidden: **treat 103 GB/s as ±5%, not as three significant figures.**

### Measured vs spec

**103.2 / 120 = 86% of the §3 spec figure.** That is the high end of the expected 60–85% window for memory-bound GPU work on Apple Silicon — §3's ~120 GB/s figure is a fair basis, marginally optimistic in absolute terms.

### Token ceilings, recomputed from the measured number

§3 derives its ceilings from the spec. Recomputed from 103.2 GB/s:

| Model | Bytes/token | Theoretical ceiling | §3's stated estimate |
|---|---|---|---|
| 27B dense @ 4-bit | ~15 GB | **6.9 tok/s** | ~8 theoretical, 5–6 realistic |
| 26B-A4B MoE @ 4-bit | ~2.2 GB | **46.9 tok/s** | 40–50 theoretical, 25–35 realistic |

§3's dense figure was slightly pessimistic in the realistic column; its conclusion is unaffected. A 40-token reply at 6.9 tok/s is ~5.8 s of generation alone — **dead on arrival for voice**, as §3 argued.

**The MoE number is a ceiling, not a prediction.** The 2.2 GB/token figure assumes MoE weight reads stream as efficiently as dense ones. They do not: per-token expert routing makes reads scattered rather than contiguous, so achieved bandwidth on MoE decode is expected to fall below the dense-equivalent ratio. T0.4 measures actual tok/s and derives effective bandwidth (`bytes_per_token × tok/s`) against this 103.2 GB/s figure. If the gap is large, §3's headline argument needs softening even though its conclusion — MoE over dense — survives either way.

---

## Measurement conditions

Recorded because an unrecorded baseline makes these numbers unreproducible.

Concurrent resident processes at measurement time (top by RSS): a container-runtime Linux VM (~1.6 GB, `Virtualization.framework`), WebKit content processes (~1.2 GB combined), a desktop application (~0.6 GB), an endpoint-security agent (~0.5 GB). Swap: **0.00 MB used, 0 swapins/swapouts** throughout.

The container runtime being resident matters little for a bandwidth measurement but matters a great deal for T0.7's memory-contention test — its VM reserves **8.32 GB**, which exceeds §4's entire 5.4 GB headroom. **Shut the container runtime down before T0.4 and T0.7.**

---

## Verification

- `bandwidth.py --self-check` asserts bandwidth does not scale with array size: 0.5 GiB → 103.1 GB/s, 2 GiB → 103.2 GB/s, ratio **1.00×**. Had the timer been measuring graph construction rather than memory traffic, the small array would have appeared far faster.
- Cross-invocation spread ±0.4% over 5 runs (bar: ≤5%).

## Not measured, and why

- **CPU-side STREAM bandwidth.** Would need a C harness and answers a question §3 does not ask — inference runs on the GPU.
- **Peak synthetic bandwidth via a hand-written Metal shader.** Would report a number closer to spec and further from what decode actually achieves.
