# Tensotron — Performance Experiment Log

Running log of optimization experiments on the training hot path. Each entry: what was
changed, how it was measured, the result, and whether it was kept. Companion to
[`PERFORMANCE.md`](PERFORMANCE.md) (findings & roadmap).

**Machine:** RTX 4090, .NET 10 SDK, Release builds. **Branch:** `perf/optimize-hot-path`.
**Correctness gate:** the full test suite (`dotnet test`, 69 tests) must stay green after every change.

## Baseline (commit `5856212`)

- **Tests:** 69/69 pass (50 s).
- **Bench** (`bench`): MLP `32→128→128→1`, batch 256, Adam, 270 timed steps.
  - **~79.1 ms/step** (79.62 / 78.87 / 78.95), vs ~98.8 for the prior commit `a99af70`.
- **Hypothesis from static analysis:** dominated by per-op launch + allocation overhead.
  ~120 kernel launches, ~48 host→device scalar copies, ~170 device allocations per step;
  ~75% of launches are the Adam step (6 params × ~15 ops). Naive GEMM; serial reductions;
  `Linear` materializes a transposed weight copy each forward.

---

## Experiments

### E0 — Instrumentation (Tier 0) — *kept*

Added launch/alloc/host-upload counters to `TensorRuntime`; rewrote `BenchExample` to report a
per-step counter breakdown, a serialized fwd/bwd/opt phase attribution, and a `benchsweep`
size sweep (batch × width). No behavior change.

**Ground truth (baseline, default MLP):** 72.3 ms/step = **126 launches, 436 device allocs, 316
host uploads** per step. Phase attribution (serialized): **fwd 6.5 / bwd 6.7 / opt 61.5 ms** →
**the optimizer is ~82% of the work.** Sweep showed launch/alloc counts are *constant* (126/436)
regardless of batch/width — overhead is size-independent and dominates small/medium nets.

### E1 — Fused Adam / AdamW / SGD kernels (Tier 1) — *kept, huge win*

Replaced the high-level optimizer math (≈15 element-wise ops + ≈8 scalar host→device copies *per
parameter per step*) with single in-place fused kernels (`Kernels.AdamStep`, `Kernels.SgdStep`):
one launch per parameter, scalars passed by value, bias-corrections folded host-side, `m`/`v`/
momentum buffers allocated once and mutated in place. One kernel serves Adam (coupled decay) and
AdamW (decoupled-decay factor).

| Metric (default MLP) | Baseline | Fused | Δ |
|---|---|---|---|
| **ms/step** | ~72.3 | **~13.0** | **5.5× faster** |
| launches/step | 126 | 42 | −67% |
| device allocs/step | 436 | 88 | −80% |
| host uploads/step | 316 | 52 | −84% |
| opt phase (serialized) | 61.5 ms | **0.18 ms** | **340×** |

**Correctness:** full suite 69/69 green; `Optimizers_MatchTorch` (torch-golden sgd/adam/adamw/
rmsprop) green. The optimizer is now effectively free; **forward (6.2 ms) + backward (6.3 ms) now
dominate.** Next targets: the `Linear` transpose-copy, per-launch stride-int uploads (the 52
remaining host uploads), and the 88 allocs/step.

### E2 — Content-cached stride/dim buffers (Tier 2) — *kept, huge & surprising win*

`AllocInt` re-uploaded each op's shape/stride int arrays to the device **every launch** (a blocking
`Allocate1D(int[])` host→device copy). These recur identically every step, so they are now cached by
content (structural `int[]` key) and uploaded **once**. Data-dependent index arrays (gather/scatter
indices, pool argmax) stay on the per-call parked-and-freed path so the cache can't grow unbounded.

| Metric (default MLP) | After E1 | After E2 | Δ |
|---|---|---|---|
| **ms/step** | ~13.0 | **~2.0** (1.6–2.4) | **~6× faster** |
| host uploads/step | 52 | **3** | −94% |
| device allocs/step | 88 | 39 | −56% |
| launches/step | 42 | 42 | — |
| **test suite wall-clock** | ~48 s | **~3 s** | **~16×** |

The ~49 eliminated host→device transfers/step were **blocking the async stream** — removing them gave
another 5.5×, and incidentally sped the whole test suite ~16×. **Cumulative: 72.3 → ~2.0 ms/step,
≈35×.** Correctness: 69/69 green. Variance is now GPU-clock jitter (absolute times are sub-millisecond
per phase). Remaining per step: 42 launches, 39 allocs, 3 uploads — now genuinely compute/launch bound.

**Size sweep after E2** (inDim 64, depth 2; ms/step). Launch/alloc counts are constant at 42/39
across *all* sizes — overhead is size-independent, so small nets sit on a ~1.5 ms launch-overhead
floor while large nets are dominated by the naive GEMM:

| batch \ width | 128 | 512 | 2048 |
|---|---|---|---|
| 64   | 1.56 | 1.68 | 4.54 |
| 256  | 1.48 | 2.35 | 16.08 |
| 1024 | 2.49 | 7.11 | **46.24** |
| 4096 | 8.09 | 15.58 | (naive GEMM ≳150) |

**Conclusion:** two distinct regimes. Small/medium nets → launch-bound (≈1.5 ms floor, attack launch
*count*). Large nets → **the naive one-thread-per-output GEMM is the entire bottleneck** (attack the
matmul kernel). Next experiments target both: a faster GEMM (tiled / cuBLAS) and launch-count cuts.

### E3 — Tiled shared-memory GEMM (Tier 3) — *kept*

Added `Kernels.MatMul2DTiled`: a 16×16 tiled SGEMM that stages A/B tiles into shared memory (each
loaded element reused 16×, vs the naive kernel re-reading global memory per MAC). Keeps the **exact
explicit-stride contract** of `MatMul2D`, so transposes stay stride-swaps and the autograd backward
is unchanged. Dispatched from `LaunchMatMul` only when M,N,K ≥ 64 (tiling's masked threads aren't
worth it for small/skinny products — those keep the naive kernel). New test `TiledMatMulTests`
verifies forward **and** both transposed-stride gradients (dA=W·Bᵀ, dB=Aᵀ·W) against a CPU reference,
with non-multiple-of-16 dims to exercise boundary masking.

Sweep, naive → tiled ms/step (compute-bound rows):

| batch × width | naive | tiled | speedup |
|---|---|---|---|
| 1024 × 512 | 7.11 | 4.43 | 1.6× |
| 4096 × 512 | 15.58 | 9.90 | 1.6× |
| 256 × 2048 | 16.08 | 9.00 | 1.8× |
| **1024 × 2048** | **46.24** | **14.79** | **3.1×** |

Small/launch-bound rows are unchanged (within jitter). **Correctness:** 70/70 green (new test added).
A basic tile (no register blocking) reaches ~2–3× of naive; the open question is whether cuBLAS
(tensor cores / TF32) does materially better — tested next (E4).
