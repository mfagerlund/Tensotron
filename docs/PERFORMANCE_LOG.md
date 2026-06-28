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
