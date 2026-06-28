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

### E4 — `Linear` without the transposed-weight copy (launch-count cut) — *kept*

`Linear.Forward` did `MatMul(x, Weight.T())`, and `T()` **materialized a contiguous transposed copy
of the weight every forward** (a `StridedCopy` launch + buffer) plus a `Permute` every backward.
Added `TensorOps.MatMulNT(a,b) = a·bᵀ` that reads both operands through strides (no copy), with a
stride-only backward (`da = g·b`, `db = gᵀ·a`); `Linear` now uses it. New `MatMulNtOpTests` checks
forward + both grads against the old `MatMul(x, W.T())` path at a small (naive) and large (tiled) size.

| Metric (default MLP) | After E2/E3 | After E4 |
|---|---|---|
| **ms/step** | ~2.0 | **~1.4** |
| launches/step | 42 | **36** |
| device allocs/step | 39 | 33 |

Removed exactly the 6 expected launches (3 forward copies + 3 backward permutes) and a weight-sized
copy per layer per direction (more bandwidth saved as width grows). **72/72 tests green.**

### E5 — cuBLAS SGEMM on CUDA (Tier 3) — *kept, decisive on large nets*

Wired ILGPU.Algorithms' `CuBlas` (vendor-tuned SGEMM) into `LaunchMatMul` for the compute-bound
regime on the CUDA backend; the tiled kernel stays as the CPU path, naive for small/skinny products.
The hard part is that cuBLAS is **column-major** while our buffers are row-major and operands can be
stride-transposed (backward) — solved with the operand-swap identity (compute Cᵀ=Bᵀ·Aᵀ; cuBLAS-A=our B,
cuBLAS-B=our A; m=N,n=M,k=K) and a per-operand trans-flag/leading-dim derived from the strides
(`contraction-contiguous ⇒ NonTranspose`). cuBLAS shares the accelerator default stream, so it
interleaves in-order with our kernels under the same single `Sync`. **Verified** by the existing
large-matmul tests (forward + dA + dB exercise all three stride patterns vs a CPU reference) — passed
on the first try.

Full sweep, **naive → tiled → cuBLAS** ms/step:

| batch × width | naive | tiled | cuBLAS | cuBLAS vs naive |
|---|---|---|---|---|
| 1024 × 512 | 7.11 | 4.43 | 2.67 | 2.7× |
| 4096 × 512 | 15.58 | 9.90 | 7.26 | 2.1× |
| 256 × 2048 | 16.08 | 9.00 | 2.77 | 5.8× |
| **1024 × 2048** | 46.24 | 14.79 | **4.43** | **10.4×** |

Small nets also improved (launch floor now **~0.7–0.9 ms**; 64×512 = 0.72). **72/72 tests green.**

**Anomaly → next experiment.** `4096 × 2048` = **196.9 ms** — wildly super-linear (4× the batch of the
4.43 ms row → 44× the time). cuBLAS is fast here; the cost is the **allocator**: 4096×2048 activations
are ~33 MB each, ×33 allocs/step ≈ 1 GB of `cudaMalloc`/free churn per step. This is the remaining
bottleneck for large-activation training and motivates E6 (caching allocator).

### E6 — Caching device allocator (Tier 1) — *kept (opt-in), confirms the cliff*

Added a size-bucketed device-buffer pool to `TensorRuntime`. **Reuse is always on** (an empty pool
just falls through to a real allocation); the pool is **fed only by explicit `Tensor.Dispose()`**, so
default non-disposing code is byte-for-byte unchanged. `Tensor.DisposeGraph()` recycles a step's
interior forward-activation buffers back to the pool — the deterministic "free the graph" to call
after backward+step. It stays **opt-in** because Tensotron *retains* graphs by default (supports
repeated backward), so auto-freeing would break those semantics — same reason PyTorch frees only when
not retaining. Capped at ~1 GB retained.

**Allocator probe** (`benchpool`, heavy configs, pool-off = no recycle vs pool-on = DisposeGraph/step):

| config | pool-off | pool-on | speedup |
|---|---|---|---|
| 1024 × 2048 | 7.3 ms (33 alloc/step) | 6.5 ms (22 alloc/step) | 1.1× |
| **4096 × 2048** | **~200–3100 ms** (33 alloc/step) | **78 ms** (22 alloc/step) | **~40×** |

The unpooled 4096×2048 figure is *catastrophic and highly variable* (196 ms in the E5 sweep, 3113 ms
here) — the signature of GC-pressure / `cudaMalloc` thrashing once per-step churn (~1 GB) outpaces the
GC. Pooling makes it ≈linear. At 1024×2048 the GC keeps up, so the win is small — the allocator matters
specifically for **large activations / large batch**. (Gradient buffers are *not* recycled — they can
alias across inputs — so pool-on still shows 22 allocs/step; recycling those too needs ref-counting,
left as future work.) **Correctness:** new `AllocatorPoolTests` trains the same net with and without
recycling and requires **bit-identical** parameters; 74/74 tests green.

---

## Conclusions

**Headline (default MLP `32→128→128→1`, batch 256, RTX 4090):**

| | baseline `5856212` | final | factor |
|---|---|---|---|
| **ms/step** | 72.3 | **~1.4** | **~50×** |
| kernel launches / step | 126 | 36 | −71% |
| device allocs / step | 436 | 33 | −92% |
| host uploads / step | 316 | 3 | −99% |
| optimizer phase | 61.5 ms | 0.05 ms | ~1000× |
| test-suite wall-clock | ~50 s | ~2 s | ~25× |

**Compute-bound regime (1024×2048):** 46.2 ms (naive) → **4.4 ms** (cuBLAS), 10.4×.
**Large-activation cliff (4096×2048):** ~200–3100 ms (alloc-bound) → **78 ms** with the pool, ~40×.

**What mattered, in order of impact:**
1. **Fused optimizer kernels (E1)** — the single biggest win. The optimizer was 82% of the work
   (~15 ops + ~8 scalar uploads per parameter per step); fusing it to one in-place kernel/param made
   it ~free. 5.5×.
2. **Caching the per-launch stride/dim uploads (E2)** — those tiny `Allocate1D(int[])` host→device
   copies were *blocking the async stream*. Caching them (upload once) gave another 6× and sped the
   test suite ~16× as a side effect.
3. **Caching device allocator (E6)** — converts the large-batch `cudaMalloc` thrashing cliff (40×)
   back to ≈linear; essential past ~1 GB/step of activation churn.
4. **cuBLAS (E5)** then **tiled GEMM (E3)** — 10.4× / 3.1× on large nets; cuBLAS wins where available.
5. **`Linear` transpose-copy elimination (E4)** — a clean launch-count + bandwidth cut.

**The big conclusion — it was never the math.** Even after every change, the realistic regimes are
**launch- and allocation-bound, not compute-bound**: at 1024×2048 the effective rate is only ~3
TFLOP/s on a ~40-TFLOP/s GPU, because ~36 launches + ~33 allocs/step dominate, not the GEMM. So
further GEMM tuning (TF32/tensor cores) has low ROI here *and* would break FP32 parity — deliberately
not pursued. The compounding wins came from **removing per-op overhead** (fused optimizer, cached
uploads, pooled allocations), exactly where the instrumentation pointed.

**Correctness throughout:** every change gated on the full suite (69→74 tests, +5 new: tiled-GEMM,
MatMulNT, allocator-transparency); torch-golden optimizer parity preserved; XOR/spiral/regression
demos still converge (loss→0 / 99.7% / MSE 0.0029).

**Remaining future work (diminishing returns, documented not done):**
- **Recycle gradient buffers** too (they can alias across inputs → needs ref-counting). Would push the
  4096×2048 case below 78 ms by removing the residual 22 allocs/step.
- **Automatic graph-free / arena** so large-activation training gets the pool without an explicit
  `DisposeGraph()` call — needs a transient-vs-persistent lifetime model (params/optimizer state must
  survive), a real design change beyond perf tuning.
- **Fuse more launches** for the small-net floor (bias+activation epilogue, fused MSE) — each shaves a
  few of the 36 launches; modest, the floor is already ~1 ms.
- **Tree-reduction kernel** — only matters for very large reductions, not seen as a bottleneck here.
