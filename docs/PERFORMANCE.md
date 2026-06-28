# Tensotron — Performance Findings & Roadmap

Status as of commit `5856212` ("Async kernel launches + device-side parameter updates").
Measured on an **NVIDIA RTX 4090** (.NET 10, Release).

> **⚡ Update — most of this roadmap is now implemented.** The Tier-0/1/2/3 work below was carried
> out and measured; see [`PERFORMANCE_LOG.md`](PERFORMANCE_LOG.md) for the per-experiment log.
> Headline (default MLP): **72.3 → ~1.4 ms/step (~50×)**; launches 126→36, host uploads 316→3.
> Compute-bound 1024×2048: **46 → 4.4 ms** (cuBLAS). Large-batch alloc cliff 4096×2048: **~3100 → 78 ms**
> (caching allocator). Done: fused optimizer kernels ✅, cached stride/dim uploads ✅, caching device
> allocator ✅ (opt-in via `DisposeGraph`), tiled GEMM ✅, cuBLAS ✅, `Linear` no-transpose-copy ✅.
> **Key conclusion:** every realistic regime is launch/allocation-bound, *not* compute-bound — the
> wins came from removing per-op overhead, not from faster math. Not done (low ROI / future work):
> gradient-buffer recycling, automatic arena, more launch fusion, tree-reduction. All 74 tests green.

## 1. Benchmark

`examples/Tensotron.Examples` → `bench` (`BenchExample.cs`). A fixed MLP trained for a fixed
number of steps, reporting **ms/step**. The net is deliberately tiny, so the number measures
*per-op overhead efficiency*, not arithmetic throughput.

- Model: `Linear(32,128) → ReLU → Linear(128,128) → ReLU → Linear(128,1)`
- Batch 256, MSE loss, Adam(lr=1e-3), 300 steps (30 warmup excluded), final host sync forced.

### Before/after — the async-launch commit

| Commit | What changed | ms/step (3 runs) |
|---|---|---|
| `a99af70` | sync-after-every-launch baseline | ~98.8 (98.32 / 99.24) |
| **`5856212`** | **async launches + device-side param updates** | **~79.1 (79.62 / 78.87 / 78.95)** |

**The async-launch commit cut ~20 ms/step (~20%)**, readings stable to <1.5%. This confirms the
batching direction is correct — but the absolute number is still dominated by overhead.

## 2. Where the 79 ms/step goes

This is a ~4 GFLOP/step workload on a 4090 (≈80 TFLOP/s fp32). If arithmetic were the bottleneck
it would be sub-millisecond. It isn't — it's **per-op launch + allocation overhead**. Rough
per-step accounting (the dominant term is the optimizer, not the math):

| Phase | Kernel launches | Tiny host→device copies | Device allocations |
|---|---|---|---|
| Forward (3× matmul+bias, 2× ReLU, MSE) | ~11 | — | ~11 |
| Backward (matmul dA/dB ×3, bias reduce, ReLU′, accum) | ~20 | — | ~20 |
| **Adam.Step (6 params × ~15 ops)** | **~90** | **~48** | **~140** |
| **Total / step** | **~120** | **~48** | **~170** |

`79 ms / ~120 launches ≈ 0.65 ms per launch-equivalent` — the signature of overhead, not compute.

Concrete root causes, grounded in the code:

1. **Optimizer op-explosion** (`Optimizers.cs`). `Adam.Step()` builds each update from individual
   broadcast tensor ops — `Mul(s.m, Scalar(_b1))`, `Add(...)`, `Square(g)`, `Sqrt(vhat)`, … ≈ 15
   kernels **per parameter per step**, each allocating a fresh result buffer.
2. **Scalar host→device copies.** `Scalar(v)` (`TensorOps.Binary.cs`) is `Tensor.FromShaped(new[]{v}, …)`
   → a `Buffer.CopyFromCPU` of a **1-float array**, every call. Adam does ~8 of these per parameter
   per step (~48/step here), each also a device allocation, all on the critical path.
3. **No caching allocator.** Every op calls `Runtime.Allocate` → `Accelerator.Allocate1D` → a device
   malloc; buffers are freed via GC/`Dispose`. ~170 device allocations/step with no reuse. PyTorch's
   single biggest small-op win is its caching allocator; Tensotron has none yet.
4. **Per-launch stride/dim uploads** (`TensorRuntime.AllocInt`). Every broadcast binary, reduce, and
   strided op uploads its small `dims`/`strides` int arrays to the device **each launch** (parked
   until the next drain). The shapes are identical every step — pure repeated transfer. The code's
   own TODO: *"cache stride buffers / use constant memory."*
5. **Naive GEMM** (`Kernels.MatMul2D`). One thread per output element, scalar `for k` accumulation,
   no shared-memory tiling, and column-strided (uncoalesced) reads of `B`. Fine for correctness;
   it will not scale to real layer sizes, and leaves the 4090's tensor cores idle.
6. **Serial reductions** (`Kernels.ReduceSum`/`Reduce`). One thread per output looping over the
   reduced axis — no tree reduction / shared memory. Hits loss, bias-gradients, norms. (Kernel TODO:
   *"tree-reduce."*)
7. **Mid-graph host stalls.** `LaunchMaxPool2d` does a `Sync()` + `GetAsArray1D()` of the argmax
   every forward — a full stream drain that defeats async batching whenever maxpool is used. (Not in
   this MLP bench, but relevant for conv nets.)
8. **Host-side graph rebuild.** Autograd is rebuilt every forward (`Build` recursion + `HashSet<Tensor>`
   topo per `Backward`), and `Backward`/`ZeroGrad` re-walk it. GC pressure, not GPU — minor here but
   real for tight loops.

## 3. Recommended enhancements (prioritized)

Ordered by expected payoff for this small-op regime first, then by what unlocks large nets.

### Tier 0 — measure first (do before optimizing)
- **Add phase timing to the bench**: report forward / backward / optimizer ms separately, plus a
  launch counter. Confirms the accounting above and turns every change below into a measured delta.
- **Add a size-sweep bench**: batch ∈ {64…4096} and width ∈ {128…2048}, so we see the crossover from
  overhead-bound (tiny) to compute-bound (large) and know which fixes matter for which regime.

### Tier 1 — kill the per-op overhead (biggest win for small/medium nets)
- **Caching device allocator.** Pool freed buffers by size and hand them back from `Runtime.Allocate`;
  return on `Tensor.Dispose`/GC. Eliminates ~170 device mallocs/step. *Highest expected impact,
  contained blast radius (`TensorRuntime` + tensor lifetime).* 
- **Fuse the optimizer step.** One custom kernel per parameter that reads `(p, g, m, v)`, updates `m`,
  `v`, `p` in place, with `β1, β2, ε, lr, bias-corrections` passed **by value** (the `UnaryFwdP`
  by-value-op mechanism already exists). Collapses ~90 launches + ~48 host→device copies + ~140
  allocations/step down to ~6 launches, 0 copies, 0 allocations. *Huge, and self-contained.*
- **Stop allocating `Scalar` tensors.** Add scalar-operand kernel variants (`tensor ⊙ float`) so
  `x * 0.5f`, bias-correction, etc. never round-trip a 1-float buffer through the device.

### Tier 2 — cheaper kernels
- **Cache stride/dim buffers** keyed by signature, or pass low-rank dims as scalar kernel args /
  constant memory (the existing TODO). Removes the per-launch int uploads in (4).
- **Tree-reduction kernel** (shared-memory parallel reduce) for `Sum`/`Mean`/reductions. Speeds loss,
  bias gradients, LayerNorm/BatchNorm.
- **Fused epilogues**: bias-add + activation in one kernel; MSE (`sub → square → mean`) in one kernel.
  Fewer launches *and* fewer intermediates.

### Tier 3 — unlock large-net throughput
- **cuBLAS GEMM** (the README's planned swap). Replace `MatMul2D`/`MatMulBatched` with `cublasSgemm` /
  strided-batched GEMM via ILGPU's CUDA interop. Same stride-based interface; gives orders-of-magnitude
  speedup on real layer sizes and TF32 tensor-core use on Ampere+/Ada essentially for free.
- **Mixed precision (FP16/BF16 + FP32 master weights)** for full tensor-core throughput once cuBLAS
  is in. Larger lift (loss scaling, dtype plumbing — currently float32-only by design).
- **CUDA Graph capture** of the static train step. The architecture is fixed across steps, so capture
  forward+backward+optimizer once and replay — drives launch overhead toward zero. The ideal endgame
  for fixed-arch training; needs raw CUDA (ILGPU doesn't expose graph capture today).
- **Keep MaxPool argmax on device** (device-side gather in backward) to remove the mid-graph `Sync()`.
- **Multi-stream overlap** for independent ops (e.g. per-parameter optimizer updates). Lower priority
  while launch overhead dominates occupancy.

### Summary
The async-launch commit was the right first step and earned ~20%. The next ~order-of-magnitude for
small/medium nets is **not** in the math — it's a **caching allocator + a fused optimizer kernel +
killing the scalar/stride host→device chatter** (Tier 1–2). The **cuBLAS swap** (Tier 3) is what makes
*large* nets competitive and lights up the 4090's tensor cores. Instrument the bench first (Tier 0) so
each step is a measured number, not a guess.
