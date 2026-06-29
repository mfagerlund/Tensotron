# CPU SIMD Backend — Progress Log

Live log of the CPU-SIMD backend build (plan: `CPU_SIMD_BACKEND_PLAN.md`). Newest status at top of
each phase. "green" = `tools/run-tests.ps1` (torch-parity + fast smoke) passes.

## Status snapshot

| Phase | What | State |
|---|---|---|
| 0 | Storage + runtime abstraction (no behavior change) | DONE — 81/81 green on ILGPU |
| 1 | CpuSimd elementwise/reduce/movement (scalar) | DONE — see below |
| 2 | Managed matmul + Linear + Adam/SGD; train MLP | DONE — see below |
| 3 | conv/pool/norm/index/losses port | DONE — 66/66 parity on simd |
| 4 | SIMD (Vector<float>) matmul + benchmark | DONE — beats hand-scalar at batch≥8 |
| 5 | Inference latency vs baseline + docs | DONE — see Conclusions |

## Design decisions

- **`Tensor.Buffer` keeps its name, changes type** to abstract `TensorStorage` (`DeviceStorage`
  wraps ILGPU `MemoryBuffer1D`; `HostStorage` wraps `float[]`). Keeps all 51 `.Buffer` op-sites
  textually unchanged — they just pass a `TensorStorage` now.
- **`TensorRuntime` becomes an abstract base**; current ILGPU code → `IlgpuRuntime`; new
  `CpuSimdRuntime`. `Instance` picks one from `RequestedBackend`. Single-backend-per-process holds,
  so `Launch*` impls downcast storage to the concrete type they own (safe).
- **New backend value `TensorBackend.CpuSimd`** (env `TENSOTRON_BACKEND=simd`), distinct from the
  existing `Cpu` (= ILGPU scalar CPUAccelerator, kept for debug/reference).

## Log

### Phase 0 — abstraction (DONE)
- `TensorStorage` (public abstract) + `DeviceStorage`/`HostStorage` (internal). `Tensor.Buffer`
  retyped; host transfers route through `ToHost`/`CopyFromHost`. **0 op-site changes** (the 51
  `.Buffer` references compile unchanged).
- `TensorRuntime` split into an abstract base (Instance factory, counters, diagnostics, abstract
  `Launch*`) + `IlgpuRuntime` (the original code, unwrapping `TensorStorage`→`DeviceStorage` via a
  one-line `B(s)` helper per method). `Device.cs` `Accelerators.Active()` no longer reads
  `.Accelerator` (derives type from `IsGpu`). MaxPool argmax handle abstracted to `object?`.
- **Verified: 81/81 green on ILGPU** (no behavior change).

### Phases 1–3 — managed kernels + full op parity (DONE)
- `CpuKernels.cs`: managed twin of every `Kernels.*` method, **reusing the same `IOp` structs**
  (so XMath transcendentals match the GPU path bit-for-bit) — the parity gate held automatically.
  GPU atomics (`Atomic.Add`) become plain `+=` (single-threaded → deterministic).
- `CpuSimdRuntime.cs`: managed backend, `HostStorage` = `new float[]` (zero-init), `Sync` no-op,
  no ILGPU in the compute loop.
- **Verified: 66/66 op-parity tests green on `TENSOTRON_BACKEND=simd`** — full surface (binary,
  unary, reduce, matmul, conv, pool, norm, index, losses, movement, structural, optimizers,
  end-to-end). (12 ILGPU-specific tests excluded: trace/replay, alloc-pool, device-probe,
  tiled-GEMM, cuBLAS-batched — none apply to the managed backend.)

### Phase 5 (partial) — the conclusion benchmark
Inference latency, policy net `8→64→64→2` tanh, NoGrad, vs an identical hand-scalar C# forward
(`examples … inference`). The ILGPU "CPU" backend is the thing we're replacing:

| backend | batch=1 | batch=8 | vs hand-scalar |
|---|---|---|---|
| ILGPU CPUAccelerator (scalar) | 6778.9 µs | 19632.6 µs | **794× / 605×** |
| CpuSimd (managed scalar) | **12.0 µs** | 46.4 µs | **1.5× / 1.5×** (b64: 1.3×) |

**Core conclusion:** the managed backend takes batch=1 inference from **6778.9 µs → 12.0 µs
(~565× faster than ILGPU-CPU)**, landing within **1.5×** of hand-written scalar C# — *with no SIMD
yet*. The ILGPU "CPU" cost was entirely per-op dispatch, not arithmetic. The residual 1.5× is fixed
per-op overhead (Tensor alloc + managed kernel call + output marshaling), which dominates at
batch=1, so SIMD helps the math-bound regimes (larger batch / training), not batch=1.

### Phase 4 — SIMD (`Vector<float>`) on the matmul
Vectorized the FLOP bulk in `CpuKernels.MatMul2D` (standalone — no `IOp` interface change, so the
GPU kernels are untouched), plus `AddInto`:
- **Dot path** (`aKs==bKs==1`, the Linear-forward / MatMulNT layout): each output is a vectorized
  K-reduction, 4 independent accumulators to hide FMA latency.
- **AXPY path** (`bNs==1`, the transposed backward `dA`/`dB` layout): output-stationary — broadcast
  `A[m,k]` and FMA the contiguous `B[k,:]` row into `C[m,:]`, vectorizing over N. Without it the
  backward matmuls hit the scalar fallback and dominated (bwd 11.9→1.8 ms; whole step 13.9→4.3 ms).
- Faithfulness note: dropped an `av==0 → skip` AXPY micro-opt — it turned `0·NaN` into `0`, diverging
  from torch/the GPU path (correctness over the sparse-gradient speedup; bwd 1.8→2.4 ms).

## Conclusions

**The thesis held, decisively.** ILGPU's "CPU" cost was per-op device dispatch, not arithmetic; a
managed `float[]` backend with no ILGPU in the loop erases it, and SIMD on the matmul makes it
competitive with — and at batch≥8 faster than — hand-written scalar C#.

**Inference** (policy `8→64→64→2` tanh, NoGrad, full marshal-in/read-out path), µs/forward:

| batch | ILGPU-CPU | CpuSimd (SIMD) | hand-scalar C# | CpuSimd vs ILGPU-CPU | vs hand-scalar |
|---|---|---|---|---|---|
| 1  | 6778.9 | **10.5** | 8.2 | **~645×** | 1.3× |
| 8  | 19632.6 | **29.4** | 31.7 | **~670×** | **0.9× (faster)** |
| 64 | — | **231.9** | 255.5 | — | **0.9× (faster)** |

**Training** (MLP `32→128→128→1`, batch 256, fwd+bwd+Adam), ms/step:

| backend | ms/step | fwd / bwd / opt |
|---|---|---|
| CUDA (RTX 4090) | 1.06 | 0.33 / 0.52 / 0.05 |
| CpuSimd | 4.79 | 2.19 / 2.43 / 0.06 |

CPU training is ~4.5× the GPU at this (deliberately compute-heavy) size — expected; the CPU backend
targets small-net inference + training, not large-batch. Op counts are identical to ILGPU
(31 launches/step), confirming the autograd graph is shared and only the kernels changed.

**What worked**
- Reusing the `IOp` structs verbatim in the managed kernels → transcendentals (Exp/Tanh/Gelu via
  XMath) matched the GPU path automatically; the full torch-fixture suite passed on the new backend
  with **zero new test authoring** (66 op-parity + 3 showcase smoke incl. real PPO/CNN training).
- Single-backend-per-process meant the storage/runtime split needed no per-tensor device tag.
- SIMD as two standalone matmul paths (dot + AXPY) — big win, no GPU risk, no `IOp` churn.

**What didn't move the needle / known limits (documented, not bugs)**
- Elementwise unary/binary (ReLU, bias-add) are still scalar — vectorizing them needs the
  `Vector<float>`-on-`IOp` seam (transcendentals don't vectorize cleanly; would need a
  `Vectorizable` gate). Low value for the small-net use case (batch=1 is overhead-bound).
- Forward dot-GEMM runs ~5 GFLOP/s (below SIMD peak) — register/tile blocking would help the
  large-batch forward; not pursued (diminishing returns for control-net inference).
- `CpuSimdRuntime` has no buffer pool (`new float[]` per alloc) and no trace/replay (nothing to
  amortize without launch overhead) — both intentional.

**Regression gate:** `tools/run-tests.ps1 -Simd` runs the op-parity suite on the managed backend.
