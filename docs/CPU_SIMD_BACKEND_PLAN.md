# CPU SIMD Backend — Architecture & Plan

**Decision (2026-06-29):** build a **full CPU backend (train + inference)** with torch parity,
**native SIMD via the struct-generic `IOp` seam** (not a bridge to an external SIMD lib, not an
inference-only path). Keep ILGPU's scalar CPUAccelerator as a separate debug backend.

## Why (measured, not assumed)

The ILGPU "CPU" backend (`TensorBackend.Cpu`) is ILGPU's **scalar `CPUAccelerator`** running the
*same* per-op dispatch machinery as CUDA: allocate a device-style buffer, upload stride/dim int
buffers, launch through the auto-grouped stream, drain. At **batch=1 inference of a tiny control
net** that overhead measured **~274× slower than hand-scalar** — and the tax is **dispatch, not
arithmetic** (the net is a handful of small matmuls = microseconds of FLOPs).

So the lever order is:

1. **A managed `float[]` execution path with zero ILGPU in the loop** removes the dispatch tax →
   the bulk of the 274×.
2. **SIMD (`Vector<float>`) on the now-dominant inner loops** is the second-order win on top.

Building SIMD `Apply` overloads and bolting them onto ILGPU dispatch would buy ~nothing — the
managed path has to come first. This plan does both, in that order.

## What carries over for free (device-agnostic already)

- **Autograd** — the toposort `Backward()`, `GradNode` graph, and broadcast-reduction logic are all
  host-side and storage-agnostic.
- **`IOp` structs** (`Ops.cs`, `Ops.Unary.cs`, `Ops.Reduce.cs`) — `default(TOp).Apply(...)` inlines
  identically in a managed loop; we *add* `Vector<float>` overloads, we don't rewrite them.
- **`Shape`/stride math**, **`Module`/`Linear`/`Sequential`**, **optimizers' host orchestration**,
  **serialization**, and **the entire torch-fixture suite** (the parity gate is reused verbatim).

## Core architecture

The runtime is **single-backend, process-wide** (`TensorRuntime.Instance`, chosen once from
`RequestedBackend`). That invariant is the whole simplification: **no per-tensor device tag is
needed** — when the CPU-SIMD backend is selected, *every* tensor is a host `float[]` and *every* op
dispatches to the managed kernels. So the abstraction is two clean swaps, not pervasive polymorphism
on each call.

### 1. Storage abstraction (`Tensor.Buffer` → `Tensor.Storage`)

- Today: `internal MemoryBuffer1D<float, Stride1D.Dense> Buffer` (`Tensor.cs:70`).
- Introduce `abstract class TensorStorage { int Length; float[] ToHost(); void CopyFromHost(float[]); void Dispose(); bool OwnsBuffer; }`
  with two impls:
  - `DeviceStorage` — wraps `MemoryBuffer1D` (CUDA / ILGPU-CPU debug).
  - `HostStorage` — wraps `float[]` (CPU-SIMD), pooled by length like the device allocator.
- The op layer's 51 `.Buffer` sites become `.Storage` (mechanical rename — they only *pass* it on).
- `ToArray`/`CopyFromHost`/`Upload`/`Detach` in `Tensor.cs` move their concrete logic into the
  storage impl. `Detach`/`Reshape` still share the parent's storage (zero-copy, `OwnsBuffer=false`).

### 2. Runtime abstraction (`ITensorRuntime`, two impls)

- Extract the public `Launch*` + `Allocate`/`Sync`/`ResetCounters`/pool surface (~30 methods) into
  `ITensorRuntime`. Signatures take `TensorStorage` instead of `MemoryBuffer1D<float>`; each impl
  downcasts to the concrete storage it owns (safe — single backend per process).
- `CudaRuntime` — today's `TensorRuntime`, behaviorally unchanged (it downcasts `Storage`→`DeviceStorage`).
- `CpuSimdRuntime` — new. `Allocate` hands back a pooled `HostStorage`; `Launch*` run **managed,
  synchronous** kernels (no stream → `Sync()` is a no-op, `FlushEvery`/`Capture` are no-ops or throw,
  the int stride/dim buffers stay as plain `int[]`, no upload). cuBLAS path → managed GEMM.
- `TensorRuntime.Instance` resolves to the impl from `RequestedBackend` at first use.

### 3. CPU kernels (`CpuKernels.*`, the managed twin of `Kernels.*`)

Each `Kernels.*` method has a managed sibling over `float[]`/`Span<float>` with the **same index
math** (the kernels are already pure strided index functions — the port is the loop, not the logic):

- **Scalar baseline first** (correctness + parity), then a **`Vector<float>` fast path** on the
  contiguous output axis. Broadcast/strided cases keep the scalar gather path.
- **Reuse the `IOp` structs**: add `Vector<float> Apply(...)` (binary) / `Vector<float> Forward/Backward`
  (unary) overloads. **Transcendentals** (Exp/Tanh/Sigmoid/Gelu/…) start as a **scalar `MathF`
  fallback** (the parity baseline) and get vectorized only where they stay within torch tolerance.
- **MatMul** — native SIMD GEMM (FMA accumulation over the contiguous-K inner loop with
  `Vector<float>`). Use Colonel.Hagrid's `MatMuler`/`Aggregator` as a *reference* for loop shape and
  ILP unrolling; write native against our stride contract so backward stays a stride swap on the same
  2D core. Optional `Parallel.For` over output rows for large products (still single-threaded **per
  op** — Tensotron's single-threaded law holds; we never run two ops concurrently).

### 4. Parity harness (the "or it doesn't land" gate, reused)

The torch fixtures already run on "whatever accelerator is active." Add a way to force the suite onto
CpuSimd — `TENSOTRON_BACKEND=simd` env, or an xUnit collection that flips `RequestedBackend` — so
every op's committed fixture asserts **CPU-SIMD forward + backward parity** against torch, same
tolerances. No new test authoring; the existing golden JSON is the spec.

## Phasing (each phase ends green; no phase regresses CUDA/ILGPU-CPU)

- **Phase 0 — abstraction (pure refactor, no behavior change).** `TensorStorage` + `ITensorRuntime`;
  ILGPU code moves behind the interface as `CudaRuntime`. Suite stays green on CUDA and ILGPU-CPU.
  *Blast radius: 51 `.Buffer`→`.Storage` renames + ~30 runtime signatures, all in `src/Tensotron`.*
- **Phase 1 — CpuSimd skeleton + elementwise/reduce.** `HostStorage`, `CpuSimdRuntime`, managed
  binary/unary fwd+bwd, `AddInto`, reductions, strided copy (scalar). Run the elementwise/reduce
  fixtures on `simd` → green.
- **Phase 2 — matmul + Linear + optimizer.** Managed GEMM (scalar→correct), fused Adam/SGD managed
  update, broadcast-add. Train a `Linear` MLP end-to-end on `simd`; parity vs CUDA.
- **Phase 3 — the rest of the op surface.** Conv (im2col/col2im), pool, norm, index/gather/scatter,
  losses — each gated by its fixture on `simd`.
- **Phase 4 — SIMD.** `Vector<float>` overloads on the `IOp` structs, vectorized GEMM inner loop and
  reductions, vectorized transcendentals where they hold tolerance. Benchmark: scalar-managed vs
  SIMD-managed vs ILGPU-CPU vs CUDA.
- **Phase 5 — inference ergonomics + the headline number.** NoGrad fast path, storage reuse, measure
  batch=1 control-net latency vs the 274× baseline; document in `PERFORMANCE_VS_PYTORCH.md`.

## Risks / open points

- **Storage downcast safety** rests on single-backend-per-process — already an invariant; assert it.
- **Transcendental parity** — vectorized Exp/Tanh must match torch within fixture tol; keep the
  scalar `MathF` path as the baseline and only vectorize when measured in-tolerance.
- **Threading** — ILGPU-CPU was implicitly multi-threaded; the managed path may `Parallel.For`
  *within* a single op for large tensors, but must never run two ops concurrently (the single-threaded
  law) and must stay deterministic enough for the fixtures (no atomic-order nondeterminism on CPU →
  CPU reductions can actually be *more* deterministic than the GPU ones).
- **`Capture`/`Replay`** are GPU-launch-overhead amortizers; on CPU there's no launch to amortize, so
  they're no-ops/unsupported on `CpuSimdRuntime` (document, don't fake).
