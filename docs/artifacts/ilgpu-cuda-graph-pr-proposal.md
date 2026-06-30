# Proposal: native CUDA Graph capture for ILGPU

**Status:** submitted upstream as **[m4rs-mt/ILGPU#1602](https://github.com/m4rs-mt/ILGPU/pull/1602)**
(base `master`, from fork `mfagerlund:feature/cuda-graph-capture`, 8 files / +452). Built and tested
against ILGPU `master` @ `ea51bcb`. See "Reference implementation" below for
the exact diff and the on-device numbers.
**Origin:** Tensotron needed to fold a fixed-shape train step into one driver call. We first built a
working reference against **stock ILGPU 1.5.3** using raw `nvcuda` P/Invoke — no fork required. This
proposes contributing the **general primitive** upstream so downstreams can drop the raw P/Invoke for an
official, error-checked API.

## Why

CUDA Graphs are the standard mechanism for amortizing launch overhead across a repeated, fixed-shape
sequence of launches — exactly the small-model training/inference regime where host-side dispatch, not
the GPU, is the bottleneck. ILGPU ships **no** graph API today (verified: zero `cuGraph` /
`BeginCapture` symbols in the assembly). Yet every building block already exists:

- non-null streams via `Accelerator.CreateStream()` (the default stream is the uncapturable NULL stream);
- explicit-stream launchers via `LoadAutoGroupedKernel` / `LoadKernel` (launcher takes an `AcceleratorStream`);
- a `CudaAPI` P/Invoke layer with `CudaException` error wrapping.

The only missing piece is a thin wrapper over ~6 driver entry points. That makes this an **additive,
low-risk** contribution: new types in `ILGPU.Runtime.Cuda`, zero behavior change to existing code.

## "But it doesn't work for OpenCL" — why that's fine, by design

CUDA Graphs are a CUDA driver feature; OpenCL has no equivalent capture/replay primitive. The PR does
**not** add a cross-backend abstraction — so OpenCL (and CPU/Velocity) are simply untouched, not stubbed
with a `NotSupportedException`. This matches how ILGPU already ships backend-specific capability:

- `CudaStream` is `sealed : AcceleratorStream` (verified `CudaStream.cs:24`); `CLStream` is a separate
  `sealed : AcceleratorStream` (`CLStream.cs:24`). The capture methods land on **`CudaStream`**, never on
  the abstract `AcceleratorStream` base (whose only abstract members are `Synchronize` /
  `AddProfilingMarkerInternal`). An OpenCL stream therefore never gains a graph method to no-op.
- The whole `ILGPU.Algorithms/Runtime/Cuda/` tree is CUDA-only public surface with **no OpenCL twin**:
  `CuBlas`, `CuFFT`, `CuRand` (each with its own `*Exception`), plus inline-PTX `CudaAsm` and libdevice
  in core. Maintainers already accept "this type exists only when you have a CUDA device." A `CudaGraph`
  type is the same shape of contribution, not a new precedent.

So the honest framing for the PR description is: *a CUDA-specific primitive in the CUDA namespace,
mirroring cuBLAS/cuFFT* — not "graph support for ILGPU." Nobody expects `CuFFT` to run on OpenCL; nobody
will expect `CudaGraph` to either. The risk isn't backend coverage; it's only whether the maintainer
wants capture-based graphs as the first cut (see "Before sending").

## Proposed public surface

```csharp
namespace ILGPU.Runtime.Cuda;

public enum CudaStreamCaptureMode { Global = 0, ThreadLocal = 1, Relaxed = 2 }

public partial class CudaStream
{
    public void BeginCapture(CudaStreamCaptureMode mode = CudaStreamCaptureMode.Global); // cuStreamBeginCapture_v2
    public CudaGraph EndCapture();                                                        // cuStreamEndCapture
}

public sealed class CudaGraph : IDisposable                 // wraps CUgraph
{
    public CudaGraphExec Instantiate();                     // cuGraphInstantiateWithFlags
    public void Dispose();                                  // cuGraphDestroy
}

public sealed class CudaGraphExec : IDisposable             // wraps CUgraphExec
{
    public void Launch(CudaStream stream);                 // cuGraphLaunch
    public void Dispose();                                 // cuGraphExecDestroy
    // public bool TryUpdate(CudaGraph graph);             // cuGraphExecUpdate — deferred (see below)
}
```

Usage mirrors the CUDA C flow and our reference impl:

```csharp
using var stream = (CudaStream)acc.CreateStream();
stream.BeginCapture();
kernel(stream, n, view);                 // any number of LoadAutoGroupedKernel launches; cuBLAS works too
using var graph = stream.EndCapture();
using var exec  = graph.Instantiate();
exec.Launch(stream);                     // one driver call replays the whole sequence
```

## Reference implementation (built & tested on RTX 4090)

Implemented against the ILGPU fork (`main` @ `ea51bcb`) and verified end to end. The surface
above is real except `TryUpdate`, deliberately deferred to a follow-up (the minimal first cut is
capture + instantiate + launch + destroy). Diff is **+119 lines across 4 edited files, 4 new files**:

- `Src/ILGPU/Runtime/Cuda/CudaAPI.xml` (+23) — six `<Import>` entries (`cuStreamBeginCapture_v2`,
  `cuStreamEndCapture`, `cuGraphInstantiateWithFlags`, `cuGraphLaunch`, `cuGraphDestroy`,
  `cuGraphExecDestroy`); the `T4.Build` step regenerates the P/Invoke layer at build.
- `Src/ILGPU/Runtime/Cuda/CudaAPI.cs` (+62) — hand-written `CudaError` wrappers in the `Graph Methods`
  region (mirrors the existing `Stream Methods` wrappers).
- `Src/ILGPU/Runtime/Cuda/CudaStream.cs` (+33) — `BeginCapture` / `EndCapture`.
- new `CudaGraph.cs`, `CudaGraphExec.cs`, `CudaStreamCaptureMode.cs` — `AcceleratorObject` subclasses,
  same lifetime/`BindScoped`/`VerifyDisposed` idiom as `CudaProfilingMarker`.
- new `Src/ILGPU.Tests/CudaGraphCapture.cs` (+ a line in `Configurations.txt`) — `SkippableFact`s
  guarded by `Accelerator.AcceleratorType != AcceleratorType.Cuda`, the canonical probe (capture does
  not execute; each `Launch` replays exactly once; an N-launch capture replays the whole sequence per
  launch). **6/6 green on the 4090** (2 methods × 3 opt-levels), 308 ms.

A standalone harness (project-references the modified ILGPU) measured eager vs.
one-graph-launch-per-step on the 4090, kernels only (no cuBLAS):

| regime | per-step shape | eager µs/step | graph µs/step | speedup |
|---|---|---|---|---|
| host-bound | n=1024, 48 kernels | 325 | 50 | **6.4×** |
| host-bound | n=4096, 48 kernels | 312 | 48 | **6.5×** |
| mixed | n=65536, 48 kernels | 296 | 51 | **5.8×** |
| gpu-bound | n=4.2M, 8 kernels | 88 | 73 | 1.2× (≈none — *expected*, the step is GPU-bound) |

The honest takeaway is the crossover, not a single trophy number: graphs amortize **host dispatch**, so
the win is large precisely when the step is launch-bound (small models / small batch / many tiny
kernels) and fades to nothing when the GPU is the bottleneck.

### The capture surface composes — prep / batch×N / finalize

Because a captured graph is an independent reusable `CUgraphExec` and `Launch` is just one driver call,
a step does **not** have to be captured as one monolith. Capture small pieces once and compose them
host-side with the repeat count as a plain loop bound:

```csharp
prepExec.Launch(stream);                              // zero accumulators / reset state (captured once)
for (int i = 0; i < N; i++) batchExec.Launch(stream); // one fixed-shape minibatch (captured once)
finalizeExec.Launch(stream);                          // epoch reduction (captured once)
```

`N` is never baked into a graph, so an epoch of 5, 20, or 60 batches reuses the *same three* executable
graphs with **zero re-capture**. State flows through persistent buffers; same-stream ordering supplies
the dependency between pieces. Even N-dependent math in `finalize` (e.g. `acc *= 1/N`) needs no
re-capture if `1/N` is read from a **device-resident scalar** refreshed by a one-float upload between
runs — the same device-scalar trick used for a scheduled LR. The harness confirms this end to end
(`prep + batch×N + finalize`, N ∈ {5,10,20,60}, mean = 1.000 for every N, **5.2–5.6×** vs eager).

This is the argument for keeping the upstream surface **granular** (capture → instantiate → launch as
separate calls) rather than a single fused "capture-and-replay-N-times": the granular primitive
composes into this pattern for free, while a monolith would bake `N` in.

## Implementation notes

- **Route through `CudaAPI`, not raw `[DllImport]`.** Add the six entry points to ILGPU's existing
  generated/maintained CUDA binding and return `CudaError`/throw `CudaException` like every other call.
  Driver functions: `cuStreamBeginCapture_v2`, `cuStreamEndCapture`, `cuGraphInstantiateWithFlags`,
  `cuGraphLaunch`, `cuGraphExecUpdate`, `cuGraphDestroy`, `cuGraphExecDestroy`.
- **Instantiate entry point.** `cuGraphInstantiateWithFlags` is CUDA 12+. If ILGPU still supports CUDA
  11 drivers, select `cuGraphInstantiate_v2` by driver version, or expose both behind one method.
- **Handle ownership.** `CudaGraph`/`CudaGraphExec` own their `CUgraph`/`CUgraphExec`; `Dispose` is
  idempotent. `EndCapture`/`Instantiate` are the only producers.
- **Capture-mode caveat.** Document that `Global` mode forbids unsafe operations from other threads
  during capture (the usual CUDA contract); `ThreadLocal`/`Relaxed` relax it.
- **No graph-node builder.** Capture-based construction only, first cut. Manual node graphs
  (`cuGraphAddKernelNode`, …) are explicitly out of scope.

## Files to touch (verified against `main` @ `ea51bcb`)

- `Src/ILGPU/Runtime/Cuda/CudaAPI.xml` — **the driver bindings are generated from this XML**, not
  hand-written. Add `<Import Name="cuStreamBeginCapture_v2">` etc. alongside the existing
  `cuStreamCreate` / `cuStreamCreateWithPriority` entries; `Src/ILGPU/Static/DllImports.tt` (driven by
  `DllImports.xml`, which `<File>`-includes `CudaAPI.xml`) regenerates the P/Invoke layer. Optional
  hand-written convenience wrappers go in the `partial class CudaAPI` (`CudaAPI.cs` — e.g. the existing
  `CreateStream` wrapper at `CudaAPI.cs:497`).
- `Src/ILGPU/Runtime/Cuda/CudaStream.cs` — add `BeginCapture` / `EndCapture`. The class already exposes
  `public IntPtr StreamPtr` (the raw `CUstream`) and a `StreamFlags` ctor that calls `CreateStream`, so a
  **non-NULL, capturable** stream is one `accelerator.CreateStream()` away — no new plumbing needed.
- new `Src/ILGPU/Runtime/Cuda/CudaGraph.cs`, `CudaGraphExec.cs`, `CudaStreamCaptureMode.cs`.
- `CudaException.ThrowIfFailed(...)` / `CudaError` already wrap every driver call (`CUresult`); reuse them
  exactly as `CudaStream` does today.

## Verification against current `main` (`ea51bcb`)

- **No prior art to collide with:** zero `cuGraph` / `BeginCapture` / `StreamCapture` symbols anywhere in
  `Src/` (the only hits are the unrelated substring in `cuStream*` creation imports). Clean additive PR.
- **Generation mechanism confirmed:** `CudaAPI.xml` → `DllImports.tt` is the real binding pipeline (the
  proposal's "route through `CudaAPI`" is the *only* sanctioned way; raw `[DllImport]` would not match the
  codebase).
- **Stream surface confirmed:** `CudaStream.StreamPtr` + `StreamFlags` ctor (above) give exactly the
  handle and the non-NULL stream the capture calls require.
- **Backend isolation confirmed:** `CudaStream`/`CLStream` are independent `sealed : AcceleratorStream`;
  the base is abstract over only `Synchronize`/profiling. Adding to `CudaStream` cannot leak into OpenCL.

## Test plan

Implemented now in `CudaGraphCapture.cs` (the first two); the rest are follow-ups to add before/with the PR:

- **[done] Capture → instantiate → launch** a trivial `v[i] += 1` kernel on a created stream; assert the
  buffer is unchanged immediately after capture (capture records, does not execute) and increments by
  exactly one per `Launch`.
- **[done] Multi-launch capture** — a capture spanning N kernel launches replays the whole sequence per
  graph launch (asserts `N × replays` increments).
- **Capture modes** — smoke each enum value.
- **cuGraphExecUpdate** — capture, change a kernel argument, `TryUpdate`, relaunch, assert new result
  (deferred with the API itself).
- **Negative** — `EndCapture` without `BeginCapture` throws a wrapped `CudaException`; a stream-capture
  invalidation surfaces as an error rather than a silent wrong graph.
- **cuBLAS interop** — capture a `CuBlas.Gemm` on the stream, launch, compare to eager (proves library
  calls on a captured stream are recordable; Tensotron relies on this).

## What stays downstream (NOT in the PR)

Tensotron-specific policy, not a general primitive: recording one replay thunk per launch, the
`AfterLaunch` capture tap, the fail-safe fallback to host-side replay, device-resident Adam bias-
correction advance, the steady-state warmup contract, and the "eager on the default stream, swap to the
capture stream only while recording" integration. The PR is the wrapper; the orchestration is ours.

## Before sending

Check the ILGPU issue tracker / discussions for an existing CUDA-graph issue or a maintainer design
preference (m4rs-mt is active but selective). A minimal first PR can ship capture + instantiate +
launch and leave `cuGraphExecUpdate` for a follow-up. Note the maintainers are mid-flight on **2.0**
(branch `new_architecture_v2`); attention may be limited and the CUDA runtime may be refactored there.

**Target branch:** `master` (the active 1.x line; `new_architecture_v2` is the 2.0 work). Our change is
additive to the current CUDA runtime.

### PR-convention conformance (verified against `master` CI gates @ `ea51bcb`)

All required `ci.yml` checks were reproduced locally and pass:

- **Line length ≤ 90** (`CheckLineLength.ps1 -path Src`) — all 8 files pass (the script's only hits are
  the *untracked, regenerated* `CudaAsm.Generated.cs`, which CI never sees because the line-length job
  checks out without building).
- **T4 line endings** (`CheckT4LineEndings.ps1`) — only scans `*.tt`/`*.ttinclude`; we touch none (the
  driver bindings are added via `CudaAPI.xml`, regenerated by the existing `DllImports.tt`).
- **Copyright headers** (`Tools/CopyrightUpdateTool`, which fails the build on any diff) — headers match
  the tool's output **exactly**: UTF-8 BOM present; centered via the tool's own `CenterAlignString`
  (validated by reproducing a known canonical line byte-for-byte); new files single-year, edited files'
  end-year bumped to the commit year. (Years are commit-date-driven and auto-managed by ILGPU's tooling;
  if the merge lands in a later year the copyright job will flag the one-line bump.)
- **Build with `-p:TreatWarningsAsErrors=true`** — core ILGPU and `ILGPU.Tests.Cuda` build **0
  warnings** (analyzers run at `AnalysisMode=AllEnabledByDefault`).
- **Tests** — `CudaGraphCapture` uses `SkippableFact` guarded on `AcceleratorType != Cuda`, so it runs
  on a GPU box (6/6 green on a 4090) and **Skips** in ILGPU's GPU-less CI — matching how their CUDA
  tests are already gated (CUDA runners are disabled in `setup-os-matrix`).
- **Idioms** — new handle types are `AcceleratorObject` subclasses using the exact
  `CudaProfilingMarker` lifetime pattern (`BindScoped` / `CudaException.VerifyDisposed`); registered in
  `Configurations.txt` like every other test.

PR surface: **4 files modified (+119 lines), 4 files added**; `global.json` left pristine (local builds
needed an SDK roll-forward, kept out of the diff).

## Evidence (Tensotron, RTX 4090)

| step | eager | software replay | native graph |
|---|---|---|---|
| batch-1 control-net (4→64→2) | 675 µs | 321 µs (2.1×) | **64 µs (10.6×)** |
| batch-32 MLP (8→128→4) | 622 µs | 307 µs (2.0×) | **128 µs (4.9×)** |
| batch-128 256³ (cuBLAS SGEMM) | 1211 µs | 450 µs (2.7×) | **118 µs (10.2×)** |

Reference implementation: `IlgpuRuntime.TryBuildCudaGraph` + `CuDriver` in
`src/Tensotron/TensorRuntime.cs`; correctness/engagement tests in
`tests/Tensotron.Tests/TraceReplayTests.cs`.
