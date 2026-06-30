# Proposal: native CUDA Graph capture for ILGPU

**Status:** draft scope for an upstream ILGPU PR. Not yet sent.
**Origin:** Tensotron needed to fold a fixed-shape train step into one driver call. We built a working
reference against **stock ILGPU 1.5.3** using raw `nvcuda` P/Invoke — no fork required — and verified
it end to end (kernels *and* cuBLAS SGEMM, replayed with a single `cuGraphLaunch`, 5–10× per step on a
4090). This proposes contributing the **general primitive** upstream so downstreams can drop the raw
P/Invoke for an official, error-checked API.

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
    public bool TryUpdate(CudaGraph graph);                // cuGraphExecUpdate (re-bind params w/o re-instantiate)
    public void Dispose();                                 // cuGraphExecDestroy
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

## Likely files to touch (confirm against current `main`)

- `Src/ILGPU/Runtime/Cuda/CudaAPI*.cs` — add the driver bindings.
- `Src/ILGPU/Runtime/Cuda/CudaStream.cs` — `BeginCapture` / `EndCapture`.
- new `Src/ILGPU/Runtime/Cuda/CudaGraph.cs`, `CudaGraphExec.cs`, `CudaStreamCaptureMode.cs`.
- `CudaException` / `CudaError` already cover `CUresult`; reuse.

## Test plan

- **Capture → instantiate → launch** a trivial `v[i] += 1` kernel on a created stream; assert the
  buffer is unchanged immediately after capture (capture records, does not execute) and increments by
  exactly one per `Launch` (this is our existing probe).
- **Capture modes** — smoke each enum value.
- **cuGraphExecUpdate** — capture, change a kernel argument, `TryUpdate`, relaunch, assert new result.
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
launch and leave `cuGraphExecUpdate` for a follow-up.

## Evidence (Tensotron, RTX 4090)

| step | eager | software replay | native graph |
|---|---|---|---|
| batch-1 control-net (4→64→2) | 675 µs | 321 µs (2.1×) | **64 µs (10.6×)** |
| batch-32 MLP (8→128→4) | 622 µs | 307 µs (2.0×) | **128 µs (4.9×)** |
| batch-128 256³ (cuBLAS SGEMM) | 1211 µs | 450 µs (2.7×) | **118 µs (10.2×)** |

Reference implementation: `IlgpuRuntime.TryBuildCudaGraph` + `CuDriver` in
`src/Tensotron/TensorRuntime.cs`; correctness/engagement tests in
`tests/Tensotron.Tests/TraceReplayTests.cs`.
