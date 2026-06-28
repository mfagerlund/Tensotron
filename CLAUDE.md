# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Tensotron is a PyTorch-faithful, **float32-only** GPU tensor + autograd library for .NET 8, built on **ILGPU**. See `README.md` for the full design rationale and current backend status; this file is the operational distillation.

## The law (non-negotiable)

**Tensotron mimics PyTorch in everything — naming, semantics, broadcasting, gradients.** If an op doesn't behave like PyTorch (down to behavior at kinks, ties, and special values), it's a bug, not a design choice. Converting PyTorch code to Tensotron should be near-mechanical. When in doubt about an op's name, signature, or edge behavior, match torch — don't invent.

## Build & test

```
tools/run-tests.ps1                          # preferred: torch-parity + fast smoke, at BelowNormal priority
tools/run-tests.ps1 -Filter "FullyQualifiedName~PoolTests"   # scope to a subset
tools/run-tests.ps1 -Showcase                # ONLY the slow Category=Showcase convergence demos (GPU)
dotnet test --filter "Category!=Showcase"    # plain-CLI equivalent of the default run
dotnet build Tensotron.sln --verbosity quiet # (use --verbosity quiet, never -q)
```

- `run-tests.ps1` runs at low process priority and first kills stray testhosts **scoped to this repo path** (safe on a shared machine). Prefer it over bare `dotnet test`.
- **A bare `dotnet test` with no filter runs everything, including the minutes-long showcase convergence tests.** The default `run-tests.ps1` excludes `Category=Showcase` but keeps the fast always-on `ShowcaseSmokeTests`.
- Tests run on ILGPU's accelerator: CUDA if present, else the **CPU accelerator** — so the full parity suite runs without a GPU. Showcase convergence demos gate on `Cuda.IsAvailable()` via `SkippableFact` and report **Skipped** on a CPU-only box.
- **No torch dependency to build or test.** torch is needed *only* to regenerate fixtures.

## Adding or changing an op (the core workflow)

Every op ships with a **golden-fixture parity test or it doesn't land** — asserting both **forward and backward** match torch within tolerance.

1. Implement the op in the relevant `src/Tensotron/TensorOps.*.cs` partial (Binary/Unary/Reduce/Movement/Matmul/Conv/Pool/Norm/Losses/Index/Select/Structural). Each op computes forward via a kernel, then (unless `NoGrad`) attaches a named `GradNode` whose closure deposits input gradients. Broadcast reduction in backward goes through `ReduceGradToShape`.
2. Add cases to `tools/fixtures/gen.py` and run `python tools/fixtures/gen.py` (the **only** place torch is needed) to emit JSON under `tests/Tensotron.Tests/Fixtures/`. The generator self-embeds its own source into each JSON for provenance.
3. **Fixtures must probe the hard parts, not just `randn` interior:** boundary inputs that hit the kink exactly (e.g. `x==0` for activations, `x==bound` for clamp), parameter sweeps that break shortcuts (e.g. LeakyReLU slope `<1`, `>1`, `<0`), and ties/special values (equal operands for min/max; exact-zero targets for KL). Edge cases use a deterministic `grad_output` (ones) so the recorded boundary gradient is unambiguous.
4. The C# test reads the committed JSON and asserts forward + backward parity — it never imports torch.

## Architecture (the big picture)

- **Storage/shape split.** A `Tensor` is `(Shape, MemoryBuffer1D<float>, autograd metadata)`. Shape/strides stay host-side; data stays on device. reshape/transpose/broadcast/expand are **zero-copy stride manipulation** (stride-0 marks a broadcast dim) — the matmul-backward transpose is a stride swap, not a copy. Materialize to host **only** on explicit `.ToArray()`/`.Item()`.
- **Autograd = explicit toposort, micrograd/PyTorch style.** Each op records a `GradNode { OpName, Inputs, Backward }`. `Backward()` does a topological sort then runs node backward closures in reverse; by the time a node is reached all consumers have accumulated into it, so accumulation is just `grad += …` (no gradient-count completion scheme). Interior grads are cleared at `Backward()` entry; **leaf grads accumulate across calls** (like torch `retain_graph=True`) — call `ZeroGrad` between passes.
- **`TensorRuntime` is a process-wide singleton** owning the single `Context`/`Accelerator`. Kernels are compiled once via `LoadAutoGroupedStreamKernel` and **cached in fields/dictionaries** — never recompiled per call. Graceful CUDA→CPU fallback. There is intentionally **no per-tensor device / `tensor.to(device)`**.
- **Elementwise kernels are struct-generic, not an op-enum switch.** ILGPU can't take a `Func`, so one arbitrary-rank strided kernel is parameterized by a `struct IOp { float Apply(...) }` that inlines at JIT (same idiom as `ILGPU.Algorithms`' `IScanReduceOperation`). Global thread id → multi-dim index via strides; no rank ceiling.
- **Matmul** is a 2D GEMM core with rank/broadcast/autograd choreography in the portable layer: handles 1D@1D, 1D@2D, 2D@1D, 2D@2D, and N-D batched with broadcast batch dims; backward is `dA = dC@Bᵀ`, `dB = Aᵀ@dC` with batch-dim reduction. Custom ILGPU kernels today; cuBLAS swap is planned behind the same interface.

## Hard constraints

- **float32 only.** No other dtype — keeps the kernel surface small. Don't add dtype generality.
- **Single-threaded execution only.** `TensorRuntime` launch methods share accelerator state without internal serialization — do **not** call ops from multiple threads concurrently. The test suite disables xUnit parallelization for this reason (`AssemblyInfo.cs`).
- **Not bitwise-deterministic.** GPU reductions use atomics; results vary across runs. PPO-style training tolerates this. Don't assume reproducibility.
- **Buffer lifetime.** `Tensor` is `IDisposable` and owns its device buffer **except** zero-copy views (`Detach`, `Reshape`) which share the parent's buffer and never free it (`OwnsBuffer` tracks this; `Dispose()` is idempotent). Disposal is the deterministic opt-in for inference / no-grad loops; autograd intermediates are GC-reclaimed via ILGPU finalizers.
- **Correctness-first runtime.** Current code `Synchronize()`s after essentially every launch and allocs per op. Async-stream execution, buffer pooling, and resident stride buffers are deliberate *future* work — don't assume they exist (`README.md` "Status"/"Next").

## Layout

- `src/Tensotron/` — `Shape`, `Tensor`, `TensorOps.*` (op surface), `Kernels`, `TensorRuntime`, `Ops*` (struct-generic op types), plus `Module`/`Conv`/`Pool`/`Norm`/`Optimizers`/`LrSchedulers`/`Init`/`DataLoader`/`Serialization`. Ships as NuGet package `Tensotron` (uses `PACKAGE.md` as the packed readme, not `README.md`).
- `tests/Tensotron.Tests/` — torch-parity tests + `Fixtures.cs` loader; `Fixtures/*.json` committed. "Tensotron works."
- `showcase/Tensotron.Showcase/` — end-to-end tasks (continuous-PPO pole-cart, MNIST CNN) asserting real learning, emitting SVG replays. "Tensotron can be used for these things."
- `examples/Tensotron.Examples/` — runnable console app (`dotnet run --project examples/Tensotron.Examples [xor|spiral|regression]`) with minimal from-scratch training loops for newcomers; spiral/regression write an SVG. Not a test project. `Plot.cs` is a local SVG helper, not part of the library.
- `tools/fixtures/gen.py` — torch fixture generator (the only torch dependency).
