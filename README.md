# Tensotron

Tensotron is a GPU tensor and autograd library for .NET, built on **ILGPU** with **float32** storage: a principled, PyTorch-faithful tensor library whose tensors live on the device.

> **Current backend status.** Matmul runs on custom ILGPU kernels (not cuBLAS — that's a planned swap, see *Status* below), and the runtime is **correctness-first**: it `Synchronize()`s after essentially every kernel launch rather than queueing on the async stream. The async-stream, sync-only-at-host-pull and cuBLAS-GEMM behavior described in *Design* below is the target architecture, not the current implementation.

> **The law:** Tensotron mimics PyTorch in everything — naming, semantics, broadcasting, gradients. If it doesn't behave like PyTorch, it's a bug. Converting PyTorch code to Tensotron should be near-mechanical, because the names and behavior are what you'd expect.

## Quick start

```csharp
using Tensotron;

var x = Tensor.FromArray(new[] { 1f, 2f, 3f }, 3).RequireGrad();
var y = (x * x).Sum();   // y = Σ xᵢ²
y.Backward();
// x.Grad == [2, 4, 6]
```

And a complete training loop is the same shape you'd expect from PyTorch:

```csharp
var model = new Sequential(
    new Linear(2, 64), Activation.Relu(),
    new Linear(64, 3));
var opt = new Adam(model.Parameters().ToList(), lr: 1e-2f);

for (int epoch = 0; epoch < 1000; epoch++)
{
    var loss = TensorOps.CrossEntropy(model.Forward(x), labels);
    opt.ZeroGrad();
    loss.Backward();
    opt.Step();
}
```

### Runnable examples

`examples/Tensotron.Examples` is a console app with three from-scratch demos — start here:

```
dotnet run --project examples/Tensotron.Examples            # runs all three
dotnet run --project examples/Tensotron.Examples xor        # smallest training loop
dotnet run --project examples/Tensotron.Examples spiral     # 3-class spiral → spiral.svg
dotnet run --project examples/Tensotron.Examples regression # noisy sine fit → regression.svg
```

No GPU needed (CPU-accelerator fallback). The spiral and regression demos write an SVG you
can open in a browser. See [examples/README.md](examples/README.md).

## Design

### Define-by-run forward graph

Each result records its `Inputs` and a per-op backward closure as it is computed. The lifecycle is PyTorch-faithful: `NoGrad`, `ZeroGrad`, gradient clipping. `Backward()` clears interior (non-leaf) node grads at entry and accumulates fresh, while leaf grads accumulate across calls — i.e. repeated `backward()` behaves like torch's `retain_graph=True`; call `ZeroGrad` between passes to reset.

Shape/stride is decoupled from storage: reshape/transpose/broadcast are zero-copy stride manipulation (stride-0 marks broadcast dims) — exactly what a strided GPU kernel wants. A host-side broadcaster computes output shapes plus the added/expanded-dimension bookkeeping that drives gradient reduction in backward.

### Autograd: explicit topological sort

Backward is the standard **micrograd/PyTorch design: explicit topological sort, then run backward sequentially in reverse.** There is no gradient-count completion scheme — by the time toposort reaches a node, every downstream consumer has already accumulated into it, so accumulation is just `grad += …` and the "have all consumers reported?" question never arises. Order is deterministic and printable.

Each op's backward is a **named-closure node**: `GradNode { string OpName, Tensor[] Inputs, Action<Tensor> Backward }`. Closures stay terse (~1 line per op), and nodes are named so the graph prints and traces are readable.

```csharp
// per op, in the dispatch helper:
result.Node = new GradNode(
    opName: "Add",
    inputs: [a, b],
    backward: g => {
        a.AddGrad(ReduceL(g));   // ReduceL/R handle broadcast reduction
        b.AddGrad(ReduceR(g));
    });

// engine:
var order = TopoSort(loss);
loss.Grad = Ones(loss.Shape);
foreach (var n in order.Reversed())
    n.Backward(n.Grad);          // n.Grad is already fully accumulated
```

### ILGPU hygiene

- One `Context`/`Accelerator`, owned centrally by `TensorRuntime`.
- Kernels compiled once via `LoadAutoGroupedStreamKernel` and **cached in fields/dictionaries** — never recompiled per call.
- Graceful CUDA → CPU fallback, so the suite runs without a GPU.

### The hard parts

- **Matmul.** A 2D GEMM core with rank/broadcast/autograd choreography in the portable layer. Handles, PyTorch-style: 1D@1D (dot→scalar), 1D@2D, 2D@1D, 2D@2D, and N-D batched with **broadcast batch dims**; rank promotion then squeeze-back; and backward `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC` **with batch-dim reduction**. Currently custom ILGPU kernels; a cuBLAS GEMM swap (via ILGPU.Algorithms) and a hand-tiled GEMM are later perf options behind the same interface.
- **Elementwise: struct-generic kernels, not an op-enum switch.** ILGPU kernels can't take a `Func<float,float>`. One generic kernel is parameterized by a `struct IOp { float Apply(...) }` so each op inlines at JIT (the same idiom as `ILGPU.Algorithms` reductions taking an `IScanReduceOperation`). A single arbitrary-rank strided kernel — global thread id → multi-dim index via the broadcaster's strides (stride-0 = broadcast) — covers every rank with no rank ceiling.
- **Stay on device.** Data is a `MemoryBuffer1D<float>`. Materialize to host **only** on explicit `.ToArray()`/`.Item()`. The eager per-op model (one kernel launch per op, in-place gradient accumulation) is accepted for correctness first; fusion comes later, behind the same surface.
- **Sync is the enemy, not launch count.** The target model runs every op on ILGPU's async default stream and `Synchronize()`s **only** at host pulls, so hundreds of tiny kernels just queue async and the device drains them. (The current implementation syncs after every launch — correctness-first; see *Status*.) Reducing launches is a later optimization, in two cheap-then-expensive steps: (1) hand-fuse the compounds PyTorch itself fuses (`addcmul`, fused bias+activation) as single kernels — still PyTorch-named; (2) only if profiling demands it, a lazy elementwise fuser. Defer (2) hard.

### Storage & dtype

- **float32 only.** Matches typical RL/training workloads and keeps the kernel surface small.
- A `Tensor` is `(Shape, MemoryBuffer1D<float>, autograd metadata)`. Shape/strides stay host-side; data stays on device.

### Concurrency

`TensorRuntime` is a **process-wide singleton** and its launch methods share accelerator state without internal serialization. **Tensor execution is single-threaded only** — do not call ops from multiple threads concurrently. (The test suite disables parallelization for this reason.) Internal launch locking is a possible future change.

### Determinism note

GPU reductions via atomics are **not** bitwise-deterministic across runs. PPO-style training tolerates this. If reproducible training is ever needed, use fixed-order tree reductions rather than atomics.

## Grounding in PyTorch

Every op ships with a **golden-fixture parity test or it doesn't land.** A small Python script (`tools/fixtures/gen.py`) emits, per op, `(inputs, forward_output, grads)` from torch into JSON; the C# test asserts **forward and backward** match within tolerance (a `gradcheck` equivalent). torch is needed only to regenerate fixtures — the committed JSON means the C# suite never imports it.

**Fixtures must probe the hard parts, not just the smooth interior.** Random (`randn`) inputs alone are a trap: they never land on a non-differentiable kink (ReLU/LeakyReLU/ELU/clamp at the boundary), never tie (`maximum`/`minimum` with `a == b`), and never hit special values (zero-probability KL targets). Those are exactly where naive implementations diverge from torch. So fixtures additionally include, for any op with a kink, tie, or special value:

- **boundary inputs** that hit the kink exactly (e.g. `x == 0` for activations, `x == bound` for clamp),
- **parameter sweeps** across regimes that break shortcuts (e.g. LeakyReLU slope `< 1`, `> 1`, and `< 0` — a `max(x, slope·x)` shortcut is only correct for `slope < 1`),
- **ties and special values** (equal operands for min/max; exact-zero targets for KL).

torch defines the truth at those points; the fixture records it. Edge cases use a deterministic `grad_output` (ones) so the recorded boundary gradient is unambiguous.

## Stack

- .NET 8.0
- ILGPU 1.5.3 + ILGPU.Algorithms 1.5.3
- Custom ILGPU matmul kernels today; cuBLAS (via ILGPU.Algorithms binding) is a planned swap
- float32

## Build & test

```
tools/run-tests.ps1              # preferred: torch-parity + fast smoke, BelowNormal priority
tools/run-tests.ps1 -Showcase    # ONLY the slow Category=Showcase convergence tests (GPU)
dotnet test --filter "Category!=Showcase"   # plain-CLI equivalent of the default run
python tools/fixtures/gen.py     # ONLY when adding/changing a fixture (needs torch)
```

Run the suite via `tools/run-tests.ps1` — it launches the tests at low process priority
(so a long run doesn't hog the machine) and kills stray test hosts first. Pass `-Filter`
to scope (e.g. `-Filter "FullyQualifiedName~PoolTests"`).

**Default run excludes `Category=Showcase`.** Those are the full-strength convergence demos
(pole-cart PPO, MNIST CNN) — minutes-to-tens-of-minutes, intended for a GPU. The default run
keeps the torch-parity suite ("Tensotron works") plus the fast always-on `ShowcaseSmokeTests`
(which assert learning *improves* cheaply). Run the convergence demos explicitly with
`-Showcase`. Note: a bare `dotnet test` with no filter runs **everything**, including the slow
showcase tests.

Without a CUDA GPU the convergence demos report **Skipped** (they gate on `Cuda.IsAvailable()`
via `SkippableFact`), so `-Showcase` is safe to run anywhere — on a CPU-only box it just skips
the expensive training instead of grinding for minutes on the CPU fallback.

Tests run on ILGPU's accelerator — CUDA if present, else the CPU accelerator (so the full
suite runs without a GPU).

## Status

**A full training stack, every op torch-verified.** The tensor/backend split, the
cached-kernel runtime (`TensorRuntime`), and toposort + `GradNode` autograd are in place,
and the op surface is broad enough to build and train real networks — each op passing
forward **and** backward torch-parity tests. Matmul backward uses the stride-swap transpose
trick (no transpose copies); broadcast gradients reduce correctly.

What's landed (all parity-tested against PyTorch):

- **Core ops** — add/sub/mul/div, unary math + activations (relu/tanh/sigmoid/gelu/exp/log/sqrt/…), broadcasting, reductions (sum/mean/var/std/min/max/argmin/argmax/prod).
- **Linear algebra** — 2D matmul and N-D batched matmul with broadcast batch dims.
- **Movement / structure** — reshape/view, squeeze/unsqueeze, flatten, expand, permute/transpose, narrow, cat/stack, chunk/split.
- **Indexing** — index_select, gather, scatter_add, repeat.
- **NN** — `Module`/`Sequential`/`Linear`, dropout, **Conv2d** (im2col + batched matmul), **MaxPool2d / AvgPool2d**, normalization (LayerNorm, BatchNorm1d/2d, GroupNorm).
- **Losses** — MSE, L1, Huber, BCE-with-logits, NLL, cross-entropy, KL-div.
- **Training** — SGD/Adam/AdamW/RMSProp, grad-norm clipping, LR schedulers (Step/Exponential/Cosine/Linear), Kaiming/Xavier init, `DataLoader`, binary serialization (save/load).
- **Device** — torch.cuda-flavored availability probe: `Cuda.IsAvailable()` / `DeviceCount()` / `GetDeviceName()`, plus `Accelerators.List()` / `Active()` diagnostics. One global accelerator (CUDA-preferred, CPU fallback); there is intentionally no per-tensor device or `tensor.to(device)`.

Layout:
- `src/Tensotron/` — `Shape`, `Tensor` (autograd), `TensorOps.*` (op surface), `Kernels`, `TensorRuntime`, `Ops` (struct-generic op types), plus `Module`/`Conv`/`Pool`/`Norm`/`Optimizers`/`LrSchedulers`/`Init`/`DataLoader`/`Serialization`.
- `tests/Tensotron.Tests/` — torch-parity tests + `Fixtures` loader; `Fixtures/*.json` committed. **"Tensotron works."**
- `showcase/Tensotron.Showcase/` — end-to-end usability tasks (continuous-PPO pole-cart control, MNIST CNN) that assert real learning and emit SVG replays. **"Tensotron can be used for these things."**
- `tools/fixtures/gen.py` — torch fixture generator (self-embeds its source into each JSON).

**Buffer lifetime.** `Tensor` is `IDisposable`. Each tensor owns its device buffer except
zero-copy views (`Detach`, `Reshape`), which share the parent's buffer and never free it
(`OwnsBuffer` tracks this; `Dispose()` is idempotent and frees only an owned buffer).
Disposal is the *deterministic opt-in* for inference / no-grad loops that want to bound device
memory; autograd intermediates stay reachable until backward and are otherwise GC-reclaimed
(ILGPU buffers carry finalizers). A pooling/arena allocator to remove per-op allocation churn
is still future work.

Next:
- cuBLAS GEMM backend behind the existing matmul interface.
- Async-stream execution: queue launches and `Synchronize()` only at host pulls (current code
  syncs after essentially every launch — correctness-first).
- Buffer pooling/arena + keep stride buffers resident (current code allocs per op / per strided launch).
- Optional internal launch serialization for multi-threaded callers.

## License

MIT — see [LICENSE](LICENSE).
