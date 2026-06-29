# Tensotron

A **PyTorch-faithful**, float32 tensor and autograd library for .NET, GPU-accelerated with
[ILGPU](https://github.com/m4rs-mt/ILGPU).

> **The law:** for every op it implements, Tensotron matches PyTorch exactly - naming, semantics,
> broadcasting, gradients (including behavior at kinks/ties/special values). Porting PyTorch code is
> near-mechanical *within the supported surface*, which is a deliberate subset of torch.

> **Scope:** feed-forward training/inference - MLPs, CNNs, small RL nets. Not yet implemented:
> RNN/LSTM/GRU, attention/transformers, `Embedding`, `ConvTranspose`, `Conv1d`/`Conv3d`, dtypes
> beyond float32, and PyTorch `state_dict` interop.

## Highlights

- **Define-by-run autograd** - toposort backward over a named op graph; `Tensor.Backward()`.
- **Broad op surface, every deterministic op torch-parity tested** - elementwise math + activations,
  broadcasting, reductions, 2D/N-D batched matmul, movement/structure ops, indexing
  (`gather`/`scatter_add`/`index_select`), `Conv2d`, `MaxPool2d`/`AvgPool2d`, LayerNorm /
  BatchNorm / GroupNorm, and the common losses (MSE, L1, Huber, BCE-with-logits, NLL,
  cross-entropy, KL-div).
- **Training stack** - `Module`/`Sequential`/`Linear`, SGD/Adam/AdamW/RMSProp, LR schedulers,
  Kaiming/Xavier init, `DataLoader`, and full-checkpoint save/load (params + buffers + optimizer &
  LR-scheduler state, so training resumes exactly).
- **Runs without a GPU** - `Auto` (the default) uses CUDA when present and otherwise falls back to a
  hand-written managed/SIMD CPU backend (`TENSOTRON_BACKEND=simd`; no per-op device dispatch, ~645×
  the ILGPU scalar CPU path at batch-1) — the fast path for small-model CPU inference/training. Its
  matmul has opt-in row parallelism (`TENSOTRON_CPU_THREADS=auto`, ~5–12× on big-batch GEMMs; off by
  default). ILGPU's scalar CPU accelerator (`TENSOTRON_BACKEND=cpu`) is kept only as a slow
  correctness-verification reference and warns loudly when selected.

## Status

`0.1.0-alpha`. Large matmul runs on **cuBLAS SGEMM** (CUDA; matches PyTorch FP32 throughput at
scale), with tiled/naive ILGPU kernels otherwise. The runtime is **async**: kernels queue on
ILGPU's in-order default stream and synchronize **only** at host pulls (`ToArray`/`Item`), not
per launch. A size-bucketed **caching allocator** (opt-in via `Dispose`/`DisposeGraph`) reuses
device buffers, and shape/stride metadata is uploaded once and cached. Adam/SGD are fused
single-kernel updates. Buffers are `IDisposable` (deterministic opt-in release); zero-copy views
never free their parent's buffer. The per-op host-side autograd graph (a `Tensor`/`GradNode` per
op every step) is the dominant cost for very small models — ~95% host-bound — with an opt-in escape
hatch: `TensorRuntime.Capture`/`CapturedGraph.Replay` records a fixed-shape step once and replays
its device launches buffer-to-buffer (~2.5–2.9× faster on a small step). Cross-op kernel fusion is
not implemented. float32-only *storage* by design (so no FP16/BF16 path), but TF32 tensor-core
matmul is an available one-line knob (`TensorRuntime.AllowTf32`), currently left off for exact FP32.

## Quick start

```csharp
using Tensotron;

var x = Tensor.FromArray(new[] { 1f, 2f, 3f }, 3).RequireGrad();
var y = (x * x).Sum();   // y = sum(x_i^2)
y.Backward();
// x.Grad == [2, 4, 6]
```

## License

MIT
