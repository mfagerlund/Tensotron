# Tensotron

A **PyTorch-faithful**, float32 tensor and autograd library for .NET, GPU-accelerated with
[ILGPU](https://github.com/m4rs-mt/ILGPU).

> **The law:** Tensotron mimics PyTorch in everything - naming, semantics, broadcasting,
> gradients. If it doesn't behave like PyTorch, it's a bug. Porting PyTorch code should be
> near-mechanical.

## Highlights

- **Define-by-run autograd** - toposort backward over a named op graph; `Tensor.Backward()`.
- **Broad op surface, every op torch-parity tested** - elementwise math + activations,
  broadcasting, reductions, 2D/N-D batched matmul, movement/structure ops, indexing
  (`gather`/`scatter_add`/`index_select`), `Conv2d`, `MaxPool2d`/`AvgPool2d`, LayerNorm /
  BatchNorm / GroupNorm, and the common losses (MSE, L1, Huber, BCE-with-logits, NLL,
  cross-entropy, KL-div).
- **Training stack** - `Module`/`Sequential`/`Linear`, SGD/Adam/AdamW/RMSProp, LR schedulers,
  Kaiming/Xavier init, `DataLoader`, save/load.
- **Runs without a GPU** - uses CUDA when present, else ILGPU's CPU accelerator.

## Status

`0.1.0-alpha`. Large matmul runs on **cuBLAS SGEMM** (CUDA; matches PyTorch FP32 throughput at
scale), with tiled/naive ILGPU kernels otherwise. The runtime is **async**: kernels queue on
ILGPU's in-order default stream and synchronize **only** at host pulls (`ToArray`/`Item`), not
per launch. A size-bucketed **caching allocator** (opt-in via `Dispose`/`DisposeGraph`) reuses
device buffers, and shape/stride metadata is uploaded once and cached. Adam/SGD are fused
single-kernel updates. Buffers are `IDisposable` (deterministic opt-in release); zero-copy views
never free their parent's buffer. The per-op host-side autograd graph (a `Tensor`/`GradNode` per
op every step) is the remaining cost for very small models — measured ~95% host-bound — and now has
an opt-in escape hatch: `TensorRuntime.Capture`/`CapturedGraph.Replay` records a fixed-shape step
once and replays its device launches buffer-to-buffer (~2.5–2.9× on a small step; spike). Still
unoptimized: cross-op kernel fusion. float32-only *storage* by design (so no FP16/BF16 path), but TF32 tensor-core
matmul is an available one-line knob (`CuBlas.MathMode`), currently left off for exact FP32.

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
