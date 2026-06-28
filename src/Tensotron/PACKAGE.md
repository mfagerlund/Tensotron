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

`0.1.0-alpha`. **Correctness-first:** matmul currently runs on custom ILGPU kernels (a cuBLAS
swap is planned) and the runtime synchronizes after essentially every launch rather than
queueing on the async stream. Buffers are `IDisposable` (deterministic opt-in release);
zero-copy views never free their parent's buffer. A pooling/arena allocator and async-stream
execution are on the roadmap.

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
