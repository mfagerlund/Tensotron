## Critical

1. `src/Tensotron/Tensor.cs:119` / `src/Tensotron/Tensor.cs:127` / `src/Tensotron/Tensor.cs:282`: `DisposeGraph()` can recycle the live root output when the root is a zero-copy view.

   `DisposeGraph()` promises to leave `this` intact, but it only skips disposing the root tensor object. If `this` is a view (`Reshape()` creates `OwnsBuffer = false` over its input buffer), traversal visits the owning interior input and disposes it at line 127. The root view then points at a buffer that has been returned to the allocator pool and may be overwritten by a later allocation. This is reachable from normal public ops: 1D dot, `Mv`, `Conv2d`'s final reshape, and many `Squeeze`/`Flatten` paths produce view roots.

   Why this diverges: PyTorch's non-retained backward frees graph internals but does not invalidate the returned loss/output tensor's storage. Here, `loss.DisposeGraph(); loss.Item()` can read recycled memory.

   Suggested fix: track storage ownership at graph-disposal time. Do not dispose any tensor whose buffer is shared with `this` or with a still-live graph output. A simple conservative fix is to collect the root buffer and skip any reachable tensor with `ReferenceEquals(t.Buffer, this.Buffer)`. A more robust fix is explicit storage ref-count/owner tracking for views.

## Correctness/Parity

1. `src/Tensotron/TensorOps.Binary.cs:21` / `src/Tensotron/TensorOps.Binary.cs:24` / `src/Tensotron/TensorOps.Binary.cs:26`: `Pow` backward is wrong at zero bases.

   The base gradient is implemented as `g * b * res / a`. For `a == 0`, this gives `0/0 => NaN` for common torch cases such as `pow(0, 2)` and `pow(0, 3)`, where PyTorch returns base grad `0`. For `pow(0, 1)`, PyTorch returns base grad `1`, while this formula again produces `NaN`. The exponent gradient `g * res * log(a)` also gives `0 * -inf => NaN` at `a == 0`, while PyTorch returns exponent grad `0`.

   Suggested fix: implement a dedicated `PowBackward` kernel that matches torch's zero-base cases instead of algebraically dividing by `a`. At minimum special-case `a == 0` for `b > 1`, `b == 1`, `0 < b < 1`, `b == 0`, and exponent-grad handling.

2. `src/Tensotron/TensorOps.Reduce.cs:54` / `src/Tensotron/TensorOps.Reduce.cs:60` / `src/Tensotron/TensorOps.Reduce.cs:69` / `src/Tensotron/Kernels.cs:405` / `src/Tensotron/Kernels.cs:455` / `src/Tensotron/CpuKernels.cs:225`: `Max`/`Min` gradients route ties to one element for all reduction forms.

   `MaxMin` accepts `dims == null` and `int[]` multi-axis reductions, but backward always calls `LaunchReduceArgGrad`, whose kernels use strict comparison and assign the full gradient to the first winning element. That matches `torch.max(x, dim).values` / `torch.min(x, dim).values`, but not `torch.max(x)` over all elements, nor `torch.amax`/`torch.amin` over one or more dims. Those split gradient evenly across all equal extrema.

   Example: for `x = [2, 2]`, `torch.max(x).backward()` gives `[0.5, 0.5]`; this code gives `[1, 0]`.

   Suggested fix: split the API internally. For a single explicit `dim` meant to model `torch.max(dim)`, first-index routing is correct. For whole-tensor or multi-dim reductions, count tied extrema in each group and distribute `g / tieCount` to every tied element, matching `torch.max(x)` and `torch.amax` semantics.

3. `src/Tensotron/TensorOps.Losses.cs:112` / `src/Tensotron/TensorOps.Losses.cs:118`: `KlDiv` forward handles `target == 0`, but target gradients diverge from PyTorch.

   The implementation avoids `0 * log(0)` by computing `target * (log(maximum(target, float.Epsilon)) - input)`. That makes the forward zero at `target == 0`, but if `target.RequiresGrad` is true, the gradient is finite (`log(float.Epsilon) - input`) because the outer multiply still contributes. PyTorch `torch.nn.functional.kl_div(..., log_target=False)` returns `target.grad == NaN` at exactly zero target.

   Suggested fix: if target gradients are supported, add a dedicated KL-div op/backward that preserves PyTorch's special value behavior. If differentiable targets are intentionally unsupported, reject `target.RequiresGrad` explicitly rather than returning a plausible but non-torch gradient.

## Resource/Lifetime

1. `src/Tensotron/TensorRuntime.cs:1276` / `src/Tensotron/TensorRuntime.cs:1279` / `src/Tensotron/TensorOps.Pool.cs:29` / `src/Tensotron/TensorOps.Pool.cs:34`: training `MaxPool2d` leaks the device argmax buffer deterministically.

   `LaunchMaxPool2d` allocates an `int` device buffer for argmax and returns it when gradients are needed. The backward closure captures it, but no code disposes it after backward or from `DisposeGraph()`. `DisposeGraph()` only recycles `Tensor` float buffers, not opaque auxiliary buffers captured by closures. The no-grad path explicitly disposes the argmax buffer at `src/Tensotron/TensorRuntime.cs:1280`, which highlights the missing training-path owner.

   Suggested fix: wrap the argmax handle in a disposable owner attached to the grad node/forward tensor and dispose it after backward or during graph disposal. Alternatively materialize argmax as a `TensorStorage`-like tracked auxiliary buffer that participates in the same lifetime management.

2. `src/Tensotron/Tensor.cs:100` / `src/Tensotron/Tensor.cs:108` / `src/Tensotron/Tensor.cs:233`: disposing an owning tensor while detached/reshaped views are live is documented as caller error, but the public API gives no guard.

   This is not necessarily a parity bug, but it is a sharp lifetime hazard for a library that exposes `IDisposable`: `Detach()` returns a non-owning view, and disposing the source returns the shared buffer to the pool. The next allocation can mutate the detached tensor's contents.

   Suggested fix: either document this prominently in API docs/examples or add storage reference tracking so owned buffers are not pooled until all views are dead/disposed.

## Concurrency/Capture

1. `src/Tensotron/Module.cs:140` / `src/Tensotron/Module.cs:144` / `src/Tensotron/Module.cs:146` / `src/Tensotron/TensorRuntime.cs:580` / `src/Tensotron/TensorRuntime.cs:600`: `Dropout` inside capture replays the same random mask forever.

   Capture records device launches, not host-side random generation or `Tensor.FromShaped` uploads. `Dropout.Forward` creates a fresh host `float[]` mask and uploads it once during capture, then the captured graph only replays the multiply using that pinned mask buffer. Every replay uses the capture-time dropout mask.

   Why this diverges: PyTorch dropout in training samples a fresh mask per forward. PyTorch CUDA graph capture requires graph-safe RNG state handling; this implementation has neither a device RNG nor a capture rejection for dropout.

   Suggested fix: make `Dropout.Forward` fail fast when `TensorRuntime` is capturing, or implement a capturable device RNG/mask update whose state advances on replay. For inference/eval dropout is fine because it returns `x`.

2. `src/Tensotron/TensorRuntime.cs:727` / `src/Tensotron/TensorRuntime.cs:730` / `src/Tensotron/TensorRuntime.cs:732`: unsupported data-dependent index launches fail during capture as intended.

   This is a good guard, but note that the failure occurs after some work and temporary buffers may already have been allocated by the attempted op. I did not verify a persistent leak on failed capture; mark this as residual risk, not a confirmed defect.

## Minor/Style

1. `src/Tensotron/TensorOps.Reduce.cs:144` / `src/Tensotron/TensorOps.Reduce.cs:149`: `Argmax`/`Argmin` return indices in a float tensor.

   This is understandable given the float32-only design, but it is not PyTorch's dtype behavior (`torch.long`). Code that mechanically ports torch indexing must remember this library-specific convention.

   Suggested fix: keep the convention if dtype support is out of scope, but make every index-taking API error message mention the float-index convention and integral-value requirement.

## Things that look correct

- `src/Tensotron/Tensor.cs:318` / `src/Tensotron/Tensor.cs:322` / `src/Tensotron/Tensor.cs:332`: backward clears interior grads while preserving leaf accumulation, matching the stated torch-retain behavior.
- `src/Tensotron/Tensor.cs:328`: non-scalar backward without an explicit gradient is rejected instead of silently summing.
- `src/Tensotron/TensorOps.Binary.cs:34` / `src/Tensotron/TensorOps.Binary.cs:46`: elementwise `Maximum`/`Minimum` tie gradients split 0.5/0.5, matching torch for binary max/min.
- `src/Tensotron/Ops.Unary.cs:130` / `src/Tensotron/Ops.Unary.cs:135`: `Clamp` uses the closed-interval gradient at bounds, avoiding the common composed-min/max 0.5 bug.
- `src/Tensotron/Ops.Unary.cs:101` / `src/Tensotron/Ops.Unary.cs:104` and `src/Tensotron/Ops.Unary.cs:109` / `src/Tensotron/Ops.Unary.cs:114`: ReLU and LeakyReLU kink gradients appear intentionally torch-matched.
- `src/Tensotron/CpuKernels.cs:401` / `src/Tensotron/CpuKernels.cs:405` / `src/Tensotron/CpuKernels.cs:415`: opt-in row-parallel CPU matmul splits disjoint output row ranges and uses the same per-row reduction routine as serial, so the "bit-identical to serial" claim looks sound.
