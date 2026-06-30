# Request: mutable per-replay scalars in a captured graph (LR first)

**From:** Evolvatron.Walker (PPO trainer, `..\Evolvatron.Walker`) · **Date:** 2026-06-30
**Against:** native CUDA-graph capture (`CapturedGraph`, `TensorRuntime.Capture`).

> **Status: shipped (LR).** Opt in with `new Adam(ps, lr, capturable: true)` (also `AdamW`, `Sgd`),
> mirroring torch's `capturable=True`. The fused optimizer kernels read LR from a device scalar; set
> `optimizer.LearningRate = …` between replays (it `Upload`s the scalar) and the next `Replay()` honours
> it — so grad-clip + `opt.Step()` can live **inside** the captured body. Verified by
> `TraceReplayTests.Captured_step_honours_live_learning_rate_per_replay` (lr=0 invariant: an exactly
> unchanged param proves the rate is read live, not frozen) and `OptimizerCapturableTests` (the
> capturable kernels are bit-for-bit the by-value path under an LR schedule, both backends). The other
> §4 scalars (entropy coef, clip threshold) are not yet device-resident — the same mechanism applies
> when needed. See `docs/PERFORMANCE_VS_PYTORCH.md`.

## TL;DR of what I want

Let me **fold the optimizer step (and grad clip) into the captured graph while still changing the
learning rate every iteration.** Today I can't: the LR is baked into the graph at capture time. I want
the fused optimizer kernels to read LR (and any other anneal-per-iteration scalar) from a **stable
device buffer I can `Upload` between replays**, never as a by-value launch argument.

Ship the LR fix first (small, high value). Then apply the same treatment to every "similar" frozen
scalar listed in §4.

## 1. Why this matters / measured stakes

I benchmarked Walker's PPO update on a 4090 with the new native graph (writeup:
`..\Evolvatron.Walker\docs\artifacts\cuda-graph-capture-perf.md`). The captured **fwd+bwd alone** went
~3.1× faster than the old ILGPU software replay (and ~9.5× vs eager). But in the **real training path**
the win collapses to **1.15–1.67×**, because grad-clip + `opt.Step()` run **eagerly outside** the graph
(I have to keep them out so the annealed LR and Adam bias-correction keep advancing). Once fwd+bwd is a
single `cuGraphLaunch`, that out-of-graph eager work dominates the step.

If LR can vary across replays, I can put clip + step **inside** the captured body and recover the full
~3× (≈9.5× vs eager) on every minibatch. The blocker is exactly the LR scalar.

## 2. Root cause (concrete, with file:line)

The fused optimizer kernels take LR **by value** and the capture thunk closes over it:

```csharp
// TensorRuntime.cs:1288  LaunchAdam(..., float lr, float eps, ..., float coupledWd, float decoupledFactor)
_adamStep(_stream, (int)p.Length, p.View, g.View, m.View, v.View,
    b1, oneMinusB1, b2, oneMinusB2, lr, eps, bc, coupledWd, decoupledFactor);
if (_capture != null) _capture.Add(() => _adamStep(_stream, ..., lr, eps, bc, ...));  // lr captured BY VALUE
```

```csharp
// Optimizers.cs:140  Adam.Step()
rt.LaunchAdam(p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
    _b1, 1f - _b1, _b2, 1f - _b2, LearningRate, _eps, s.adv.Buffer, ...);  // LearningRate is a host float
```

For host-side software replay the closure holds the frozen `lr`; for the native graph the value is
baked into the kernel-node params. Either way `Optimizer.LearningRate = newLr` between replays has **no
effect**. Contrast: Adam bias-correction already advances correctly across replays *because* it is read
from a device buffer (`advS`, written by `LaunchAdvanceAdam`, `TensorRuntime.cs:1278`) rather than passed
by value. That is the pattern to copy.

**The general rule (please state it in the `Capture` XML doc):** a scalar consumed inside a captured
step is frozen at capture **iff** it is a by-value launch argument or a `Scalar(v)` literal materialized
during capture. A scalar **read from a persistent device buffer** is *not* frozen — an `Upload` to that
buffer between replays is honoured by the next `cuGraphLaunch` (same mechanism as the minibatch input
tensors, `Tensor.cs:247`). So: **anything that must vary across replays has to live in a device buffer,
not in a kernel argument.**

## 3. Requested API — the LR fix (do this first)

Make the optimizer's LR a persistent device scalar that the fused kernel reads, and that `Step()`
refreshes from the host `LearningRate` each call via a tiny H2D copy.

1. **Kernel:** add a device-scalar LR path to `AdamStep`/`SgdStep`. Either an overload that takes a
   `lrBuf` `ArrayView<float>` (length 1) read inside the kernel, or reuse/extend the existing `adv`
   buffer to also carry LR. Keep the `float lr` overload for the eager/non-captured path.
2. **Runtime:** `LaunchAdam(..., TensorStorage lrBuf, ...)` reads `lr` from `lrBuf.View[0]` instead of a
   by-value arg; the capture thunk closes over the **buffer**, not the value (exactly like `advS`).
3. **Optimizer:** `Adam`/`AdamW`/`Sgd` hold a persistent `Tensor _lrDevice = Tensor.FromArray(new[]{lr},1)`.
   At the top of `Step()`, `_lrDevice.Upload(new[]{ LearningRate })` (this H2D copy is **not** part of the
   captured body — it runs each iteration before `Replay()`, just like feeding the next minibatch).
   AdamW's `decoupledFactor = 1 − lr·wd` is LR-derived, so it must be computed **on-device** inside the
   kernel from the LR scalar + a constant `wd` (don't precompute it on the host — that re-freezes it).
4. **Behavioural contract:** with the optimizer step captured, calling `opt.LearningRate = x` (or stepping
   an `LrScheduler`) then `Replay()` must apply rate `x` on that replay, matching an eager loop that set
   the same rate — to 1e-6.

This is the minimum that unblocks me. It's narrow: LR buffer + one kernel arg change + an `Upload` in
`Step()`.

## 4. "Anything similar" — every other scalar that freezes the same way

Please audit and give the same device-buffer treatment (or document the workaround) for each scalar that
a caller might reasonably want to **anneal or otherwise change per iteration** while it sits inside a
captured graph:

| Scalar | Where it freezes | Changes per-iter in practice? | Fix |
|---|---|---|---|
| **Adam/AdamW LR** | `LaunchAdam` by-value `lr` (`TensorRuntime.cs:1288`) | **Yes** (LR anneal) | §3 — device scalar |
| **AdamW decoupled factor** `1−lr·wd` | host-precomputed (`Optimizers.cs:212`) | **Yes** (LR-derived) | compute on-device from LR scalar |
| **SGD LR** | `LaunchSgd` by-value `lr` (`TensorRuntime.cs:1304`) | Yes (LR anneal) | device scalar, same as Adam |
| **grad-clip `maxNorm`/`eps`** | `Scalar(maxNorm)` literal in `GradUtils.ClipGradNorm` (`Optimizers.cs:333`) | rarely | mutable-scalar input (§5) if annealed; else fine to leave frozen — just **document** it |
| **PPO loss coefficients** (entropy, value, clip-ε) | `Scalar(coef)` literals built into the loss inside the captured body | sometimes (entropy/clip anneal) | mutable-scalar input (§5) |
| Adam betas / eps, weight decay, gamma/momentum | by-value, but constant for a run | No | leave frozen; **document** as capture-frozen |

The constants (betas, eps, fixed wd) are genuinely fine to bake — I just want the freeze behaviour
**documented** so callers know which knobs die at capture and which stay live.

## 5. Nice-to-have — a first-class mutable-scalar input

For the loss-coefficient cases (entropy/clip-ε anneal), a small helper would make this clean instead of
me hand-rolling a 1-element tensor and remembering to `Upload` it:

```csharp
var entCoef = rt.ScalarInput(0.01f);          // persistent, host-writable device scalar (0-d or [1])
// ... inside the captured body: loss = pg + vf - entCoef * entropy;   // broadcast-mul, reads the buffer
// ... between replays, outside the body:
entCoef.Upload(new[]{ annealedCoef });          // honoured by the next Replay()
```

i.e. expose `Scalar()` as a **named, persistent, host-writable** input (it's currently `internal` and
throwaway, `TensorOps.Binary.cs:6`). If broadcasting a `[1]`/0-d tensor through the existing ops already
works inside a captured body, this may need no kernel work at all — just the public factory + a one-line
guarantee in the docs that uploads to it survive replay. Please confirm that path works (or say what
breaks).

## 6. Acceptance test I'd like to see (in your suite)

A single test that proves the whole point end-to-end:

1. Build a tiny MLP + `Adam`. **Capture** a full step body: forward → loss → backward →
   `ClipGradNorm(returnTotalNorm:false)` → `opt.Step()`.
2. Replay N=20 times, **changing LR each replay** (e.g. cosine schedule) via `opt.LearningRate = …`
   (which uploads the LR scalar). Feed a fixed input each time so only LR varies.
3. Run the **same** sequence eagerly (no capture), same LR schedule, same input.
4. Assert the parameter tensors match between captured-replay and eager to **1e-6** after all N steps,
   and assert `graph.UsesNativeGraph == true` on CUDA.

That test failing today (params frozen to the capture-time LR) is exactly my blocker; passing it is
"done."

## 7. Notes / non-asks

- The per-iteration LR **`Upload` stays outside the captured body** — I'm not asking to capture the H2D
  copy, just to have the kernel read LR from the buffer the copy writes. (Same shape as minibatch feed.)
- I'm **not** asking to make betas/eps/weight-decay mutable — those are fine frozen; just document it.
- Walker references Tensotron by `ProjectReference`, so a rebuild picks this up with no packaging step.
  I'll re-run the §6-style A/B (`bench=run nativegraph=0|1`, clip+step in vs out of graph) and update
  `..\Evolvatron.Walker\docs\artifacts\cuda-graph-capture-perf.md` once this lands.
