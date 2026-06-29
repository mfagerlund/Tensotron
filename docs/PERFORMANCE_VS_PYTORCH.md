# Performance vs PyTorch

How Tensotron stacks up against PyTorch on the same hardware. The question this
answers: *if we match PyTorch on real workloads, we can claim we're "up there."*
Short version — **we're in the same league: within ~2x everywhere, dead-even on
raw FP32 GEMM at scale, ahead on small MLPs, behind ~1.5–2.5x on conv.**

## Setup

- **GPU:** RTX 4090, unloaded.
- **Tensotron:** this repo, Release build, cuBLAS GEMM path.
- **PyTorch:** 2.5.1+cu121, two baselines that differ *only* in TF32:
  - **strict** — TF32 off everywhere (`allow_tf32 = False` for matmul *and* cuDNN). True FP32, the apples-to-apples comparison for a float32-only library.
  - **default** — PyTorch's out-of-the-box config: FP32 matmul but **cuDNN conv TF32 on**. `cudnn.benchmark = True` in both.
- Both sides report `ms_per_step` (training: fwd + bwd + Adam) or `gflops` (forward GEMM), warmed up, same problem sizes.
- Reproduce: `dotnet run --project examples/Tensotron.Examples -c Release -- ladder` and `python -u tools/bench/torch_bench.py`.

## Results

### MLP — ms/step (lower is better)

| workload | Tensotron | torch FP32 | torch TF32 |
|---|---|---|---|
| small (b256, in32, w128, d2) | **1.22** | 2.04 | 1.47 |
| large (b1024, in128, w512, d2) | 3.34 | **2.04** | **1.40** |

### CNN — MNIST conv net, ms/step (lower is better)

| workload | Tensotron | torch FP32 | torch TF32 |
|---|---|---|---|
| b64 | 4.79 | 2.84 | **1.92** |
| b256 | 5.04 | 2.66 | **1.96** |

Net: `(1,8,3,p1) → ReLU → MaxPool2 → (8,16,3,p1) → ReLU → MaxPool2 → Flatten → Linear(784,64) → ReLU → Linear(64,10)`.

### GEMM — forward, TFLOP/s (higher is better)

| size | Tensotron FP32 | Tensotron TF32* | torch FP32 | torch TF32 |
|---|---|---|---|---|
| 1024³ | 41.7 | 61.5 | 34.8 | 55.4 |
| 2048³ | 48.7 | 74.5 | 42.5 | 62.1 |
| 4096³ | **48.7** | **76.1** | 45.7 | 72.6 |

\*Tensotron TF32 is **opt-in** (`TensorRuntime.AllowTf32 = true`), off by default to keep matmul
exact-FP32. Measured on a clean/unloaded GPU. Tensotron **ties (or marginally beats) torch FP32 at
every size** — both call cuBLAS `Sgemm` — and the TF32 knob recovers torch's TF32 speedup (~1.5–1.6×).
(An earlier draft showed a noisy `~1 TFLOP/s` at 1024³; that was GPU contention, not real — an
isolated GEMM is compute-bound at all three sizes.)

## What the numbers mean

### Where we tie: raw FP32 GEMM at scale

At 4096³, Tensotron and torch-FP32 both hit ~46 TFLOP/s because **both call cuBLAS
`Sgemm`**. There is no daylight to find here, and that's the point — the matmul core is
not where we lose.

PyTorch's TF32 column is ~1.5x faster. **TF32 is a matmul *math mode*, not a storage dtype** —
inputs stay FP32 in memory but are rounded to 19-bit (8-bit exponent, 10-bit mantissa) and
multiplied on the tensor cores, accumulating back in FP32. So it does **not** conflict with the
float32-only law (storage never changes; nothing is converted at load/save). The knob is now wired
(`TensorRuntime.AllowTf32`, mapping to cuBLAS `CUBLAS_TF32_TENSOR_OP_MATH`); **measured**, it takes
Tensotron from 48.7 → 76.1 TFLOP/s at 4096³, matching torch's TF32. Off by default for exact-FP32
parity; flip it for a free ~1.5× on matmul-bound work. (cuBLAS's *legacy* `TensorOpMath` flag is
faster still, ~134 TFLOP/s, but it down-converts to FP16-class precision — avoid it for training.)
*True* 16-bit (FP16/BF16) **storage** — 2x bandwidth on top of tensor-core throughput — **is**
excluded by float32-only and is a major project (loss scaling, FP32 master weights, dtype plumbing
through every op and fixture). That, not TF32, is the "use 16-bit throughout, convert at the edges"
model.

### Where we win: small MLPs

At b256/w128 Tensotron is *faster* than PyTorch (1.22 vs 1.47–2.04 ms). At tiny sizes the
work is dominated by per-step launch/dispatch overhead, and our overhead is lower than
PyTorch's Python op-dispatch.

### Where we lose: large MLP, and conv — and *why*

Two distinct causes, both known and both fixable:

**1. Per-op *host* overhead, in aggregate.** This is *not* a per-launch GPU sync (the runtime is
async — it drains only at host pulls plus a safety valve every 64 launches) and *not* a per-GEMM
tax (an isolated GEMM is compute-bound at every size, per the table). The cost is **host-side, per
op, multiplied by the op count of a full step**: each op allocates a fresh `Tensor` (+ `GradNode` in
grad mode), recomputes broadcast bookkeeping, and dispatches a launch from C#. A training step is
*dozens* of small ops (matmul, bias-add, ReLU, pool, the whole backward, Adam) plus an autograd
graph rebuilt every step. On a small net the GPU math is microseconds, so this orchestration is the
step — ~97% of the 3.34 ms large-MLP step is host work, not GEMM compute, which is why it loses 1.6x
even though each of its GEMMs ties in isolation. This is the lever for small models (trace/replay,
below), and it's the same cost that would dominate **CPU inference** of a tiny net.

**2. Conv vs cuDNN.** Our conv is `im2col → batched GEMM → col2im`. cuDNN ships fused,
autotuned conv kernels (and, in the default config, runs them on tensor cores). We will not
match that at FP32. We don't have to be *as fast* to be "up there" — but this is a real
~1.7–1.9x gap (FP32) / ~2.5x (cuDNN-TF32).

## The big win that got us here

The CNN was **~9x slower than PyTorch** before this work. Root cause (5-whys): the MNIST
step was 98% backward → conv backward was the whole thing → **bias gradient was 98% of conv
backward** → it routed through `ReduceGradToShape` → the naive `ReduceSum` kernel used **one
thread per output element**. Conv1's 8-channel bias reduction therefore ran *8 threads*,
each serially summing 50,176 strided elements.

Fix: a **chunked parallel reduction** — split each output's reduction across many threads
that atomic-combine into a pre-zeroed output, sized so total threads ≈ 8192 regardless of
output count. Conv1 bias gradient: **22.8 ms → 0.74 ms**. Full CNN step:
**28.8 ms → ~5 ms (~6–10x).** (Atomic accumulation makes the result non-bitwise-deterministic
— already a documented design property; the allocator-pool transparency test was relaxed
from bit-equality to a 1e-3 tolerance, which still catches real corruption by orders of
magnitude.)

This is *why* the honest answer to "are we up there?" flipped from "no, 9x off on conv" to
"yes, same order of magnitude, ~2x on the worst case."

## Where the next 1.5–2x lives

The remaining gap on multi-op workloads is **host-side overhead, not algorithm or GPU sync**.
The easy wins are already in (async stream, caching allocator, cached stride buffers, fused
optimizer, parallel reductions, cuBLAS), so what's left is the per-op C# orchestration:

- **Trace/replay the fixed-shape step (biggest attainable lever — landed as a spike).** A small
  model runs the *identical* op+buffer sequence every step, yet we rebuild the autograd graph and
  allocate a fresh `Tensor`/`GradNode` per op every time. A direct `stepbreakdown` measurement
  pinned the cost: a PPO-scale step is **95% host-bound** (2272 µs host-dispatch vs 121 µs
  device-tail), **36.7 µs/op of which is pure autograd-graph construction**. `TensorRuntime.Capture`
  now records the step's device launches once and `CapturedGraph.Replay` re-fires them
  buffer-to-buffer with no host graph work (a software CUDA-Graph). **Measured ~2.5–2.9× faster per
  step**, landing replay at the raw ILGPU dispatch floor (~9.75 µs/launch). Confirms host overhead —
  not math — is the small-model tax. Productionization (buffer reclamation, step-dependent optimizer
  scalars, conv/pool ops, PPO wiring) is follow-up.
- **CUDA Graph capture (the endgame, but blocked).** Capture forward+backward+optimizer once and
  replay with a single driver call → per-launch host cost goes to ~zero. **ILGPU 1.5.3 exposes no
  graph-capture API** (verified — zero `cuGraph`/`BeginCapture` symbols in the DLL), so this needs
  raw CUDA driver interop, not a flag flip.
- **Conv as a single large GEMM** — fold the batch into one im2col matrix instead of a per-matrix
  cuBLAS loop, to amortize launch overhead. Won't catch cuDNN; narrows the gap.
- **TF32 matmul math mode (nearly free, not yet enabled)** — `CuBlas.MathMode = TensorOpMath`
  routes our FP32 SGEMM through the tensor cores for ~the torch-TF32 speedup, with FP32 storage
  unchanged. ~1.5x on matmul-bound work for one line; doesn't help host-bound small models.
- **cuDNN conv** — out of scope; the fused/autotuned conv path an ILGPU-portable library doesn't chase.

## Verdict

We match PyTorch where it's a fair fight (FP32 GEMM), beat it on small nets, and trail
~1.5–2.5x on conv and large MLPs — and that trailing gap is per-op host overhead plus a TF32
matmul knob we haven't flipped (and cuDNN's conv kernels), **not** a weakness in the math. Same
league. Up there.
