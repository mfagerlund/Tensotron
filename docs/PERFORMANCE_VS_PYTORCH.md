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
(An isolated GEMM is compute-bound at all three sizes.)

## What the numbers mean

### Where we tie: raw FP32 GEMM at scale

At 4096³, Tensotron and torch-FP32 both hit ~46 TFLOP/s because **both call cuBLAS
`Sgemm`**. There is no daylight to find here, and that's the point — the matmul core is
not where we lose.

PyTorch's TF32 column is ~1.5x faster. **TF32 is a matmul *math mode*, not a storage dtype** —
inputs stay FP32 in memory but are rounded to 19-bit (8-bit exponent, 10-bit mantissa) and
multiplied on the tensor cores, accumulating back in FP32. So it does **not** conflict with the
float32-only law (storage never changes; nothing is converted at load/save). The knob
(`TensorRuntime.AllowTf32`, mapping to cuBLAS `CUBLAS_TF32_TENSOR_OP_MATH`) raises
Tensotron from 48.7 to 76.1 TFLOP/s at 4096³, matching torch's TF32. Off by default for exact-FP32
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

## Conv bias-gradient reduction

Conv backward dominates the MNIST step, and the **bias gradient is ~98% of conv backward** — it
routes through `ReduceGradToShape`. A naive `ReduceSum` kernel with **one thread per output
element** would run Conv1's 8-channel bias reduction on just *8 threads*, each serially summing
50,176 strided elements.

Instead, a **chunked parallel reduction** splits each output's reduction across many threads that
atomic-combine into a pre-zeroed output, sized so total threads ≈ 8192 regardless of output count.
This keeps the Conv1 bias gradient at ~0.74 ms and the full CNN step at ~5 ms. (Atomic accumulation
makes the result non-bitwise-deterministic — a documented design property; the allocator-pool
transparency test asserts a 1e-3 tolerance rather than bit-equality, which still catches real
corruption by orders of magnitude.)

Conv therefore stays within ~2x of PyTorch on the worst case — same order of magnitude.

## Where the next 1.5–2x lives

The remaining gap on multi-op workloads is **host-side overhead, not algorithm or GPU sync**.
The async stream, caching allocator, cached stride buffers, fused optimizer, parallel reductions,
and cuBLAS path are in place, so the remaining lever is the per-op C# orchestration:

- **Step capture — the biggest small-model lever, and it is in place.** A small model runs the
  *identical* op+buffer sequence every step, yet the autograd graph is rebuilt and a fresh
  `Tensor`/`GradNode` allocated per op every time. A `stepbreakdown` measurement pins the cost: a
  PPO-scale step is **95% host-bound** (2272 µs host-dispatch vs 121 µs device-tail), **36.7 µs/op of
  which is pure autograd-graph construction**. `TensorRuntime.Capture(body)` records the step's device
  launches once; `CapturedGraph.Replay()` re-runs it with no host graph work. Two replay tiers:
  - *Native CUDA graph* (CUDA): the recorded launches — kernels **and** cuBLAS SGEMMs — are folded
    into one executable CUDA graph via driver interop (`cuStreamBeginCapture`/`cuGraphInstantiate`/
    `cuGraphLaunch` through `nvcuda`) on a dedicated owned stream (ILGPU's default stream is the
    uncapturable NULL stream), and replayed with a **single** `cuGraphLaunch`. Measured on a 4090:
    **~5–10× per step vs eager** — a batch-1 control-net step 675 → 64 µs (10.6×), a 256³ cuBLAS step
    1211 → 118 µs (10.2×). This erases nearly all per-launch host dispatch, the floor the software
    tier still pays.
  - *Software replay* (fallback / non-CUDA / `EnableCudaGraph=false`): re-fires the recorded launches
    buffer-to-buffer at the raw ILGPU dispatch floor (~9.75 µs/launch), **~2–2.7×** per step.

  Adam bias correction advances on the device so training stays exact across replays; capture in
  steady state (persistent optimizer/state buffers allocated by a warmup step first), the contract
  PyTorch CUDA graphs also impose. Data-dependent index ops (maxpool argmax, gather) are not
  capturable and throw at capture time rather than corrupting a replay.
- **Strided-batched GEMM (in place).** A constant-stride batched matmul — bmm/attention, and the
  broadcast 2D@3D pattern conv rides — issues **one** `cublasSgemmStridedBatched` for the whole batch
  instead of a per-matrix SGEMM loop (direct cuBLAS P/Invoke, since ILGPU binds only single-matrix
  GEMM). On small-matrix batches where the loop is launch-bound this is **~20–200×** (e.g. batch-128
  64³ bmm 2969 → 15 µs); it collapses to one launch, which also matters under capture/CUDA-graph.
- **cuDNN conv** — out of scope; the fused/autotuned conv path an ILGPU-portable library doesn't chase.

## Verdict

We match PyTorch where it's a fair fight (FP32 GEMM), beat it on small nets, and trail
~1.5–2.5x on conv and large MLPs — and that trailing gap is per-op host overhead plus a TF32
matmul knob left off by default (and cuDNN's conv kernels), **not** a weakness in the math. Same
league. Up there.
