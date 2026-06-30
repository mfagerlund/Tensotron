# Performance vs PyTorch

How Tensotron stacks up against PyTorch on the same hardware. The question this
answers: *if we match PyTorch on real workloads, we can claim we're "up there."*
Short version — **we're in the same league: dead-even on raw FP32 GEMM at scale,
~1.9× ahead on a small-batch MLP step, ~par on a large MLP, and ~2–3× behind on conv.**

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
| small (b256, in32, w128, d2) | **0.71** | 1.32 | 1.38 |
| large (b1024, in128, w512, d2) | 1.62 | 1.37 | **1.35** |

### CNN — MNIST conv net, ms/step (lower is better)

| workload | Tensotron | torch FP32 | torch TF32 |
|---|---|---|---|
| b64 | 3.56 | **1.86** | 1.89 |
| b256 | 6.22 | 1.89 | **1.85** |

Net: `(1,8,3,p1) → ReLU → MaxPool2 → (8,16,3,p1) → ReLU → MaxPool2 → Flatten → Linear(784,64) → ReLU → Linear(64,10)`.

### GEMM — forward, TFLOP/s (higher is better)

| size | Tensotron FP32 | Tensotron TF32* | torch FP32 | torch TF32 |
|---|---|---|---|---|
| 1024³ | 41.6 | 55.9 | 42.2 | 64.5 |
| 2048³ | 50.1 | 69.5 | 50.3 | 72.2 |
| 4096³ | **53.2** | 72.0 | 53.0 | **80.3** |

\*Tensotron TF32 is **opt-in** (`TensorRuntime.AllowTf32 = true`), off by default to keep matmul
exact-FP32. Measured on a clean/unloaded GPU. Tensotron FP32 is **dead even with torch FP32 at
every size** — both call cuBLAS `Sgemm` — and the TF32 knob recovers most of the tensor-core speedup
(~1.35× for us at 4096³, ~1.5× for torch). (An isolated GEMM is compute-bound at all three sizes.)

## What the numbers mean

### Where we tie: raw FP32 GEMM at scale

At 4096³, Tensotron and torch-FP32 both hit ~53 TFLOP/s because **both call cuBLAS
`Sgemm`**. There is no daylight to find here, and that's the point — the matmul core is
not where we lose.

PyTorch's TF32 column is ~1.5x faster. **TF32 is a matmul *math mode*, not a storage dtype** —
inputs stay FP32 in memory but are rounded to 19-bit (8-bit exponent, 10-bit mantissa) and
multiplied on the tensor cores, accumulating back in FP32. So it does **not** conflict with the
float32-only law (storage never changes; nothing is converted at load/save). The knob
(`TensorRuntime.AllowTf32`, mapping to cuBLAS `CUBLAS_TF32_TENSOR_OP_MATH`) raises
Tensotron from 53.2 to 72.0 TFLOP/s at 4096³, most of the way to torch's TF32 (80.3). Off by default
for exact-FP32 parity; flip it for a free ~1.35× on matmul-bound work. (cuBLAS's *legacy* `TensorOpMath` flag is
faster still, ~134 TFLOP/s, but it down-converts to FP16-class precision — avoid it for training.)
*True* 16-bit (FP16/BF16) **storage** — 2x bandwidth on top of tensor-core throughput — **is**
excluded by float32-only and is a major project (loss scaling, FP32 master weights, dtype plumbing
through every op and fixture). That, not TF32, is the "use 16-bit throughout, convert at the edges"
model.

### Where we win: small MLPs

At b256/w128 Tensotron is *faster* than PyTorch (0.71 vs ~1.3 ms — ~1.9×). At tiny sizes the
work is dominated by per-step launch/dispatch overhead, and our overhead is lower than
PyTorch's Python op-dispatch.

### Where we lose: conv — and the host-overhead story

Two causes, both known and both being chipped away:

**1. Per-op *host* overhead, in aggregate.** This is *not* a per-launch GPU sync (the runtime is
async — it drains only at host pulls plus a safety valve every 64 launches) and *not* a per-GEMM
tax (an isolated GEMM is compute-bound at every size, per the table). The cost is **host-side, per
op, multiplied by the op count of a full step**: each op allocates a fresh `Tensor` (+ `GradNode` in
grad mode), recomputes broadcast bookkeeping, and dispatches a launch from C#. A training step is
*dozens* of small ops (matmul, bias-add, ReLU, the whole backward, Adam) plus an autograd graph
rebuilt every step. On a small net the GPU math is microseconds, so this orchestration *is* the step.
De-overheading work (fused loss epilogue, a steady-state caching allocator, cached shape/stride
uploads) has already roughly halved the MLP steps since these numbers were first taken — the large
MLP is now **~par** with torch (1.62 vs 1.37 ms) rather than 1.6× behind. The remaining host floor is
what **step capture** erases: recording the fixed-shape step once and replaying it as a single
`cuGraphLaunch` is ~6–8× per step at control-net/PPO scale (see below). It's the same cost that would
dominate **CPU inference** of a tiny net.

**2. Conv vs cuDNN.** Our conv is `im2col → batched GEMM → col2im`. cuDNN ships fused,
autotuned conv kernels. We will not match that at FP32 — it's a real ~1.9× gap at b64 widening to
~3.3× at b256. (On these small MNIST-scale convs cuDNN's TF32 path buys nothing — torch's strict and
default configs tie — so the gap is FP32-structural, not a tensor-core deficit.)

## Conv bias-gradient reduction

Conv backward dominates the MNIST step, and the **bias gradient is ~98% of conv backward** — it
routes through `ReduceGradToShape`. A naive `ReduceSum` kernel with **one thread per output
element** would run Conv1's 8-channel bias reduction on just *8 threads*, each serially summing
50,176 strided elements.

Instead, a **chunked parallel reduction** splits each output's reduction across many threads that
atomic-combine into a pre-zeroed output, sized so total threads ≈ 8192 regardless of output count.
This keeps the Conv1 bias gradient at ~0.74 ms and the full CNN step at ~3.6–6 ms (b64–b256). (Atomic accumulation
makes the result non-bitwise-deterministic — a documented design property; the allocator-pool
transparency test asserts a 1e-3 tolerance rather than bit-equality, which still catches real
corruption by orders of magnitude.)

Conv therefore stays within ~2–3.4× of PyTorch (b64→b256) — same order of magnitude.

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
    uncapturable NULL stream), and replayed with a **single** `cuGraphLaunch`. Measured on a 4090
    (`... -- replay`): **~6–8× per step vs eager** — a batch-1 control-net step 1252 → 149 µs (8.4×),
    a batch-512 PPO step 1290 → 225 µs (5.7×), a b256 MLP step 1609 → 242 µs (6.6×). This erases nearly
    all per-launch host dispatch, the floor the software tier still pays.
  - *Software replay* (fallback / non-CUDA / `EnableCudaGraph=false`): re-fires the recorded launches
    buffer-to-buffer at the raw ILGPU dispatch floor (~9.75 µs/launch), **~2–2.7×** per step.

  Adam bias correction advances on the device so training stays exact across replays; capture in
  steady state (persistent optimizer/state buffers allocated by a warmup step first), the contract
  PyTorch CUDA graphs also impose. A host scalar baked into a captured kernel is frozen at capture, so
  a scheduled learning rate would otherwise freeze — the **capturable optimizer mode** (`new Adam(…,
  capturable: true)`, also `AdamW`/`Sgd`, mirroring torch's `capturable=True`) keeps the LR in a device
  scalar that `optimizer.LearningRate = …` uploads, so an LR set between replays is honoured by the next
  one. Data-dependent index ops (maxpool argmax, gather) are not capturable and throw at capture time
  rather than corrupting a replay.
- **Strided-batched GEMM (in place).** A constant-stride batched matmul — bmm/attention, and the
  broadcast 2D@3D pattern conv rides — issues **one** `cublasSgemmStridedBatched` for the whole batch
  instead of a per-matrix SGEMM loop (direct cuBLAS P/Invoke, since ILGPU binds only single-matrix
  GEMM). On small-matrix batches where the loop is launch-bound this is **~20–200×** (e.g. batch-128
  64³ bmm 2969 → 15 µs); it collapses to one launch, which also matters under capture/CUDA-graph.
- **cuDNN conv** — out of scope; the fused/autotuned conv path an ILGPU-portable library doesn't chase.

## Verdict

We match PyTorch where it's a fair fight (FP32 GEMM), beat it on small-batch MLP steps, sit ~par on
large MLPs, and trail ~2–3× on conv — and that trailing gap is per-op host overhead plus a TF32
matmul knob left off by default (and cuDNN's conv kernels), **not** a weakness in the math. Same
league. Up there.
