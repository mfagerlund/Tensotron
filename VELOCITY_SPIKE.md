# Spike: ILGPU Velocity (SIMD CPU backend) — NEGATIVE RESULT

Branch: `spike/ilgpu-velocity`. Date: 2026-06-28. Verdict: **not usable as-is.**

## Question
The user asked whether we could "skip NuGet and target GitHub if it works and is fast" — i.e.
reference ILGPU from master source (which contains the **Velocity** SIMD CPU accelerator that ships
in no released package, through 1.5.3) and run Tensotron on it for a fast CPU path.

## What works
- **ILGPU master builds from source cleanly** (`Src/ILGPU` + `Src/ILGPU.Algorithms`, net8.0, 0 warnings).
  Target framework matches Tensotron exactly, so the repoint is a one-line csproj swap
  (`PackageReference` → `ProjectReference`).
- **Velocity is reachable and selectable.** Wiring is trivial: `b.Default().Velocity().EnableAlgorithms()`,
  add `using ILGPU.Runtime.Velocity;`, select `AcceleratorType.Velocity`. Velocity enables Scalar2 +
  **Vector128** lanes (128-bit max — SSE-level, *not* AVX2/256), 64-bit only.
- **Tensotron core compiles against master with zero API drift.**
- **Pure elementwise / unary ops run correctly on Velocity** (UnaryTests pass).

## What breaks (the blocker)
Any kernel that decomposes a linear index over tensor dims with integer `%` / `/` (reduce, batched
matmul, gather/scatter, im2col, pooling — most of the non-elementwise kernels) throws
`System.DivideByZeroException` on Velocity.

### Root cause
ILGPU's Velocity backend is a **managed** SIMD backend (`System.Numerics.Vector<T>` over `Parallel.For`).
`Vector<int>` has **no** integer division, so ILGPU **scalarizes** `%`/`÷` lane-by-lane. The kernel
launch is **padded to the vector width**, and **masked memory loads return 0 for the inactive padding
lanes**. So a padding lane loads `dim = 0` and the unconditional scalar divide `idx % dim` / `rem / dim`
divides by zero — even though that lane's result is discarded.

### Why the obvious fix doesn't work
Guarding the divisor at the source — `int od = outDims[ax] < 1 ? 1 : outDims[ax];` — does **not**
prevent the throw. Velocity evaluates the division on the raw masked-zero value regardless of the
conditional select. The integer-division-under-masking problem is below the C# source level; it lives
in ILGPU's Velocity lowering.

## Cost to actually make it work
One of:
1. **Upstream ILGPU fix** — make Velocity's integer-division lowering force inactive-lane divisors to 1
   (mask the divisor, not just the result). The correct fix, but it's an ILGPU PR + release cycle.
2. **Rewrite every index-decomposition kernel** to avoid integer division on data that can be zero for
   masked lanes (e.g. precomputed reciprocal-free addressing). Invasive, touches ~15 kernels, risks the
   PyTorch-parity correctness the library is built on, for a backend capped at 128-bit managed SIMD.

Neither is justified right now. Even if fixed, Velocity is managed `Vector128` — a modest win over the
scalar CPUAccelerator, not a GPU substitute. The **CPU-rollout split** (snapshot weights → launch-free
host inference for PPO rollouts) already removed the dominant CPU cost for the actual workload, without
touching ILGPU.

## Recommendation
Stay on **NuGet ILGPU 1.5.3** (CUDA + scalar CPU). Keep this branch as the record. Revisit Velocity only
if ILGPU ships it in a release with the integer-division lowering fixed, or if a profiled CPU bottleneck
makes a managed-SIMD path worth the kernel-rewrite cost.

## Repro
```
git checkout spike/ilgpu-velocity   # csproj points at ../../../ILGPU/Src (clone master there first)
TENSOTRON_BACKEND=velocity dotnet test tests/Tensotron.Tests --filter "FullyQualifiedName~UnaryTests"      # PASS
TENSOTRON_BACKEND=velocity dotnet test tests/Tensotron.Tests --filter "FullyQualifiedName~BinaryOpTests"   # FAIL: divide by zero
```
