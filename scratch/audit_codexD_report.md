## Must-fix before announcing

- `README.md:151` - Claim: the matmul tiers are "cuBLAS SGEMM ... a hand-tiled shared-memory kernel (the CPU large path), and the naive one-thread-per-output kernel." Why it's a problem: this is verified wrong. The managed CPU backend uses `CpuKernels.MatMul2D` via `CpuSimdRuntime.LaunchMatMul` (`src/Tensotron/CpuSimdRuntime.cs:133-137`), while the tiled shared-memory kernel is in the ILGPU runtime fallback when `large && tiledFits` (`src/Tensotron/TensorRuntime.cs:942-958`). Calling it "the CPU large path" will be challenged immediately by ILGPU readers. Suggested correction: "cuBLAS SGEMM (`M,N,K >= 64` on CUDA), a hand-tiled ILGPU shared-memory kernel when cuBLAS is unavailable, and a naive one-thread-per-output ILGPU kernel for tiny/skinny products; the managed CPU backend has its own `Vector<float>` matmul."

- `src/Tensotron/PACKAGE.md:41-43` - Claim: `TensorRuntime.Capture` / `CapturedGraph.Replay` replays buffer-to-buffer "~2.5-2.9x faster on a small step." Why it's a problem: this NuGet readme is stale relative to the current README and code. Native CUDA graph replay is implemented (`TensorRuntime.EnableCudaGraph` defaults true at `src/Tensotron/TensorRuntime.cs:164-168`; CUDA capture implementation begins at `src/Tensotron/TensorRuntime.cs:588`; `CapturedGraph` exposes `UsesNativeGraph` at `src/Tensotron/CapturedGraph.cs:12-18`), and the public README now advertises one `cuGraphLaunch` and ~5-10x / ~6-8x. A packed NuGet page saying only 2.5-2.9x will make the project look internally inconsistent. Suggested correction: align PACKAGE.md with README: native CUDA graph replay on CUDA, software replay fallback ~2-2.7x, and managed SIMD CPU does not support capture (`src/Tensotron/TensorRuntime.cs:187-210`).

- `README.md:263-277` - Claim: "Step capture (fixed-shape training/inference)" without an immediate backend caveat. Why it's a problem: the API is not backend-general. The base implementation throws `NotSupportedException`, and the doc comment says the managed CPU backend "does not support it" (`src/Tensotron/TensorRuntime.cs:187-210`). The paragraph later says "On CUDA", but the heading can be read as applying to all backends. Suggested correction: "Step capture on the ILGPU/CUDA backend..." and explicitly say managed/SIMD CPU has no launch overhead to amortize and no capture support.

## Claims not reproducible / unverifiable

- `README.md:7` - Claim: the corridor PPO demo trains "from scratch to flawless laps in ~2 minutes on the CPU backend." Why it's a problem: I found the showcase test (`showcase/Tensotron.Showcase/CorridorTests.cs:36-57`), but no committed benchmark log or timing artifact supporting the "~2 minutes" number. The test itself is correctness/convergence, not a reproducible timing table, and timing will vary heavily by CPU and thread env. Suggested correction: either add a committed artifact with hardware, backend, command, and result, or soften to "the CPU backend can train the demo; on the author's machine it reaches stable laps in about 2 minutes."

- `README.md:32`, `README.md:44`, `tools/bench/plot_perf.py:117-119` - Claim: single-agent CPU inference is "~8x faster" than PyTorch CPU. Why it's a problem: the benchmark scripts exist (`examples/Tensotron.Examples/BenchExample.cs:320-395`, `tools/bench/torch_infer.py:34-71`), but the raw output files used to generate `docs/img/cpu_inference.png` are not committed. `plot_perf.py` expects `infer_simd.txt`, `infer_cuda.txt`, and `infer_torch.txt` (`tools/bench/plot_perf.py:72-79`) but only the PNG is present. Suggested correction: commit the raw `RESULT infer ...` logs under `docs/artifacts/bench/` or add exact command + captured output in the performance doc. Also state CPU model and `TENSOTRON_CPU_THREADS` setting.

- `README.md:51`, `src/Tensotron/PACKAGE.md:26-27`, `CLAUDE.md:40` - Claim: SIMD CPU batch-1 is "~645x faster than the ILGPU scalar CPU accelerator." Why it's a problem: the inference benchmark can compare `TENSOTRON_BACKEND=cpu` vs `simd` (`examples/Tensotron.Examples/BenchExample.cs:312-319`), but no raw result artifact in the repo backs 645x. Source comments also mention a different measured relationship: "~274x vs hand-scalar at batch=1" (`src/Tensotron/CpuSimdRuntime.cs:5-10`), which is a different baseline and invites confusion. Suggested correction: commit the CPU/scalar backend raw log or rephrase as "hundreds of times faster in the included inference microbenchmark; author measurement ~645x on <hardware>."

- `docs/PERFORMANCE_VS_PYTORCH.md:20-47` - Claim: exact MLP/CNN/GEMM tables. Why it's a problem: the Tensotron and PyTorch benchmark scripts exist (`examples/Tensotron.Examples/BenchExample.cs:72-100`, `tools/bench/torch_bench.py:46-131`), but the exact table values are hard-coded in the doc/plot script (`tools/bench/plot_perf.py:127-139`) and no raw `RESULT` logs are committed. This makes the numbers reproducible in principle but not auditable as the numbers actually measured. Suggested correction: add raw logs for the listed RTX 4090 runs, including median aggregation if used, or include a script target that regenerates the markdown table from raw outputs.

- `docs/PERFORMANCE_VS_PYTORCH.md:123-132` - Claim: `stepbreakdown` pins a PPO-scale step at 95% host-bound, 2272 us host-dispatch vs 121 us device-tail, 36.7 us/op autograd construction; replay numbers 1252 -> 149 us, 1290 -> 225 us, 1609 -> 242 us. Why it's a problem: `StepBreakdown` and `ReplayBench` exist (`examples/Tensotron.Examples/BenchExample.cs:214-309`, `examples/Tensotron.Examples/BenchExample.cs:130-200`), but the raw output is not committed. The doc cites precise microseconds without hardware/software/run-count evidence beyond "Measured on a 4090". Suggested correction: commit `stepbreakdown.txt` and `replay.txt` raw outputs or downgrade to approximate examples with command and hardware.

- `docs/PERFORMANCE_VS_PYTORCH.md:145-149` - Claim: strided-batched GEMM is "~20-200x" and batch-128 64^3 bmm is 2969 -> 15 us. Why it's a problem: the code and tests prove the path exists (`src/Tensotron/TensorRuntime.cs:1127-1150`, `tests/Tensotron.Tests/BatchedCuBlasMatMulTests.cs:55-100`), but I did not find a benchmark mode or committed result that reproduces this 2969 -> 15 us number. Suggested correction: add a `bench` command for strided batched GEMM or move the number to an artifact-backed note.

- `README.md:252-254` - Claim: `cublasSgemmStridedBatched` gives "~20-200x on small-matrix batches." Why it's a problem: same unverifiable number as above, and it appears in the top-level README. Suggested correction: either cite a committed benchmark artifact or remove the numeric range from README and keep the qualitative statement.

## API/code-sample drift

- `README.md:72-75` and `src/Tensotron/PACKAGE.md:52-55` - Claim/sample: `x.Grad == [2, 4, 6]`. Why it's not wrong but slightly imprecise: `x.Grad` is a nullable `Tensor`, not an array (`src/Tensotron/Tensor.cs:78-80`). The snippet compiles and produces `2,4,6` when read as `x.Grad!.ToArray()`, but the comment could be clearer for C# users. Suggested correction: `// x.Grad!.ToArray() == [2, 4, 6]`.

- `README.md:81-84` - Sample: `new Adam(model.Parameters().ToList(), lr: 1e-2f)`. Why it's okay but has an implicit using: this compiles only with `System.Linq` available. Modern SDK projects with implicit usings are fine, and `src/Tensotron/Tensotron.csproj:5` enables implicit usings, but a minimal older/non-implicit consumer may need `using System.Linq;`. Suggested correction: optional, add `using System.Linq;` to longer sample if you want copy/paste robustness outside SDK defaults.

- `README.md:276` - Claim: `Tensor.ScalarInput(v)` refreshed with `Upload` between replays. Why it's correct but underspecified: `Upload` takes `float[]` (`src/Tensotron/Tensor.cs:263-268`), so a user must call something like `coef.Upload(new[] { v })`, not `coef.Upload(v)`. Suggested correction: show the concrete scalar update form once in docs.

## Scope accuracy

- `README.md:12` - Scope says PyTorch `state_dict`/`safetensors` interop is not implemented. Accurate, but keep the "PyTorch interop" qualifier. The library does have its own `Module.StateDict` (`src/Tensotron/Module.cs:50-56`) and save/load/checkpoint APIs (`src/Tensotron/Serialization.cs:15-18`, `src/Tensotron/Serialization.cs:94-119`), so dropping "PyTorch interop" would become wrong.

- `src/Tensotron/PACKAGE.md:10-12` - Scope says PyTorch `state_dict` interop is not implemented, but omits `safetensors` while README includes it (`README.md:12`). Why it's a problem: minor scope inconsistency between public README and packed NuGet readme. Suggested correction: match README: "PyTorch `state_dict`/`safetensors` interop."

- `README.md:232` - Claimed present: `Conv2d`, `MaxPool2d` / `AvgPool2d`, LayerNorm, BatchNorm1d/2d, GroupNorm. Verified supported by public classes/functions (`src/Tensotron/Conv.cs:10-29`, `src/Tensotron/Pool.cs:8-32`, `src/Tensotron/Norm.cs:11-178`) and tests/fixtures (`tests/Tensotron.Tests/ConvTests.cs`, `PoolTests.cs`, `NormTests.cs`).

- `README.md:228-234` - Claimed op/training surface is broad and mostly accurate. I did not find public `Embedding`, RNN/LSTM/GRU, attention modules, ConvTranspose, Conv1d, or Conv3d in `src/Tensotron`. The "Not implemented" list is directionally accurate and not missing an obvious implemented item.

## Overstatement/framing

- `README.md:10`, `CLAUDE.md:9` - Claim: "matches PyTorch exactly" and "mimics PyTorch in everything." Why it's a risk: the qualifier "for every op it implements" is present in README but not in CLAUDE line 9, and "exactly" is a strong public claim for GPU floating-point reductions that are explicitly not bitwise deterministic (`README.md:165-167`). Suggested correction for public docs: "matches PyTorch semantics and gradients within the project tolerances for implemented ops; floating-point reductions are not bitwise deterministic on GPU."

- `docs/PERFORMANCE_VS_PYTORCH.md:3-6`, `docs/PERFORMANCE_VS_PYTORCH.md:152-157` - Claim/framing: "we're in the same league", "dead-even", "Same league. Up there." Why it's a problem: this reads hype-forward for an ILGPU/ML expert audience, especially because some precise numbers lack raw artifacts and conv trails 2-3.4x. Suggested correction: lead with the concrete table and caveats, then a restrained conclusion: "Tensotron matches PyTorch FP32 GEMM throughput, is faster on the measured small MLP step, and is slower on MNIST-scale conv because it uses im2col + GEMM rather than cuDNN."

- `README.md:36` - Claim: "Tensotron is in PyTorch's league" plus "matches PyTorch on FP32 GEMM." Why it's mostly supported but should be caveated: the GEMM claim is code-backed by cuBLAS use (`src/Tensotron/TensorRuntime.cs:947-950`) and benchmark scripts exist, but "in PyTorch's league" is subjective. Suggested correction: "On the measured RTX 4090 workloads, Tensotron matches PyTorch FP32 GEMM, is faster on the small MLP step, roughly par on the large MLP step, and slower on conv."

- `README.md:40` - Claim: native CUDA graph replay "erases nearly all the per-step host dispatch." Why it's a little too absolute: the replay benchmark supports large speedups in principle, but some host work remains and the exact numbers are not artifact-backed. Suggested correction: "removes most of the per-step host graph rebuild/dispatch cost in these fixed-shape benchmarks."

## Minor (typos/links)

- `README.md:44` - Reproduction text says run inference under `TENSOTRON_BACKEND=simd` then `=cuda`, but not `=cpu` for the separate 645x scalar-CPU claim. Suggested correction: add `TENSOTRON_BACKEND=cpu` if keeping the scalar-CPU comparison.

- `examples/Tensotron.Examples/Program.cs:31` - Unknown-command help omits valid commands mentioned in docs (`inference`, `replay`, `stepbreakdown`). This is not in markdown, but it affects the documented commands if mistyped. Suggested correction: update the help string.

- `README.md:58-63` - Installation says not yet published to NuGet and local pack produces `Tensotron.0.1.0-alpha.nupkg`. This matches `src/Tensotron/Tensotron.csproj:11-12`. No issue.

- `docs/PERFORMANCE_VS_PYTORCH.md:16` - Reproduce command is correct for GPU ladder. It does not mention that `GemmTf32` is a separate example command (`examples/Tensotron.Examples/BenchExample.cs:104-121`) if readers want to reproduce the Tensotron TF32 column. Suggested correction: add `dotnet run --project examples/Tensotron.Examples -c Release -- gemmtf32` if that command is wired in `Program.cs`; otherwise wire it or remove the Tensotron TF32 column from the table.

## Accurate & well-supported (brief)

- The README quick-start APIs exist and compile: `Tensor.FromArray(...).RequireGrad()`, operator `*`, `.Sum()`, and `.Backward()` are present (`src/Tensotron/Tensor.cs:140-148`, `src/Tensotron/Tensor.cs:204-208`, `src/Tensotron/TensorOps.cs:249-254`, `src/Tensotron/Tensor.cs:309-343`). A scratch compile produced gradient `2,4,6`.

- Backend env vars are real: `TENSOTRON_BACKEND` supports `auto|cuda|cpu|simd` (`src/Tensotron/TensorRuntime.cs:56-87`), `TENSOTRON_CPU_THREADS` supports `auto|max|off|N` (`src/Tensotron/CpuSimdRuntime.cs:21-37`), and `TENSOTRON_CPU_MINFLOPS` exists (`src/Tensotron/CpuSimdRuntime.cs:25-28`).

- Device diagnostics named in README exist: `Cuda.IsAvailable()`, `Cuda.DeviceCount()`, `Cuda.GetDeviceName()`, `Accelerators.List()`, and `Accelerators.Active()` (`src/Tensotron/Device.cs:14-55`), with tests in `tests/Tensotron.Tests/DeviceTests.cs`.

- The core parity-test claim is substantially supported: deterministic op fixtures are committed under `tests/Tensotron.Tests/Fixtures/`, loaded by the parity tests, and dropout has property tests (`tests/Tensotron.Tests/DropoutTests.cs:7-56`). The solution builds cleanly with `dotnet build Tensotron.sln --verbosity quiet`.
