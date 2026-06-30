using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Tensotron;

/// <summary>
/// Which backend the runtime drives. <see cref="Auto"/> (the default) prefers a CUDA GPU and, when
/// none is present, falls back to the fast managed/SIMD CPU backend (<see cref="CpuSimd"/>) — NOT
/// the slow ILGPU CPU accelerator. <see cref="Cuda"/> forces the GPU (throws if absent).
/// <see cref="CpuSimd"/> selects the hand-written, ILGPU-free managed/SIMD CPU backend (no per-op
/// device dispatch) — the fast path for small-model CPU inference/training.
/// <see cref="Cpu"/> forces ILGPU's *scalar* CPU accelerator: a correctness/verification reference
/// ONLY (full per-op device-dispatch overhead, ~600x slower at batch-1) — selecting it prints a
/// loud warning. Use <see cref="CpuSimd"/> for real CPU work.
/// </summary>
public enum TensorBackend { Auto, Cuda, Cpu, CpuSimd }

/// <summary>Structural equality/hash for int[] so shape/stride arrays can key a content cache.</summary>
internal sealed class IntArrayComparer : IEqualityComparer<int[]>
{
    public bool Equals(int[]? a, int[]? b)
    {
        if (ReferenceEquals(a, b)) return true;
        if (a is null || b is null || a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public int GetHashCode(int[] a)
    {
        var h = new HashCode();
        foreach (var v in a) h.Add(v);
        return h.ToHashCode();
    }
}

/// <summary>
/// Backend-agnostic compute surface: storage allocation + the <c>Launch*</c> op kernels + a few
/// diagnostics. Exactly one concrete runtime is live per process (chosen once from
/// <see cref="RequestedBackend"/>): <see cref="IlgpuRuntime"/> (CUDA or ILGPU-CPU) or
/// <see cref="CpuSimdRuntime"/> (hand-written managed/SIMD). The op layer talks only to this base
/// type, passing <see cref="TensorStorage"/>; each runtime downcasts to the storage it owns.
/// </summary>
public abstract class TensorRuntime : IDisposable
{
    private static readonly Lazy<TensorRuntime> _instance = new(Create);
    public static TensorRuntime Instance => _instance.Value;

    private static TensorBackend _requestedBackend = ParseBackendEnv();

    /// <summary>
    /// Backend to use, read once when the runtime is first created. Set this before touching any
    /// tensor, or set the <c>TENSOTRON_BACKEND</c> env var (auto|cuda|cpu|simd). Tensotron runs on
    /// exactly one backend process-wide, so — like PyTorch's device model — tensors never mix
    /// backends; selecting here picks that single backend. Throws if changed after the runtime has
    /// already been created (the choice is latched on the first tensor op), rather than silently
    /// ignoring the new value.
    /// </summary>
    public static TensorBackend RequestedBackend
    {
        get => _requestedBackend;
        set
        {
            if (_instance.IsValueCreated && value != _requestedBackend)
                throw new InvalidOperationException(
                    $"RequestedBackend cannot be changed to {value}: the runtime was already " +
                    $"initialized as {_requestedBackend} on the first tensor op. Set RequestedBackend " +
                    "(or the TENSOTRON_BACKEND env var) before touching any tensor.");
            _requestedBackend = value;
        }
    }

    private static TensorBackend ParseBackendEnv() =>
        (Environment.GetEnvironmentVariable("TENSOTRON_BACKEND")?.Trim().ToLowerInvariant()) switch
        {
            "cuda" or "gpu" => TensorBackend.Cuda,
            "simd" or "cpu-simd" or "cpusimd" => TensorBackend.CpuSimd,
            "cpu" => TensorBackend.Cpu,
            _ => TensorBackend.Auto,
        };

    private static TensorRuntime Create()
    {
        switch (RequestedBackend)
        {
            case TensorBackend.CpuSimd:
                return new CpuSimdRuntime();
            case TensorBackend.Auto:
                // Best available: a CUDA GPU if present, else the FAST managed/SIMD CPU backend.
                // NOT the slow ILGPU scalar CPU accelerator — that is a verification-only reference,
                // reachable only by an explicit TENSOTRON_BACKEND=cpu.
                return CudaPresent() ? new IlgpuRuntime(TensorBackend.Cuda) : new CpuSimdRuntime();
            default: // Cuda or Cpu — both go to ILGPU; the constructor warns LOUDLY if it actually
                     // lands on the scalar CPU accelerator, whatever the path that got it there.
                return new IlgpuRuntime(RequestedBackend);
        }
    }

    /// <summary>
    /// Big, unmissable banner emitted whenever the live backend is ILGPU's *scalar* CPU accelerator.
    /// That path is a correctness/verification reference only — full per-op device dispatch, ~600x
    /// slower than the managed/SIMD CPU backend at batch-1. Anything real should be on CUDA or
    /// <c>TENSOTRON_BACKEND=simd</c>.
    /// </summary>
    protected static void WarnIlgpuCpuIsSlow()
    {
        const string bar = "************************************************************************";
        Console.Error.WriteLine();
        Console.Error.WriteLine(bar);
        Console.Error.WriteLine("**  WARNING: Tensotron is running on the ILGPU *SCALAR* CPU accelerator. **");
        Console.Error.WriteLine("**                                                                      **");
        Console.Error.WriteLine("**  This is a SLOW correctness/verification reference ONLY -- it carries **");
        Console.Error.WriteLine("**  full per-op device-dispatch overhead (~600x slower than the managed  **");
        Console.Error.WriteLine("**  CPU backend at batch-1). DO NOT use it for real workloads.           **");
        Console.Error.WriteLine("**                                                                      **");
        Console.Error.WriteLine("**  For fast CPU execution set:  TENSOTRON_BACKEND=simd                  **");
        Console.Error.WriteLine("**  (Auto already prefers SIMD-CPU when no CUDA GPU is present.)         **");
        Console.Error.WriteLine(bar);
        Console.Error.WriteLine();
    }

    /// <summary>
    /// Probe for a CUDA device without touching <see cref="Instance"/> (we are mid-construction of
    /// it, so <see cref="Cuda.IsAvailable"/> would re-enter the <see cref="Lazy{T}"/>). Uses a
    /// throwaway CUDA-only context; returns false if CUDA is absent or the driver errors.
    /// </summary>
    private static bool CudaPresent()
    {
        try
        {
            using var ctx = Context.Create(b => b.Cuda());
            return ctx.Devices.Any(d => d.AcceleratorType == AcceleratorType.Cuda);
        }
        catch
        {
            return false;
        }
    }

    // ---- lightweight instrumentation (perf bench only; negligible when unread) ----
    /// <summary>Kernel launches + device copies issued.</summary>
    public long Launches { get; protected set; }
    /// <summary>Buffer allocations (float results + int stride/dim buffers).</summary>
    public long Allocs { get; protected set; }
    /// <summary>Host→device uploads (scalar/leaf copies + per-launch stride buffers).</summary>
    public long HostUploads { get; protected set; }
    public virtual void ResetCounters() { Launches = 0; Allocs = 0; HostUploads = 0; }
    internal void NoteHostUpload() => HostUploads++;

    // ---- diagnostics (overridable; CPU-safe defaults) ----
    public abstract string DeviceName { get; }
    public abstract bool IsGpu { get; }
    public abstract Context Context { get; }
    public virtual bool UsesCuBlas => false;
    public virtual bool AllowTf32 { get => false; set { } }
    public virtual void SetCuBlasMathMode(int mode) { }
    public virtual int FlushEvery { get; set; } = 64;
    public virtual bool PoolingEnabled { get; set; } = true;
    public virtual long PoolHits => 0;

    /// <summary>When true (default), a step captured by <see cref="Capture"/> on the CUDA backend is
    /// folded into a native CUDA graph and replayed with a single <c>cuGraphLaunch</c>. Set false to
    /// force software replay (re-firing the recorded launches) — an escape hatch if the body uses an
    /// op that is not graph-capturable, and the knob a benchmark flips to compare the two paths.</summary>
    public virtual bool EnableCudaGraph { get; set; } = true;

    /// <summary>When true (default), a constant-stride batched matmul on the CUDA backend issues one
    /// <c>cublasSgemmStridedBatched</c> for the whole batch instead of a per-matrix SGEMM loop. Set
    /// false to force the loop (escape hatch / benchmark knob). No effect off CUDA.</summary>
    public virtual bool EnableStridedBatchedGemm { get; set; } = true;
    /// <summary>True when a constant-stride batched matmul will take the single
    /// <c>cublasSgemmStridedBatched</c> path (CUDA, entry point resolved, and not disabled).</summary>
    public virtual bool UsesStridedBatchedGemm => false;

    /// <summary>
    /// Max worker threads for the managed CPU backend's row-parallel matmul (the only multi-threaded
    /// op). 1 = serial. Only the SIMD CPU backend honors it; GPU/ILGPU ignore it. Set it lower (or 1)
    /// when you run many Tensotron instances in parallel (e.g. PPO agents per core) so they don't
    /// oversubscribe; raise it for a single big-batch trainer. Defaults from <c>TENSOTRON_CPU_THREADS</c>
    /// (<c>auto</c> = physical cores, <c>max</c> = all logical, N, or off=1).
    /// </summary>
    public virtual int CpuMatMulThreads { get; set; } = 1;

    /// <summary>Record a fixed-shape step and return a graph that replays its device launches with
    /// no host-side graph rebuild. Only the ILGPU backend amortizes per-launch overhead this way;
    /// the managed CPU backend has no launch to amortize and does not support it.
    ///
    /// Capture the body in STEADY STATE: every persistent buffer it touches (optimizer moments m/v,
    /// the Adam step state, any lazily-built parameter) must already be allocated before capture, so
    /// the recorded step does not contain one-time zero-init that would re-run — and corrupt — on
    /// every replay. Run at least one eager warmup step first (the same requirement as PyTorch's CUDA
    /// graphs). Per-step grad zeroing is fine; it is meant to re-run each replay.</summary>
    public virtual CapturedGraph Capture(Func<Tensor> body) =>
        throw new NotSupportedException($"{GetType().Name} does not support trace capture.");

    /// <summary>Re-fire a captured graph's recorded launches (no host graph rebuild).</summary>
    internal void Replay(List<Action> thunks)
    {
        for (int i = 0; i < thunks.Count; i++) thunks[i]();
    }

    // ---- storage + compute surface ----
    public abstract TensorStorage Allocate(int length);
    internal abstract void ReturnToPool(TensorStorage buf);
    public abstract void Sync();
    public abstract void ZeroBuffer(TensorStorage buf);
    public abstract void DeviceCopy(TensorStorage src, TensorStorage dst);

    public abstract void LaunchBinary<TOp>(TensorStorage a, TensorStorage b, TensorStorage outv,
        int[] outDims, int[] aStride, int[] bStride) where TOp : struct, IBinaryOp;
    public abstract void LaunchUnaryFwd<TOp>(TensorStorage x, TensorStorage outv) where TOp : struct, IUnaryOp;
    public abstract void LaunchUnaryBwd<TOp>(TensorStorage x, TensorStorage y, TensorStorage gy, TensorStorage gx)
        where TOp : struct, IUnaryOp;
    public abstract void LaunchUnaryFwdP<TOp>(TOp op, TensorStorage x, TensorStorage outv) where TOp : unmanaged, IUnaryOp;
    public abstract void LaunchUnaryBwdP<TOp>(TOp op, TensorStorage x, TensorStorage y, TensorStorage gy, TensorStorage gx)
        where TOp : unmanaged, IUnaryOp;
    public abstract void LaunchAddInto(TensorStorage target, TensorStorage source);
    public abstract void LaunchReduceSum(TensorStorage inp, TensorStorage outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask);
    public abstract void LaunchReduce<TR>(TensorStorage inp, TensorStorage outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask) where TR : struct, IReduceOp;
    public abstract void LaunchReduceArg(TensorStorage inp, TensorStorage outIdx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax);
    public abstract void LaunchMatMul(TensorStorage a, TensorStorage b, TensorStorage c,
        int M, int N, int K, int aMs, int aKs, int bKs, int bNs);
    public abstract void LaunchReduceArgGrad(TensorStorage inp, TensorStorage gout, TensorStorage gx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax);
    public abstract void LaunchProdGrad(TensorStorage inp, TensorStorage gout, TensorStorage gx,
        int[] inDims, int[] inStrides, int[] outStrides, int[] reduceMask);
    public abstract void LaunchStridedCopy(TensorStorage inp, TensorStorage outp,
        int[] outDims, int[] inStrides, int baseOff = 0);
    public abstract void LaunchScatterAxisRange(TensorStorage src, TensorStorage dst,
        int[] srcDims, int[] dstStrides, int axis, int axisOffset);
    public abstract void LaunchMatMulBatched(TensorStorage a, TensorStorage b, TensorStorage c,
        int batchCount, int M, int N, int K, int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBatchStrides, int[] bBatchStrides);
    public abstract void LaunchGatherAxis(TensorStorage x, TensorStorage outp,
        int[] index, int[] outDims, int[] xStrides, int axis, int mode);
    public abstract void LaunchScatterAddAxis(TensorStorage g, TensorStorage gx,
        int[] index, int[] srcDims, int[] dstStrides, int axis, int mode);
    public abstract void LaunchRepeat(TensorStorage x, TensorStorage outp,
        int[] outDims, int[] inDims, int[] inStrides);
    public abstract void LaunchRepeatGrad(TensorStorage g, TensorStorage gx,
        int[] outDims, int[] inDims, int[] inStrides);
    public abstract void LaunchIm2Col(TensorStorage x, TensorStorage col, int[] cfg);
    public abstract void LaunchCol2Im(TensorStorage gcol, TensorStorage gx, int[] cfg);
    /// <summary>MaxPool forward; returns an opaque argmax handle (ILGPU device int buffer or a host
    /// int[]) consumed by <see cref="LaunchMaxPool2dGrad"/>, or null when keepArgmax is false.</summary>
    public abstract object? LaunchMaxPool2d(TensorStorage x, TensorStorage outp, int[] cfg, bool keepArgmax);
    public abstract void LaunchMaxPool2dGrad(TensorStorage g, TensorStorage gx, object argmax);
    public abstract void LaunchAvgPool2d(TensorStorage x, TensorStorage outp, int[] cfg);
    public abstract void LaunchAvgPool2dGrad(TensorStorage g, TensorStorage gx, int[] cfg);
    public abstract void LaunchAdam(TensorStorage p, TensorStorage g, TensorStorage m, TensorStorage v,
        float b1, float oneMinusB1, float b2, float oneMinusB2,
        float lr, float eps, TensorStorage adv, float coupledWd, float decoupledFactor);
    /// <summary>Advance a parameter's device-resident Adam state <c>adv</c> = [t, invBc1, invBc2]
    /// (increment t, recompute bias corrections). Call once per param per step before
    /// <see cref="LaunchAdam"/>; a captured graph replays it so bias correction stays correct.</summary>
    public abstract void LaunchAdvanceAdam(TensorStorage adv, float b1, float b2);
    public abstract void LaunchSgd(TensorStorage p, TensorStorage g, TensorStorage buf,
        float lr, float momentum, float weightDecay, float dampening, float nesterov, float hasBuf);

    public abstract void Dispose();
}

/// <summary>
/// Owns the single ILGPU Context + Accelerator and caches compiled kernels. One accelerator,
/// kernels loaded once and cached. The backend (CUDA-preferred, else ILGPU-CPU) is chosen at
/// construction. This is the device path; the math lives in <see cref="Kernels"/>.
/// </summary>
internal sealed class IlgpuRuntime : TensorRuntime
{
    public override Context Context { get; }
    public Accelerator Accelerator { get; }
    public override string DeviceName => Accelerator.Name;
    public override bool IsGpu => Accelerator.AcceleratorType != AcceleratorType.CPU;

    // The stream every launch currently targets. Normally the DEFAULT stream, so host uploads and
    // pulls — which ILGPU issues on the default stream — stay ordered with kernels (no cross-stream
    // races). It is swapped to _graphStream only while a native CUDA graph is being recorded (see
    // TryBuildCudaGraph); the launch helpers and capture thunks all read this field, so the swap
    // redirects every launch with no duplicated code.
    private AcceleratorStream _stream;
    // Owned non-null stream used ONLY to capture and launch the native CUDA graph. ILGPU's default
    // stream is the CUDA NULL stream, which cuStreamBeginCapture cannot record; a created stream can.
    // Null on the CPU backends, which never build a graph.
    private readonly AcceleratorStream? _graphStream;

    // Unwrap a backend-agnostic storage handle to the ILGPU buffer it must be on this backend.
    private static MemoryBuffer1D<float, Stride1D.Dense> B(TensorStorage s) => ((DeviceStorage)s).Buffer;

    // cuBLAS GEMM handle — only on the CUDA backend (null on CPU). Runs on the default stream like
    // every other launch; during native-graph capture its stream is briefly pointed at _graphStream
    // (and restored) so the SGEMMs are recorded into the graph too.
    private readonly CuBlas? _blas;
    public override bool UsesCuBlas => _blas != null;

    // TF32 tensor-core matmul is a cuBLAS *math-mode* flip — storage stays FP32, only the GEMM's
    // internal multiply rounds inputs to 19-bit TF32 on the tensor cores (FP32 accumulate). Off by
    // default so matmul stays exact-FP32 (golden-fixture parity, torch-strict equivalence). Opt-in
    // for ~1.5x on matmul-bound work. mode: 0=Default(true FP32), 1=TensorOp(legacy), 3=TF32TensorOp.
    private int _mathMode;
    public override void SetCuBlasMathMode(int mode)
    {
        _mathMode = mode;
        if (_blas != null) _blas.MathMode = (CuBlasMathMode)mode;
    }
    public override bool AllowTf32 { get => _mathMode == 3; set => SetCuBlasMathMode(value ? 3 : 0); }

    // Non-generic kernels: loaded once into fields.
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>> _addInto;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _reduceSum;
    private readonly Action<AcceleratorStream, Index1D, int, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _reduceSumChunked;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        int, int, int, int, int, int, int> _matmul;
    private readonly Action<AcceleratorStream, KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        int, int, int, int, int, int, int> _matmulTiled;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, int> _stridedCopy;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, int, int> _scatterAxisRange;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        int, int, int, int, int, int, int, int,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _matmulBatched;

    // Generic elementwise kernels: one per op struct, cached by type.
    private readonly ConcurrentDictionary<Type, object> _binaryKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryFwdKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryBwdKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryFwdPKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryBwdPKernels = new();
    private readonly ConcurrentDictionary<Type, object> _reduceKernels = new();
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _reduceArg;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _reduceArgGrad;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _prodGrad;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _gatherAxis;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _scatterAddAxis;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _repeat;
    private readonly Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _repeatGrad;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _im2col;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _col2im;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>> _maxPool2d;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _maxPool2dGrad;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _avgPool2d;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _avgPool2dGrad;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        float, float, float, float, float, float, ArrayView<float>, float, float> _adamStep;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, float, float> _advanceAdam;
    private readonly Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        float, float, float, float, float, float> _sgdStep;

    public IlgpuRuntime(TensorBackend backend)
    {
        // Enable CPU + all GPUs (Default), then pick one device per the requested backend.
        Context = Context.Create(b => b.Default().EnableAlgorithms());
        Accelerator = SelectDevice(Context, backend).CreateAccelerator(Context);

        // The ILGPU CPU accelerator is the slow scalar verification reference, never the fast path.
        // Whatever route selected it (explicit cpu, or any future fallback), say so unmistakably.
        if (Accelerator.AcceleratorType == AcceleratorType.CPU)
            WarnIlgpuCpuIsSlow();

        // Launches go on the default stream by default, keeping them ordered with ILGPU's
        // default-stream host copies. _graphStream is reserved for native CUDA-graph capture (CUDA
        // only) — the default stream is the NULL stream and cannot be recorded.
        _stream = Accelerator.DefaultStream;
        _graphStream = Accelerator is CudaAccelerator ? Accelerator.CreateStream() : null;

        _addInto = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(Kernels.AddInto);

        _reduceSum = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.ReduceSum);

        _reduceSumChunked = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.ReduceSumChunked);

        _matmul = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            int, int, int, int, int, int, int>(Kernels.MatMul2D);

        _matmulTiled = Accelerator.LoadKernel<
            ArrayView<float>, ArrayView<float>, ArrayView<float>,
            int, int, int, int, int, int, int>(Kernels.MatMul2DTiled);

        _reduceArg = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(Kernels.ReduceArg);

        _reduceArgGrad = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(Kernels.ReduceArgGrad);

        _prodGrad = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.ProdGrad);

        _stridedCopy = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, int>(Kernels.StridedCopy);

        _scatterAxisRange = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, int, int>(Kernels.ScatterAxisRange);

        _matmulBatched = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            int, int, int, int, int, int, int, int,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.MatMulBatched);

        _gatherAxis = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Kernels.GatherAxis);

        _scatterAddAxis = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Kernels.ScatterAddAxis);

        _repeat = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.Repeat);

        _repeatGrad = Accelerator.LoadAutoGroupedKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.RepeatGrad);

        _im2col = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.Im2Col);
        _col2im = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.Col2Im);

        _maxPool2d = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>>(Kernels.MaxPool2d);
        _maxPool2dGrad = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.MaxPool2dGrad);
        _avgPool2d = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.AvgPool2d);
        _avgPool2dGrad = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.AvgPool2dGrad);

        _adamStep = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            float, float, float, float, float, float, ArrayView<float>, float, float>(Kernels.AdamStep);
        _advanceAdam = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, float, float>(Kernels.AdvanceAdam);
        _sgdStep = Accelerator.LoadAutoGroupedKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            float, float, float, float, float, float>(Kernels.SgdStep);

        // cuBLAS for the compute-bound matmul regime (vendor-tuned SGEMM). CUDA only; the
        // tiled/naive kernels remain the CPU path and the small-matmul path.
        if (Accelerator is CudaAccelerator cuda)
            _blas = new CuBlas(cuda) { PointerMode = CuBlasPointerMode.Host };
    }

    private static Device SelectDevice(Context ctx, TensorBackend backend)
    {
        Device? pick = backend switch
        {
            TensorBackend.Cuda => ctx.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda),
            TensorBackend.Cpu => ctx.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.CPU),
            _ => null, // Auto
        };
        if (backend != TensorBackend.Auto && pick is null)
            throw new InvalidOperationException(
                $"Requested backend {backend} but no matching ILGPU device is available.");
        return pick ?? ctx.GetPreferredDevice(preferCPU: false);
    }

    // ---- caching device allocator (size-bucketed free list) ----
    // Reuse is always on (harmless: an empty pool just falls through to a real allocation). The
    // pool is FED only by explicit Tensor.Dispose() — so default, non-disposing code is byte-for-byte
    // unchanged. Code that bounds its own lifetime (inference loops, or training that frees each
    // step's graph via Tensor.DisposeGraph) recycles buffers instead of churning cudaMalloc/free,
    // which is the dominant cost once activations get large (see PERFORMANCE_LOG E6).
    private readonly Dictionary<int, Stack<MemoryBuffer1D<float, Stride1D.Dense>>> _floatPool = new();
    private long _pooledBytes;
    private const long MaxPooledBytes = 1L << 30; // cap retained pool memory at ~1 GB
    private long _poolHits;
    /// <summary>Buffers served from the pool (no device allocation). Bench/diagnostics only.</summary>
    public override long PoolHits => _poolHits;

    public override TensorStorage Allocate(int length)
    {
        MemoryBuffer1D<float, Stride1D.Dense> buf;
        if (PoolingEnabled && _floatPool.TryGetValue(length, out var stack) && stack.Count > 0)
        {
            buf = stack.Pop();
            _pooledBytes -= (long)length * sizeof(float);
            _poolHits++;
        }
        else
        {
            Allocs++;
            buf = Accelerator.Allocate1D<float>(length);
        }
        if (_capture != null) { (_pinned ??= new()).Add(buf); _capturePins!.Add(buf); }  // pin trace memory against pool reuse
        return new DeviceStorage(buf);
    }

    // Return an owned float buffer for reuse (called by Tensor.Dispose). Over the cap → really free.
    internal override void ReturnToPool(TensorStorage bufS)
    {
        var buf = B(bufS);
        if (_pinned != null && _pinned.Contains(buf)) return;  // trace owns it — never recycle/free
        long bytes = (long)buf.Length * sizeof(float);
        if (!PoolingEnabled || _pooledBytes + bytes > MaxPooledBytes) { buf.Dispose(); return; }
        int len = (int)buf.Length;
        if (!_floatPool.TryGetValue(len, out var stack)) { stack = new(); _floatPool[len] = stack; }
        stack.Push(buf);
        _pooledBytes += bytes;
    }

    // ---- async batching ----
    // Kernels launch on the accelerator's in-order default stream and are NOT synced per
    // launch. The stream is drained (a) at every host pull (Sync, via ToArray/Item) and
    // (b) every FlushEvery launches as a safety valve to bound the in-flight queue.
    private int _flushEvery = 64;
    /// <summary>Number of kernel launches between safety-valve stream drains. Larger = fewer
    /// auto-syncs (good for fixed-shape steps). Minimum 1.</summary>
    public override int FlushEvery
    {
        get => _flushEvery;
        set => _flushEvery = Math.Max(1, value);
    }
    private int _opsSinceSync;

    // ---- trace capture (software CUDA-Graph) ----
    // For a fixed-shape step the per-op host cost — building a fresh Tensor/GradNode per op and
    // rebuilding the whole C# graph every step — dominates (≈95% of a small training step; the
    // GPU sits idle). Capture records one replay thunk per kernel launch; Replay re-fires them
    // buffer-to-buffer with zero host graph work. While capturing, every Allocate'd buffer is
    // pinned so the pool can't recycle trace memory out from under a future replay.
    private List<Action>? _capture;   // non-null while capturing: each launch appends a replay thunk
    private int _captureExpected;     // launches seen during capture; must equal _capture.Count
    private HashSet<MemoryBuffer1D<float, Stride1D.Dense>>? _pinned;  // trace-owned; pool must not recycle
    private List<MemoryBuffer1D<float, Stride1D.Dense>>? _capturePins; // this capture's pinned buffers

    public override CapturedGraph Capture(Func<Tensor> body)
    {
        if (_capture != null) throw new InvalidOperationException("Nested trace capture is not supported.");
        Sync();                                  // drain prior work so the trace starts clean
        var thunks = new List<Action>();
        var pins = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        _capture = thunks;
        _capturePins = pins;
        _captureExpected = 0;
        Tensor output;
        try
        {
            output = body();                     // runs the step once eagerly, recording a thunk per launch
        }
        finally
        {
            // Always clear the capture flag — an un-tapped op (or any error) in the body must not
            // leave the runtime stuck in capture mode for every subsequent call.
            _capture = null;
            _capturePins = null;
        }
        Sync();                                  // finish the capture step's device work
        // Reclamation: the graph's pinned buffers stay un-recyclable for its lifetime; disposing the
        // graph un-pins them so their owning tensors can return them to the pool normally. Without
        // this, _pinned grew forever across captures (a leak when many graphs are captured).
        Action onDispose = () => Unpin(pins);

        // Best-effort native CUDA graph: re-fire the recorded launches once under driver stream
        // capture to fold them into ONE executable graph, launched per step with a single
        // cuGraphLaunch instead of N host-side kernel dispatches. Stream capture RECORDS launches
        // without executing them, so this has no numeric effect (body() above already ran the step
        // once). Any driver failure (or a non-CUDA backend) leaves nativeReplay null and the
        // CapturedGraph falls back to re-firing the thunks — correctness never rides on the graph.
        Action? nativeReplay = null;
        if (EnableCudaGraph && _graphStream is CudaStream cs)
        {
            var (exec, graph) = TryBuildCudaGraph(cs, thunks);
            if (exec != IntPtr.Zero)
            {
                IntPtr sp = cs.StreamPtr;
                nativeReplay = () =>
                {
                    // New inputs are uploaded on the default stream (Tensor.Upload); make them visible
                    // before the graph — which runs on _graphStream — reads them.
                    Accelerator.DefaultStream.Synchronize();
                    CuDriver.cuGraphLaunch(exec, sp);
                    Launches++;
                };
                onDispose = () => { Unpin(pins); CuDriver.cuGraphExecDestroy(exec); CuDriver.cuGraphDestroy(graph); };
            }
        }
        return new CapturedGraph(this, thunks, output, nativeReplay, onDispose);
    }

    // nvcuda driver entry points for CUDA Graph capture (present on any machine with the CUDA driver;
    // only invoked on the CUDA backend). A captured stream's launches are recorded into a graph and
    // instantiated into an executable graph that replays with a single cuGraphLaunch.
    private static class CuDriver
    {
        private const string D = "nvcuda";
        [DllImport(D)] public static extern int cuStreamBeginCapture_v2(IntPtr stream, int mode);
        [DllImport(D)] public static extern int cuStreamEndCapture(IntPtr stream, out IntPtr graph);
        [DllImport(D)] public static extern int cuGraphInstantiateWithFlags(out IntPtr exec, IntPtr graph, ulong flags);
        [DllImport(D)] public static extern int cuGraphLaunch(IntPtr exec, IntPtr stream);
        [DllImport(D)] public static extern int cuGraphDestroy(IntPtr graph);
        [DllImport(D)] public static extern int cuGraphExecDestroy(IntPtr exec);
    }

    // Re-fire the recorded thunks once under stream capture, building an executable CUDA graph.
    // Returns (Zero, Zero) on any failure — the caller then uses software replay. The thunks are
    // raw kernel launches (no AfterLaunch / no Drain), so nothing synchronizes mid-capture; GLOBAL
    // capture mode is safe because the runtime is single-threaded and the re-fire allocates nothing
    // (shape/stride int buffers were already cached during the eager body() pass).
    private (IntPtr exec, IntPtr graph) TryBuildCudaGraph(CudaStream cs, List<Action> thunks)
    {
        IntPtr sp = cs.StreamPtr;
        const int CaptureModeGlobal = 0;
        if (CuDriver.cuStreamBeginCapture_v2(sp, CaptureModeGlobal) != 0) return (IntPtr.Zero, IntPtr.Zero);

        // Point the launch stream (and cuBLAS) at the capture stream while re-firing the thunks, then
        // restore — the same launch code records onto _graphStream instead of running on the default.
        var prevStream = _stream;
        var prevBlasStream = _blas?.Stream;
        _stream = cs;
        if (_blas != null) _blas.Stream = cs;
        bool fired = true;
        try { for (int i = 0; i < thunks.Count; i++) thunks[i](); }
        catch { fired = false; }
        finally
        {
            _stream = prevStream;
            if (_blas != null && prevBlasStream != null) _blas.Stream = prevBlasStream;
        }
        if (!fired) { CuDriver.cuStreamEndCapture(sp, out _); return (IntPtr.Zero, IntPtr.Zero); }
        if (CuDriver.cuStreamEndCapture(sp, out var graph) != 0 || graph == IntPtr.Zero)
            return (IntPtr.Zero, IntPtr.Zero);
        if (CuDriver.cuGraphInstantiateWithFlags(out var exec, graph, 0UL) != 0 || exec == IntPtr.Zero)
        {
            CuDriver.cuGraphDestroy(graph);
            return (IntPtr.Zero, IntPtr.Zero);
        }
        return (exec, graph);
    }

    // Release a captured graph's buffers from the pin set (called on CapturedGraph.Dispose). They are
    // not force-freed — their owning tensors recycle them via the normal Dispose path — but they stop
    // blocking pool reuse, so re-capturing or steady-state training doesn't accumulate dead pins.
    private void Unpin(List<MemoryBuffer1D<float, Stride1D.Dense>> pins)
    {
        if (_pinned == null) return;
        foreach (var b in pins) _pinned.Remove(b);
    }

    // Read-only shape/stride/config int buffers recur IDENTICALLY every step (the model's
    // shapes are fixed), so they are cached by content and uploaded to the device exactly
    // once — not re-uploaded and freed per launch. Data-dependent index arrays (gather/scatter
    // idx, pool argmax) are NOT cached (their contents vary per step); those use the parked path.
    private readonly Dictionary<int[], MemoryBuffer1D<int, Stride1D.Dense>> _intCache =
        new(new IntArrayComparer());
    private readonly List<MemoryBuffer1D<int, Stride1D.Dense>> _pendingInts = new();

    // Index/stride buffers: never allocate length 0 (rank-0 tensors) — ILGPU NREs on
    // a zero-length view. The kernels take an explicit `rank`, so padded content is unused.
    // cache=true: recurring shape metadata (cached). cache=false: per-call data-dependent
    // indices (parked, freed on the next drain so they outlive the async kernel).
    private MemoryBuffer1D<int, Stride1D.Dense> AllocInt(int[] a, bool cache = true)
    {
        if (cache && _intCache.TryGetValue(a, out var hit)) return hit;
        Allocs++;
        HostUploads++;
        var buf = Accelerator.Allocate1D(a.Length == 0 ? new[] { 0 } : a);
        if (cache) _intCache[(int[])a.Clone()] = buf;
        else _pendingInts.Add(buf);
        return buf;
    }

    // Called after every async launch: periodic drain to bound the in-flight queue. During trace
    // capture, asserts the launching method recorded a replay thunk (CallerMemberName names it),
    // so an un-tapped op fails loud at capture time rather than silently corrupting a replay.
    private void AfterLaunch([System.Runtime.CompilerServices.CallerMemberName] string caller = "")
    {
        Launches++;
        if (_capture != null && _capture.Count != ++_captureExpected)
            throw new InvalidOperationException(
                $"Trace capture: '{caller}' issued a kernel without recording a replay thunk — this op " +
                "is not trace/replay-supported. Data-dependent index ops (gather, scatter-add, maxpool " +
                "argmax) cannot be replayed because their indices vary per step; keep them outside the " +
                "captured body.");
        if (++_opsSinceSync >= _flushEvery) Drain();
    }

    private void Drain()
    {
        Accelerator.Synchronize();
        foreach (var b in _pendingInts) b.Dispose();
        _pendingInts.Clear();
        _opsSinceSync = 0;
    }

    /// <summary>Drain the default stream and free parked stride buffers. The single sync
    /// point — host pulls (ToArray/Item) call it before reading device memory.</summary>
    public override void Sync() => Drain();

    /// <summary>Stream-ordered device→device copy (no host round-trip). Used by in-place
    /// parameter updates and Clone.</summary>
    public override void DeviceCopy(TensorStorage srcS, TensorStorage dstS)
    {
        var src = B(srcS);
        var dst = B(dstS);
        src.View.CopyTo(_stream, dst.View);
        if (_capture != null) _capture.Add(() => src.View.CopyTo(_stream, dst.View));
        AfterLaunch();
    }

    /// <summary>Stream-ordered device-side zero-fill — no host array, no host→device copy.</summary>
    public override void ZeroBuffer(TensorStorage bufS)
    {
        var buf = B(bufS);
        buf.MemSetToZero(_stream);
        if (_capture != null) _capture.Add(() => buf.MemSetToZero(_stream));
        AfterLaunch();
    }

    private Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> GetBinaryKernel<TOp>()
        where TOp : struct, IBinaryOp
    {
        return (Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>)
            _binaryKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedKernel<
                    Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                    ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.BinaryEltwise<TOp>));
    }

    // ---- launch helpers ----

    public override void LaunchBinary<TOp>(
        TensorStorage aS, TensorStorage bS, TensorStorage outvS,
        int[] outDims, int[] aStride, int[] bStride)
    {
        var a = B(aS); var b = B(bS); var outv = B(outvS);
        var kernel = GetBinaryKernel<TOp>();
        var dOut = AllocInt(outDims);
        var dA = AllocInt(aStride);
        var dB = AllocInt(bStride);
        kernel(_stream, (int)outv.Length, outDims.Length, a.View, b.View, outv.View, dOut.View, dA.View, dB.View);
        if (_capture != null) _capture.Add(() => kernel(_stream, (int)outv.Length, outDims.Length, a.View, b.View, outv.View, dOut.View, dA.View, dB.View));
        AfterLaunch();
    }

    public override void LaunchUnaryFwd<TOp>(TensorStorage xS, TensorStorage outvS)
    {
        var x = B(xS); var outv = B(outvS);
        var kernel = (Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>>)
            _unaryFwdKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryFwd<TOp>));
        kernel(_stream, (int)outv.Length, x.View, outv.View);
        if (_capture != null) _capture.Add(() => kernel(_stream, (int)outv.Length, x.View, outv.View));
        AfterLaunch();
    }

    public override void LaunchUnaryBwd<TOp>(
        TensorStorage xS, TensorStorage yS, TensorStorage gyS, TensorStorage gxS)
    {
        var x = B(xS); var y = B(yS); var gy = B(gyS); var gx = B(gxS);
        var kernel = (Action<AcceleratorStream, Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>)
            _unaryBwdKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryBwd<TOp>));
        kernel(_stream, (int)gx.Length, x.View, y.View, gy.View, gx.View);
        if (_capture != null) _capture.Add(() => kernel(_stream, (int)gx.Length, x.View, y.View, gy.View, gx.View));
        AfterLaunch();
    }

    public override void LaunchUnaryFwdP<TOp>(
        TOp op, TensorStorage xS, TensorStorage outvS)
    {
        var x = B(xS); var outv = B(outvS);
        var kernel = (Action<AcceleratorStream, Index1D, TOp, ArrayView<float>, ArrayView<float>>)
            _unaryFwdPKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedKernel<Index1D, TOp, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryFwdP<TOp>));
        kernel(_stream, (int)outv.Length, op, x.View, outv.View);
        if (_capture != null) _capture.Add(() => kernel(_stream, (int)outv.Length, op, x.View, outv.View));
        AfterLaunch();
    }

    public override void LaunchUnaryBwdP<TOp>(
        TOp op, TensorStorage xS, TensorStorage yS, TensorStorage gyS, TensorStorage gxS)
    {
        var x = B(xS); var y = B(yS); var gy = B(gyS); var gx = B(gxS);
        var kernel = (Action<AcceleratorStream, Index1D, TOp, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>)
            _unaryBwdPKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedKernel<
                    Index1D, TOp, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryBwdP<TOp>));
        kernel(_stream, (int)gx.Length, op, x.View, y.View, gy.View, gx.View);
        if (_capture != null) _capture.Add(() => kernel(_stream, (int)gx.Length, op, x.View, y.View, gy.View, gx.View));
        AfterLaunch();
    }

    public override void LaunchAddInto(TensorStorage targetS, TensorStorage sourceS)
    {
        var target = B(targetS); var source = B(sourceS);
        _addInto(_stream, (int)target.Length, target.View, source.View);
        if (_capture != null) _capture.Add(() => _addInto(_stream, (int)target.Length, target.View, source.View));
        AfterLaunch();
    }

    public override void LaunchReduceSum(
        TensorStorage inpS, TensorStorage outpS,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        var inp = B(inpS); var outp = B(outpS);
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);

        int outputCount = (int)outp.Length;
        long reducedCount = 1;
        for (int ax = 0; ax < reduceMask.Length; ax++)
            if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        // Few outputs + large reduced extent (e.g. conv bias grad: 8 outputs over 50k elements)
        // starves the one-thread-per-output kernel. Split the reduced extent across `parts` threads
        // per output and atomic-combine into a pre-zeroed target; otherwise the parallelism is
        // already fine and the simple kernel avoids the atomics.
        int parts = (int)Math.Min(reducedCount, Math.Max(1, ReduceTargetThreads / Math.Max(1, outputCount)));
        if (parts > 1)
        {
            ZeroBuffer(outpS);
            _reduceSumChunked(_stream, outputCount * parts, inDims.Length, parts, inp.View, outp.View,
                dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
            if (_capture != null) _capture.Add(() => _reduceSumChunked(_stream, outputCount * parts, inDims.Length, parts,
                inp.View, outp.View, dInDims.View, dInStrides.View, dOutDims.View, dMask.View));
        }
        else
        {
            _reduceSum(_stream, outputCount, inDims.Length, inp.View, outp.View,
                dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
            if (_capture != null) _capture.Add(() => _reduceSum(_stream, outputCount, inDims.Length, inp.View, outp.View,
                dInDims.View, dInStrides.View, dOutDims.View, dMask.View));
        }
        AfterLaunch();
    }

    public override void LaunchReduce<TR>(
        TensorStorage inpS, TensorStorage outpS,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        var inp = B(inpS); var outp = B(outpS);
        var kernel = (Action<AcceleratorStream, Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>)
            _reduceKernels.GetOrAdd(typeof(TR), _ =>
                Accelerator.LoadAutoGroupedKernel<
                    Index1D, int, ArrayView<float>, ArrayView<float>,
                    ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.Reduce<TR>));

        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        kernel(_stream, (int)outp.Length, inDims.Length, inp.View, outp.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
        if (_capture != null) _capture.Add(() => kernel(_stream, (int)outp.Length, inDims.Length, inp.View, outp.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View));
        AfterLaunch();
    }

    public override void LaunchReduceArg(
        TensorStorage inpS, TensorStorage outIdxS,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        var inp = B(inpS); var outIdx = B(outIdxS);
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        _reduceArg(_stream, (int)outIdx.Length, inDims.Length, inp.View, outIdx.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0);
        if (_capture != null) _capture.Add(() => _reduceArg(_stream, (int)outIdx.Length, inDims.Length,
            inp.View, outIdx.View, dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0));
        AfterLaunch();
    }

    public override void LaunchMatMul(
        TensorStorage aS, TensorStorage bS, TensorStorage cS,
        int M, int N, int K, int aMs, int aKs, int bKs, int bNs)
    {
        var a = B(aS); var b = B(bS); var c = B(cS);
        // Compute-bound regime: cuBLAS SGEMM on CUDA (vendor-tuned), else the tiled shared-memory
        // kernel. Small/skinny products use the naive one-thread-per-output kernel.
        const int TS = Kernels.MatMulTile;
        bool large = M >= 64 && N >= 64 && K >= 64;
        bool tiledFits = TS * TS <= Accelerator.MaxNumThreadsPerGroup;
        if (large && _blas != null)
        {
            CuBlasGemm(a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs);
            if (_capture != null) _capture.Add(() => CuBlasGemm(a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs));
        }
        else if (large && tiledFits)
        {
            var grid = new Index3D((N + TS - 1) / TS, (M + TS - 1) / TS, 1);
            var group = new Index3D(TS, TS, 1);
            _matmulTiled(_stream, new KernelConfig(grid, group), a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs);
            if (_capture != null) _capture.Add(() => _matmulTiled(_stream, new KernelConfig(grid, group),
                a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs));
        }
        else
        {
            _matmul(_stream, M * N, a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs);
            if (_capture != null) _capture.Add(() => _matmul(_stream, M * N, a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs));
        }
        AfterLaunch();
    }

    // Map our row-major strided matmul to one column-major cuBLAS SGEMM. cuBLAS computes a
    // column-major product; a row-major (R,C) buffer is bit-identical to a column-major (C,R), so
    // we compute Cᵀ = Bᵀ·Aᵀ by swapping operands (cuBLAS-A = our B, cuBLAS-B = our A) with
    // m=N,n=M,k=K and ldc=N. Each operand is a contiguous matrix read either normally (contraction
    // stride 1) or transposed (output-dim stride 1); that maps to the trans flag + leading dim.
    private void CuBlasGemm(
        ArrayView<float> a, ArrayView<float> b, ArrayView<float> c,
        int M, int N, int K, int aMs, int aKs, int bKs, int bNs)
    {
        bool aNormal = aKs == 1;   // our A is row-major (M,K)            (else stored Aᵀ: K×M)
        bool bNormal = bNs == 1;   // our B is row-major (K,N)            (else stored Bᵀ: N×K)
        var transA = bNormal ? CuBlasOperation.NonTranspose : CuBlasOperation.Transpose; // cuBLAS-A = our B
        int ldA = bNormal ? N : K;
        var transB = aNormal ? CuBlasOperation.NonTranspose : CuBlasOperation.Transpose; // cuBLAS-B = our A
        int ldB = aNormal ? K : M;
        _blas!.Gemm(transA, transB, N, M, K, 1f, b, ldA, a, ldB, 0f, c, N);
    }

    // cuBLAS strided-batched SGEMM, P/Invoked directly (ILGPU's CuBlas binds only the single-matrix
    // entry points). cublasSgemmStridedBatched is resolved once from whichever cuBLAS the process
    // already loaded; null when unavailable (non-CUDA-12 toolkits), so callers fall back to a loop.
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int SgemmStridedBatchedFn(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr a, int lda, long strideA,
        IntPtr b, int ldb, long strideB,
        ref float beta, IntPtr c, int ldc, long strideC, int batchCount);

    private static readonly SgemmStridedBatchedFn? SgemmStridedBatched = LoadSgemmStridedBatched();

    public override bool UsesStridedBatchedGemm => EnableStridedBatchedGemm && _blas != null && SgemmStridedBatched != null;

    private static SgemmStridedBatchedFn? LoadSgemmStridedBatched()
    {
        foreach (var name in new[] { "cublas64_12", "cublas64_11", "cublas" })
        {
            try
            {
                if (NativeLibrary.TryLoad(name, out var h) &&
                    NativeLibrary.TryGetExport(h, "cublasSgemmStridedBatched", out var p))
                    return Marshal.GetDelegateForFunctionPointer<SgemmStridedBatchedFn>(p);
            }
            catch { /* try the next candidate name */ }
        }
        return null;
    }

    // True when every consecutive batch matrix sits at a constant element offset (the offset sequence
    // is arithmetic), so the whole batch is one strided-batched GEMM. A fully-broadcast operand (all
    // batch strides 0) yields step 0, which cuBLAS reuses for every matrix. Non-arithmetic offsets
    // (mixed broadcast over several dims) return false → the per-matrix loop handles them. Cheap
    // integer math over the (small) batch.
    private static bool TryConstStride(int[] dims, int[] strides, int count, out long step)
    {
        step = 0;
        if (count <= 1) return true;                 // single matrix: stride is irrelevant
        int rank = dims.Length;
        var idx = new int[rank];
        for (int e = 0; e < count; e++)
        {
            long off = 0;
            for (int j = 0; j < rank; j++) off += (long)idx[j] * strides[j];
            if (e == 1) step = off;                  // e=0 offset is 0; the e=1 offset defines the stride
            else if (e > 1 && off != (long)e * step) return false;
            for (int j = rank - 1; j >= 0; j--) { if (++idx[j] < dims[j]) break; idx[j] = 0; }
        }
        return true;
    }

    // One strided-batched SGEMM for the whole batch, mapped with the same row-major→column-major
    // operand swap as CuBlasGemm (cuBLAS-A = our B, cuBLAS-B = our A; m=N, n=M, k=K, ldc=N). Base
    // pointers are batch element 0 (offset 0); cuBLAS advances by the element strides per matrix.
    private void CuBlasGemmStridedBatched(
        MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int batchCount, int M, int N, int K, int aMs, int aKs, int bKs, int bNs,
        long strideAElems, long strideBElems)
    {
        bool aNormal = aKs == 1;
        bool bNormal = bNs == 1;
        int transA = (int)(bNormal ? CuBlasOperation.NonTranspose : CuBlasOperation.Transpose); // cuBLAS-A = our B
        int ldA = bNormal ? N : K;
        int transB = (int)(aNormal ? CuBlasOperation.NonTranspose : CuBlasOperation.Transpose); // cuBLAS-B = our A
        int ldB = aNormal ? K : M;
        float alpha = 1f, beta = 0f;
        int status = SgemmStridedBatched!(_blas!.Handle, transA, transB, N, M, K,
            ref alpha, b.NativePtr, ldA, strideBElems,   // cuBLAS-A = our B
            a.NativePtr, ldB, strideAElems,              // cuBLAS-B = our A
            ref beta, c.NativePtr, N, (long)M * N, batchCount);
        if (status != 0)
            throw new InvalidOperationException($"cublasSgemmStridedBatched failed (cublasStatus={status}).");
    }

    public override void LaunchReduceArgGrad(
        TensorStorage inpS, TensorStorage goutS, TensorStorage gxS,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        var inp = B(inpS); var gout = B(goutS); var gx = B(gxS);
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        _reduceArgGrad(_stream, (int)gout.Length, inDims.Length, inp.View, gout.View, gx.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0);
        if (_capture != null) _capture.Add(() => _reduceArgGrad(_stream, (int)gout.Length, inDims.Length,
            inp.View, gout.View, gx.View, dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0));
        AfterLaunch();
    }

    public override void LaunchProdGrad(
        TensorStorage inpS, TensorStorage goutS, TensorStorage gxS,
        int[] inDims, int[] inStrides, int[] outStrides, int[] reduceMask)
    {
        var inp = B(inpS); var gout = B(goutS); var gx = B(gxS);
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutStrides = AllocInt(outStrides);
        var dMask = AllocInt(reduceMask);
        _prodGrad(_stream, (int)inp.Length, inDims.Length, inp.View, gout.View, gx.View,
            dInDims.View, dInStrides.View, dOutStrides.View, dMask.View);
        if (_capture != null) _capture.Add(() => _prodGrad(_stream, (int)inp.Length, inDims.Length,
            inp.View, gout.View, gx.View, dInDims.View, dInStrides.View, dOutStrides.View, dMask.View));
        AfterLaunch();
    }

    public override void LaunchStridedCopy(
        TensorStorage inpS, TensorStorage outpS,
        int[] outDims, int[] inStrides, int baseOff = 0)
    {
        var inp = B(inpS); var outp = B(outpS);
        var dOut = AllocInt(outDims);
        var dIn = AllocInt(inStrides);
        _stridedCopy(_stream, (int)outp.Length, outDims.Length, inp.View, outp.View, dOut.View, dIn.View, baseOff);
        if (_capture != null) _capture.Add(() => _stridedCopy(_stream, (int)outp.Length, outDims.Length,
            inp.View, outp.View, dOut.View, dIn.View, baseOff));
        AfterLaunch();
    }

    public override void LaunchScatterAxisRange(
        TensorStorage srcS, TensorStorage dstS,
        int[] srcDims, int[] dstStrides, int axis, int axisOffset)
    {
        var src = B(srcS); var dst = B(dstS);
        var dSrc = AllocInt(srcDims);
        var dDst = AllocInt(dstStrides);
        _scatterAxisRange(_stream, (int)src.Length, srcDims.Length, src.View, dst.View,
            dSrc.View, dDst.View, axis, axisOffset);
        if (_capture != null) _capture.Add(() => _scatterAxisRange(_stream, (int)src.Length, srcDims.Length,
            src.View, dst.View, dSrc.View, dDst.View, axis, axisOffset));
        AfterLaunch();
    }

    // Per-matrix work above which a batched product is worth one cuBLAS SGEMM per element rather
    // than the naive one-thread-per-output kernel. Conv (im2col+matmul) rides this path.
    private const long CuBlasBatchedMinWork = 1L << 18;

    // Target thread count for the chunked parallel reduction.
    private const int ReduceTargetThreads = 8192;

    public override void LaunchMatMulBatched(
        TensorStorage aS, TensorStorage bS, TensorStorage cS,
        int batchCount, int M, int N, int K,
        int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBatchStrides, int[] bBatchStrides)
    {
        var a = B(aS); var b = B(bS); var c = B(cS);
        // Compute-bound batched products run on cuBLAS. When the batch reduces to constant per-matrix
        // element strides (the common bmm/attention case, and the broadcast 2D@3D conv case where a
        // shared operand has stride 0), ONE cublasSgemmStridedBatched call covers the whole batch.
        // ILGPU's CuBlas binds no strided-batched entry point, so it's a direct cuBLAS P/Invoke
        // (resolved across DLL versions; null/failure falls back to the per-matrix SGEMM loop below).
        if (_blas != null && (long)M * N * K >= CuBlasBatchedMinWork)
        {
            if (UsesStridedBatchedGemm &&
                TryConstStride(batchDims, aBatchStrides, batchCount, out long strideA) &&
                TryConstStride(batchDims, bBatchStrides, batchCount, out long strideB))
            {
                void RunStrided() =>
                    CuBlasGemmStridedBatched(a, b, c, batchCount, M, N, K, aMs, aKs, bKs, bNs, strideA, strideB);
                RunStrided();
                if (_capture != null) _capture.Add(RunStrided);
                AfterLaunch();
                return;
            }

            void RunBatched()
            {
                int rank = batchDims.Length;
                var idx = new int[rank];
                for (int e = 0; e < batchCount; e++)
                {
                    long aOff = 0, bOff = 0;
                    for (int j = 0; j < rank; j++)
                    {
                        aOff += (long)idx[j] * aBatchStrides[j];
                        bOff += (long)idx[j] * bBatchStrides[j];
                    }
                    long cOff = (long)e * M * N;
                    CuBlasGemm(
                        a.View.SubView(aOff, (long)M * K),
                        b.View.SubView(bOff, (long)K * N),
                        c.View.SubView(cOff, (long)M * N),
                        M, N, K, aMs, aKs, bKs, bNs);
                    for (int j = rank - 1; j >= 0; j--) { if (++idx[j] < batchDims[j]) break; idx[j] = 0; }
                }
            }
            RunBatched();
            if (_capture != null) _capture.Add(RunBatched);
            AfterLaunch();
            return;
        }

        var dBd = AllocInt(batchDims);
        var dAbs = AllocInt(aBatchStrides);
        var dBbs = AllocInt(bBatchStrides);
        _matmulBatched(_stream, batchCount * M * N, a.View, b.View, c.View,
            batchDims.Length, M, N, K, aMs, aKs, bKs, bNs,
            dBd.View, dAbs.View, dBbs.View);
        if (_capture != null) _capture.Add(() => _matmulBatched(_stream, batchCount * M * N, a.View, b.View, c.View,
            batchDims.Length, M, N, K, aMs, aKs, bKs, bNs, dBd.View, dAbs.View, dBbs.View));
        AfterLaunch();
    }

    // index/gather family. The index host array is uploaded per launch (same
    // correctness-first policy as the stride buffers above).

    public override void LaunchGatherAxis(
        TensorStorage xS, TensorStorage outpS,
        int[] index, int[] outDims, int[] xStrides, int axis, int mode)
    {
        var x = B(xS); var outp = B(outpS);
        var dIdx = AllocInt(index, cache: false); // data-dependent
        var dOut = AllocInt(outDims);
        var dStr = AllocInt(xStrides);
        _gatherAxis(_stream, (int)outp.Length, outDims.Length, x.View, outp.View,
            dIdx.View, dOut.View, dStr.View, axis, mode);
        AfterLaunch();
    }

    public override void LaunchScatterAddAxis(
        TensorStorage gS, TensorStorage gxS,
        int[] index, int[] srcDims, int[] dstStrides, int axis, int mode)
    {
        var g = B(gS); var gx = B(gxS);
        var dIdx = AllocInt(index, cache: false); // data-dependent
        var dSrc = AllocInt(srcDims);
        var dDst = AllocInt(dstStrides);
        _scatterAddAxis(_stream, (int)g.Length, srcDims.Length, g.View, gx.View,
            dIdx.View, dSrc.View, dDst.View, axis, mode);
        AfterLaunch();
    }

    public override void LaunchRepeat(
        TensorStorage xS, TensorStorage outpS,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        var x = B(xS); var outp = B(outpS);
        var dOut = AllocInt(outDims);
        var dInD = AllocInt(inDims);
        var dInS = AllocInt(inStrides);
        _repeat(_stream, (int)outp.Length, outDims.Length, x.View, outp.View,
            dOut.View, dInD.View, dInS.View);
        if (_capture != null) _capture.Add(() => _repeat(_stream, (int)outp.Length, outDims.Length,
            x.View, outp.View, dOut.View, dInD.View, dInS.View));
        AfterLaunch();
    }

    public override void LaunchRepeatGrad(
        TensorStorage gS, TensorStorage gxS,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        var g = B(gS); var gx = B(gxS);
        var dOut = AllocInt(outDims);
        var dInD = AllocInt(inDims);
        var dInS = AllocInt(inStrides);
        _repeatGrad(_stream, (int)g.Length, outDims.Length, g.View, gx.View,
            dOut.View, dInD.View, dInS.View);
        if (_capture != null) _capture.Add(() => _repeatGrad(_stream, (int)g.Length, outDims.Length,
            g.View, gx.View, dOut.View, dInD.View, dInS.View));
        AfterLaunch();
    }

    public override void LaunchIm2Col(TensorStorage xS, TensorStorage colS, int[] cfg)
    {
        var x = B(xS); var col = B(colS);
        var dCfg = AllocInt(cfg);
        _im2col(_stream, (int)col.Length, x.View, col.View, dCfg.View);
        if (_capture != null) _capture.Add(() => _im2col(_stream, (int)col.Length, x.View, col.View, dCfg.View));
        AfterLaunch();
    }

    public override void LaunchCol2Im(TensorStorage gcolS, TensorStorage gxS, int[] cfg)
    {
        var gcol = B(gcolS); var gx = B(gxS);
        var dCfg = AllocInt(cfg);
        _col2im(_stream, (int)gcol.Length, gcol.View, gx.View, dCfg.View);
        if (_capture != null) _capture.Add(() => _col2im(_stream, (int)gcol.Length, gcol.View, gx.View, dCfg.View));
        AfterLaunch();
    }

    // MaxPool forward writes the per-output argmax (flat input offsets). For the training path we
    // keep that buffer ON THE DEVICE and hand it straight to backward — no host sync, no readback.
    // In no-grad we don't need it, so we drain once and free it.
    public override object? LaunchMaxPool2d(
        TensorStorage xS, TensorStorage outpS, int[] cfg, bool keepArgmax)
    {
        var x = B(xS); var outp = B(outpS);
        var dCfg = AllocInt(cfg);
        var dArg = Accelerator.Allocate1D<int>(outp.Length);
        _maxPool2d(_stream, (int)outp.Length, x.View, outp.View, dArg.View, dCfg.View);
        AfterLaunch();
        if (keepArgmax) return dArg;          // device-resident; backward consumes it directly
        Sync();                                // no-grad: drain so the free can't race the kernel
        dArg.Dispose();
        return null;
    }

    public override void LaunchMaxPool2dGrad(
        TensorStorage gS, TensorStorage gxS, object argmax)
    {
        var g = B(gS); var gx = B(gxS);
        var dArg = (MemoryBuffer1D<int, Stride1D.Dense>)argmax;
        _maxPool2dGrad(_stream, (int)g.Length, g.View, gx.View, dArg.View);
        AfterLaunch();
    }

    public override void LaunchAvgPool2d(TensorStorage xS, TensorStorage outpS, int[] cfg)
    {
        var x = B(xS); var outp = B(outpS);
        var dCfg = AllocInt(cfg);
        _avgPool2d(_stream, (int)outp.Length, x.View, outp.View, dCfg.View);
        if (_capture != null) _capture.Add(() => _avgPool2d(_stream, (int)outp.Length, x.View, outp.View, dCfg.View));
        AfterLaunch();
    }

    public override void LaunchAvgPool2dGrad(TensorStorage gS, TensorStorage gxS, int[] cfg)
    {
        var g = B(gS); var gx = B(gxS);
        var dCfg = AllocInt(cfg);
        _avgPool2dGrad(_stream, (int)g.Length, g.View, gx.View, dCfg.View);
        if (_capture != null) _capture.Add(() => _avgPool2dGrad(_stream, (int)g.Length, g.View, gx.View, dCfg.View));
        AfterLaunch();
    }

    /// <summary>Advance a parameter's device-resident Adam state (see <see cref="Kernels.AdvanceAdam"/>).</summary>
    public override void LaunchAdvanceAdam(TensorStorage advS, float b1, float b2)
    {
        var adv = B(advS);
        _advanceAdam(_stream, 1, adv.View, b1, b2);
        if (_capture != null) _capture.Add(() => _advanceAdam(_stream, 1, adv.View, b1, b2));
        AfterLaunch();
    }

    /// <summary>Fused Adam/AdamW update (in place). See <see cref="Kernels.AdamStep"/>. Bias
    /// corrections are read from <paramref name="advS"/>[1..3] (written by <see cref="LaunchAdvanceAdam"/>).</summary>
    public override void LaunchAdam(
        TensorStorage pS, TensorStorage gS, TensorStorage mS, TensorStorage vS,
        float b1, float oneMinusB1, float b2, float oneMinusB2,
        float lr, float eps, TensorStorage advS,
        float coupledWd, float decoupledFactor)
    {
        var p = B(pS); var g = B(gS); var m = B(mS); var v = B(vS);
        var bc = B(advS).View.SubView(1, 2);          // [invBc1, invBc2]
        _adamStep(_stream, (int)p.Length, p.View, g.View, m.View, v.View,
            b1, oneMinusB1, b2, oneMinusB2, lr, eps, bc, coupledWd, decoupledFactor);
        if (_capture != null) _capture.Add(() => _adamStep(_stream, (int)p.Length, p.View, g.View, m.View, v.View,
            b1, oneMinusB1, b2, oneMinusB2, lr, eps, bc, coupledWd, decoupledFactor));
        AfterLaunch();
    }

    /// <summary>Fused SGD update (in place). See <see cref="Kernels.SgdStep"/>.</summary>
    public override void LaunchSgd(
        TensorStorage pS, TensorStorage gS, TensorStorage bufS,
        float lr, float momentum, float weightDecay, float dampening, float nesterov, float hasBuf)
    {
        var p = B(pS); var g = B(gS); var buf = B(bufS);
        _sgdStep(_stream, (int)p.Length, p.View, g.View, buf.View,
            lr, momentum, weightDecay, dampening, nesterov, hasBuf);
        if (_capture != null) _capture.Add(() => _sgdStep(_stream, (int)p.Length, p.View, g.View, buf.View,
            lr, momentum, weightDecay, dampening, nesterov, hasBuf));
        AfterLaunch();
    }

    public override void Dispose()
    {
        _blas?.Dispose();
        foreach (var stack in _floatPool.Values)
            foreach (var b in stack) b.Dispose();
        _floatPool.Clear();
        foreach (var b in _intCache.Values) b.Dispose();
        _intCache.Clear();
        foreach (var b in _pendingInts) b.Dispose();
        _pendingInts.Clear();
        _graphStream?.Dispose();
        Accelerator.Dispose();
        Context.Dispose();
    }
}
