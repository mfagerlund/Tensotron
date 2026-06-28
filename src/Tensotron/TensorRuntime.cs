using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace Tensotron;

/// <summary>
/// Which ILGPU accelerator the runtime drives. <see cref="Auto"/> prefers a CUDA GPU and falls
/// back to the CPU accelerator; <see cref="Cuda"/> / <see cref="Cpu"/> force one.
/// NOTE: ILGPU (through 1.5.3, its latest) has no SIMD/vectorized CPU accelerator — <see cref="Cpu"/>
/// is ILGPU's scalar CPUAccelerator. A fast SIMD CPU path, if ever added, would be a separate
/// hand-written backend, not an ILGPU device.
/// </summary>
public enum TensorBackend { Auto, Cuda, Cpu }

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
/// Owns the single ILGPU Context + Accelerator and caches compiled kernels.
/// One accelerator, kernels loaded once and cached. The backend is chosen once, at first use,
/// from <see cref="RequestedBackend"/> (CUDA-preferred by default so tests run anywhere).
/// </summary>
public sealed class TensorRuntime : IDisposable
{
    private static readonly Lazy<TensorRuntime> _instance = new(() => new TensorRuntime());
    public static TensorRuntime Instance => _instance.Value;

    /// <summary>
    /// Backend to use, read once when the runtime is first created. Set this before touching any
    /// tensor, or set the <c>TENSOTRON_BACKEND</c> env var (auto|cuda|velocity|cpu). Tensotron
    /// runs on exactly one accelerator process-wide, so — like PyTorch's device model — tensors
    /// never mix backends; selecting here picks that single backend.
    /// </summary>
    public static TensorBackend RequestedBackend { get; set; } = ParseBackendEnv();

    private static TensorBackend ParseBackendEnv() =>
        (Environment.GetEnvironmentVariable("TENSOTRON_BACKEND")?.Trim().ToLowerInvariant()) switch
        {
            "cuda" or "gpu" => TensorBackend.Cuda,
            "cpu" => TensorBackend.Cpu,
            _ => TensorBackend.Auto,
        };

    public Context Context { get; }
    public Accelerator Accelerator { get; }
    public string DeviceName => Accelerator.Name;
    public bool IsGpu => Accelerator.AcceleratorType != AcceleratorType.CPU;

    // Non-generic kernels: loaded once into fields.
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _addInto;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _reduceSum;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        int, int, int, int, int, int, int> _matmul;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, int> _stridedCopy;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, int, int> _scatterAxisRange;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        int, int, int, int, int, int, int, int,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _matmulBatched;

    // Generic elementwise kernels: one per op struct, cached by type.
    private readonly ConcurrentDictionary<Type, object> _binaryKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryFwdKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryBwdKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryFwdPKernels = new();
    private readonly ConcurrentDictionary<Type, object> _unaryBwdPKernels = new();
    private readonly ConcurrentDictionary<Type, object> _reduceKernels = new();
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _reduceArg;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _reduceArgGrad;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _prodGrad;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _gatherAxis;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _scatterAddAxis;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _repeat;
    private readonly Action<Index1D, int, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> _repeatGrad;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _im2col;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _col2im;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>> _maxPool2d;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _maxPool2dGrad;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _avgPool2d;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>> _avgPool2dGrad;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        float, float, float, float, float, float, float, float, float, float> _adamStep;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        float, float, float, float, float, float> _sgdStep;

    private TensorRuntime()
    {
        // Enable CPU + all GPUs (Default), then pick one device per the requested backend.
        Context = Context.Create(b => b.Default().EnableAlgorithms());
        Accelerator = SelectDevice(Context, RequestedBackend).CreateAccelerator(Context);

        _addInto = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(Kernels.AddInto);

        _reduceSum = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.ReduceSum);

        _matmul = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            int, int, int, int, int, int, int>(Kernels.MatMul2D);

        _reduceArg = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(Kernels.ReduceArg);

        _reduceArgGrad = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(Kernels.ReduceArgGrad);

        _prodGrad = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.ProdGrad);

        _stridedCopy = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, int>(Kernels.StridedCopy);

        _scatterAxisRange = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, int, int>(Kernels.ScatterAxisRange);

        _matmulBatched = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            int, int, int, int, int, int, int, int,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.MatMulBatched);

        _gatherAxis = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Kernels.GatherAxis);

        _scatterAddAxis = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Kernels.ScatterAddAxis);

        _repeat = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.Repeat);

        _repeatGrad = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.RepeatGrad);

        _im2col = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.Im2Col);
        _col2im = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.Col2Im);

        _maxPool2d = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<int>>(Kernels.MaxPool2d);
        _maxPool2dGrad = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.MaxPool2dGrad);
        _avgPool2d = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.AvgPool2d);
        _avgPool2dGrad = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>>(Kernels.AvgPool2dGrad);

        _adamStep = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            float, float, float, float, float, float, float, float, float, float>(Kernels.AdamStep);
        _sgdStep = Accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            float, float, float, float, float, float>(Kernels.SgdStep);
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

    // ---- lightweight instrumentation (perf bench only; negligible when unread) ----
    /// <summary>Kernel launches + device copies issued (each <see cref="AfterLaunch"/>).</summary>
    public long Launches { get; private set; }
    /// <summary>Device buffer allocations (float results + int stride/dim buffers).</summary>
    public long Allocs { get; private set; }
    /// <summary>Host→device uploads (scalar/leaf CopyFromCPU + per-launch stride buffers).</summary>
    public long HostUploads { get; private set; }
    public void ResetCounters() { Launches = 0; Allocs = 0; HostUploads = 0; }
    internal void NoteHostUpload() => HostUploads++;

    public MemoryBuffer1D<float, Stride1D.Dense> Allocate(int length)
    {
        Allocs++;
        return Accelerator.Allocate1D<float>(length);
    }

    // ---- async batching ----
    // Kernels launch on the accelerator's in-order default stream and are NOT synced per
    // launch. The stream is drained (a) at every host pull (Sync, via ToArray/Item) and
    // (b) every FlushEvery launches as a safety valve to bound the in-flight queue.
    private const int FlushEvery = 64;
    private int _opsSinceSync;

    // Read-only shape/stride/config int buffers recur IDENTICALLY every step (the model's
    // shapes are fixed), so they are cached by content and uploaded to the device exactly
    // once — not re-uploaded and freed per launch. Bounded in practice by the number of
    // distinct shapes the model touches. Data-dependent index arrays (gather/scatter idx,
    // pool argmax) are NOT cached (their contents vary per step, which would grow the cache
    // unbounded); those use the parked-and-freed path below.
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

    // Called after every async launch: periodic drain to bound the in-flight queue.
    private void AfterLaunch()
    {
        Launches++;
        if (++_opsSinceSync >= FlushEvery) Drain();
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
    public void Sync() => Drain();

    /// <summary>Stream-ordered device→device copy (no host round-trip). Used by in-place
    /// parameter updates and Clone, which previously bounced through host memory.</summary>
    public void DeviceCopy(MemoryBuffer1D<float, Stride1D.Dense> src,
                           MemoryBuffer1D<float, Stride1D.Dense> dst)
    {
        src.View.CopyTo(Accelerator.DefaultStream, dst.View);
        AfterLaunch();
    }

    private Action<Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<int>, ArrayView<int>, ArrayView<int>> GetBinaryKernel<TOp>()
        where TOp : struct, IBinaryOp
    {
        return (Action<Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>>)
            _binaryKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, int, ArrayView<float>, ArrayView<float>, ArrayView<float>,
                    ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.BinaryEltwise<TOp>));
    }

    // ---- launch helpers (correctness-first: temp int buffers are synced+freed per call;
    //      TODO: cache stride buffers / use constant memory to keep launches async) ----

    public void LaunchBinary<TOp>(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> outv,
        int[] outDims, int[] aStride, int[] bStride)
        where TOp : struct, IBinaryOp
    {
        var kernel = GetBinaryKernel<TOp>();
        var dOut = AllocInt(outDims);
        var dA = AllocInt(aStride);
        var dB = AllocInt(bStride);
        kernel((int)outv.Length, outDims.Length, a.View, b.View, outv.View, dOut.View, dA.View, dB.View);
        AfterLaunch();
    }

    public void LaunchUnaryFwd<TOp>(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outv)
        where TOp : struct, IUnaryOp
    {
        var kernel = (Action<Index1D, ArrayView<float>, ArrayView<float>>)
            _unaryFwdKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryFwd<TOp>));
        kernel((int)outv.Length, x.View, outv.View);
        AfterLaunch();
    }

    public void LaunchUnaryBwd<TOp>(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        MemoryBuffer1D<float, Stride1D.Dense> gy,
        MemoryBuffer1D<float, Stride1D.Dense> gx)
        where TOp : struct, IUnaryOp
    {
        var kernel = (Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>)
            _unaryBwdKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryBwd<TOp>));
        kernel((int)gx.Length, x.View, y.View, gy.View, gx.View);
        AfterLaunch();
    }

    public void LaunchUnaryFwdP<TOp>(
        TOp op,
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outv)
        where TOp : unmanaged, IUnaryOp
    {
        var kernel = (Action<Index1D, TOp, ArrayView<float>, ArrayView<float>>)
            _unaryFwdPKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedStreamKernel<Index1D, TOp, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryFwdP<TOp>));
        kernel((int)outv.Length, op, x.View, outv.View);
        AfterLaunch();
    }

    public void LaunchUnaryBwdP<TOp>(
        TOp op,
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        MemoryBuffer1D<float, Stride1D.Dense> gy,
        MemoryBuffer1D<float, Stride1D.Dense> gx)
        where TOp : unmanaged, IUnaryOp
    {
        var kernel = (Action<Index1D, TOp, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>)
            _unaryBwdPKernels.GetOrAdd(typeof(TOp), _ =>
                Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, TOp, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
                    Kernels.UnaryBwdP<TOp>));
        kernel((int)gx.Length, op, x.View, y.View, gy.View, gx.View);
        AfterLaunch();
    }

    public void LaunchAddInto(
        MemoryBuffer1D<float, Stride1D.Dense> target,
        MemoryBuffer1D<float, Stride1D.Dense> source)
    {
        _addInto((int)target.Length, target.View, source.View);
        AfterLaunch();
    }

    public void LaunchReduceSum(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        _reduceSum((int)outp.Length, inDims.Length, inp.View, outp.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
        AfterLaunch();
    }

    public void LaunchReduce<TR>(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
        where TR : struct, IReduceOp
    {
        var kernel = (Action<Index1D, int, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>)
            _reduceKernels.GetOrAdd(typeof(TR), _ =>
                Accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, int, ArrayView<float>, ArrayView<float>,
                    ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(Kernels.Reduce<TR>));

        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        kernel((int)outp.Length, inDims.Length, inp.View, outp.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
        AfterLaunch();
    }

    public void LaunchReduceArg(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outIdx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        _reduceArg((int)outIdx.Length, inDims.Length, inp.View, outIdx.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0);
        AfterLaunch();
    }

    public void LaunchMatMul(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int M, int N, int K,
        int aMs, int aKs, int bKs, int bNs)
    {
        _matmul(M * N, a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs);
        AfterLaunch();
    }

    public void LaunchReduceArgGrad(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> gout,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutDims = AllocInt(outDims);
        var dMask = AllocInt(reduceMask);
        _reduceArgGrad((int)gout.Length, inDims.Length, inp.View, gout.View, gx.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0);
        AfterLaunch();
    }

    public void LaunchProdGrad(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> gout,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] inDims, int[] inStrides, int[] outStrides, int[] reduceMask)
    {
        var dInDims = AllocInt(inDims);
        var dInStrides = AllocInt(inStrides);
        var dOutStrides = AllocInt(outStrides);
        var dMask = AllocInt(reduceMask);
        _prodGrad((int)inp.Length, inDims.Length, inp.View, gout.View, gx.View,
            dInDims.View, dInStrides.View, dOutStrides.View, dMask.View);
        AfterLaunch();
    }

    public void LaunchStridedCopy(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] outDims, int[] inStrides, int baseOff = 0)
    {
        var dOut = AllocInt(outDims);
        var dIn = AllocInt(inStrides);
        _stridedCopy((int)outp.Length, outDims.Length, inp.View, outp.View, dOut.View, dIn.View, baseOff);
        AfterLaunch();
    }

    public void LaunchScatterAxisRange(
        MemoryBuffer1D<float, Stride1D.Dense> src,
        MemoryBuffer1D<float, Stride1D.Dense> dst,
        int[] srcDims, int[] dstStrides, int axis, int axisOffset)
    {
        var dSrc = AllocInt(srcDims);
        var dDst = AllocInt(dstStrides);
        _scatterAxisRange((int)src.Length, srcDims.Length, src.View, dst.View,
            dSrc.View, dDst.View, axis, axisOffset);
        AfterLaunch();
    }

    public void LaunchMatMulBatched(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int batchCount, int M, int N, int K,
        int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBatchStrides, int[] bBatchStrides)
    {
        var dBd = AllocInt(batchDims);
        var dAbs = AllocInt(aBatchStrides);
        var dBbs = AllocInt(bBatchStrides);
        _matmulBatched(batchCount * M * N, a.View, b.View, c.View,
            batchDims.Length, M, N, K, aMs, aKs, bKs, bNs,
            dBd.View, dAbs.View, dBbs.View);
        AfterLaunch();
    }

    // index/gather family. The index host array is uploaded per launch (same
    // correctness-first policy as the stride buffers above).

    public void LaunchGatherAxis(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] index, int[] outDims, int[] xStrides, int axis, int mode)
    {
        var dIdx = AllocInt(index, cache: false); // data-dependent
        var dOut = AllocInt(outDims);
        var dStr = AllocInt(xStrides);
        _gatherAxis((int)outp.Length, outDims.Length, x.View, outp.View,
            dIdx.View, dOut.View, dStr.View, axis, mode);
        AfterLaunch();
    }

    public void LaunchScatterAddAxis(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] index, int[] srcDims, int[] dstStrides, int axis, int mode)
    {
        var dIdx = AllocInt(index, cache: false); // data-dependent
        var dSrc = AllocInt(srcDims);
        var dDst = AllocInt(dstStrides);
        _scatterAddAxis((int)g.Length, srcDims.Length, g.View, gx.View,
            dIdx.View, dSrc.View, dDst.View, axis, mode);
        AfterLaunch();
    }

    public void LaunchRepeat(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        var dOut = AllocInt(outDims);
        var dInD = AllocInt(inDims);
        var dInS = AllocInt(inStrides);
        _repeat((int)outp.Length, outDims.Length, x.View, outp.View,
            dOut.View, dInD.View, dInS.View);
        AfterLaunch();
    }

    public void LaunchRepeatGrad(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        var dOut = AllocInt(outDims);
        var dInD = AllocInt(inDims);
        var dInS = AllocInt(inStrides);
        _repeatGrad((int)g.Length, outDims.Length, g.View, gx.View,
            dOut.View, dInD.View, dInS.View);
        AfterLaunch();
    }

    public void LaunchIm2Col(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> col,
        int[] cfg)
    {
        var dCfg = AllocInt(cfg);
        _im2col((int)col.Length, x.View, col.View, dCfg.View);
        AfterLaunch();
    }

    public void LaunchCol2Im(
        MemoryBuffer1D<float, Stride1D.Dense> gcol,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] cfg)
    {
        var dCfg = AllocInt(cfg);
        _col2im((int)gcol.Length, gcol.View, gx.View, dCfg.View);
        AfterLaunch();
    }

    // MaxPool forward returns the per-output argmax (flat input offsets) downloaded to the
    // host, so the backward closure can route gradients without holding a device buffer.
    public int[] LaunchMaxPool2d(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] cfg)
    {
        var dCfg = AllocInt(cfg);
        using var dArg = Accelerator.Allocate1D<int>(outp.Length);
        _maxPool2d((int)outp.Length, x.View, outp.View, dArg.View, dCfg.View);
        Sync(); // host readback of the argmax indices below: drain before reading
        return dArg.GetAsArray1D();
    }

    public void LaunchMaxPool2dGrad(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] argmax)
    {
        var dArg = AllocInt(argmax, cache: false); // data-dependent
        _maxPool2dGrad((int)g.Length, g.View, gx.View, dArg.View);
        AfterLaunch();
    }

    public void LaunchAvgPool2d(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] cfg)
    {
        var dCfg = AllocInt(cfg);
        _avgPool2d((int)outp.Length, x.View, outp.View, dCfg.View);
        AfterLaunch();
    }

    public void LaunchAvgPool2dGrad(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] cfg)
    {
        var dCfg = AllocInt(cfg);
        _avgPool2dGrad((int)g.Length, g.View, gx.View, dCfg.View);
        AfterLaunch();
    }

    /// <summary>Fused Adam/AdamW update (in place). See <see cref="Kernels.AdamStep"/>.</summary>
    public void LaunchAdam(
        MemoryBuffer1D<float, Stride1D.Dense> p,
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> m,
        MemoryBuffer1D<float, Stride1D.Dense> v,
        float b1, float oneMinusB1, float b2, float oneMinusB2,
        float lr, float eps, float invBc1, float invBc2,
        float coupledWd, float decoupledFactor)
    {
        _adamStep((int)p.Length, p.View, g.View, m.View, v.View,
            b1, oneMinusB1, b2, oneMinusB2, lr, eps, invBc1, invBc2, coupledWd, decoupledFactor);
        AfterLaunch();
    }

    /// <summary>Fused SGD update (in place). See <see cref="Kernels.SgdStep"/>.</summary>
    public void LaunchSgd(
        MemoryBuffer1D<float, Stride1D.Dense> p,
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> buf,
        float lr, float momentum, float weightDecay, float dampening, float nesterov, float hasBuf)
    {
        _sgdStep((int)p.Length, p.View, g.View, buf.View,
            lr, momentum, weightDecay, dampening, nesterov, hasBuf);
        AfterLaunch();
    }

    public void Dispose()
    {
        foreach (var b in _intCache.Values) b.Dispose();
        _intCache.Clear();
        foreach (var b in _pendingInts) b.Dispose();
        _pendingInts.Clear();
        Accelerator.Dispose();
        Context.Dispose();
    }
}
