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

    public MemoryBuffer1D<float, Stride1D.Dense> Allocate(int length)
        => Accelerator.Allocate1D<float>(length);

    // Index/stride buffers: never allocate length 0 (rank-0 tensors) — ILGPU NREs on
    // a zero-length view. The kernels take an explicit `rank`, so padded content is unused.
    private MemoryBuffer1D<int, Stride1D.Dense> AllocInt(int[] a)
        => Accelerator.Allocate1D(a.Length == 0 ? new[] { 0 } : a);

    public void Sync() => Accelerator.Synchronize();

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
        using var dOut = AllocInt(outDims);
        using var dA = AllocInt(aStride);
        using var dB = AllocInt(bStride);
        kernel((int)outv.Length, outDims.Length, a.View, b.View, outv.View, dOut.View, dA.View, dB.View);
        Sync();
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
        Sync();
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
        Sync();
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
        Sync();
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
        Sync();
    }

    public void LaunchAddInto(
        MemoryBuffer1D<float, Stride1D.Dense> target,
        MemoryBuffer1D<float, Stride1D.Dense> source)
    {
        _addInto((int)target.Length, target.View, source.View);
        Sync();
    }

    public void LaunchReduceSum(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        using var dInDims = AllocInt(inDims);
        using var dInStrides = AllocInt(inStrides);
        using var dOutDims = AllocInt(outDims);
        using var dMask = AllocInt(reduceMask);
        _reduceSum((int)outp.Length, inDims.Length, inp.View, outp.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
        Sync();
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

        using var dInDims = AllocInt(inDims);
        using var dInStrides = AllocInt(inStrides);
        using var dOutDims = AllocInt(outDims);
        using var dMask = AllocInt(reduceMask);
        kernel((int)outp.Length, inDims.Length, inp.View, outp.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View);
        Sync();
    }

    public void LaunchReduceArg(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outIdx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        using var dInDims = AllocInt(inDims);
        using var dInStrides = AllocInt(inStrides);
        using var dOutDims = AllocInt(outDims);
        using var dMask = AllocInt(reduceMask);
        _reduceArg((int)outIdx.Length, inDims.Length, inp.View, outIdx.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0);
        Sync();
    }

    public void LaunchMatMul(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int M, int N, int K,
        int aMs, int aKs, int bKs, int bNs)
    {
        _matmul(M * N, a.View, b.View, c.View, M, N, K, aMs, aKs, bKs, bNs);
        Sync();
    }

    public void LaunchReduceArgGrad(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> gout,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        using var dInDims = AllocInt(inDims);
        using var dInStrides = AllocInt(inStrides);
        using var dOutDims = AllocInt(outDims);
        using var dMask = AllocInt(reduceMask);
        _reduceArgGrad((int)gout.Length, inDims.Length, inp.View, gout.View, gx.View,
            dInDims.View, dInStrides.View, dOutDims.View, dMask.View, isMax ? 1 : 0);
        Sync();
    }

    public void LaunchProdGrad(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> gout,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] inDims, int[] inStrides, int[] outStrides, int[] reduceMask)
    {
        using var dInDims = AllocInt(inDims);
        using var dInStrides = AllocInt(inStrides);
        using var dOutStrides = AllocInt(outStrides);
        using var dMask = AllocInt(reduceMask);
        _prodGrad((int)inp.Length, inDims.Length, inp.View, gout.View, gx.View,
            dInDims.View, dInStrides.View, dOutStrides.View, dMask.View);
        Sync();
    }

    public void LaunchStridedCopy(
        MemoryBuffer1D<float, Stride1D.Dense> inp,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] outDims, int[] inStrides, int baseOff = 0)
    {
        using var dOut = AllocInt(outDims);
        using var dIn = AllocInt(inStrides);
        _stridedCopy((int)outp.Length, outDims.Length, inp.View, outp.View, dOut.View, dIn.View, baseOff);
        Sync();
    }

    public void LaunchScatterAxisRange(
        MemoryBuffer1D<float, Stride1D.Dense> src,
        MemoryBuffer1D<float, Stride1D.Dense> dst,
        int[] srcDims, int[] dstStrides, int axis, int axisOffset)
    {
        using var dSrc = AllocInt(srcDims);
        using var dDst = AllocInt(dstStrides);
        _scatterAxisRange((int)src.Length, srcDims.Length, src.View, dst.View,
            dSrc.View, dDst.View, axis, axisOffset);
        Sync();
    }

    public void LaunchMatMulBatched(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int batchCount, int M, int N, int K,
        int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBatchStrides, int[] bBatchStrides)
    {
        using var dBd = AllocInt(batchDims);
        using var dAbs = AllocInt(aBatchStrides);
        using var dBbs = AllocInt(bBatchStrides);
        _matmulBatched(batchCount * M * N, a.View, b.View, c.View,
            batchDims.Length, M, N, K, aMs, aKs, bKs, bNs,
            dBd.View, dAbs.View, dBbs.View);
        Sync();
    }

    // index/gather family. The index host array is uploaded per launch (same
    // correctness-first policy as the stride buffers above).

    public void LaunchGatherAxis(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] index, int[] outDims, int[] xStrides, int axis, int mode)
    {
        using var dIdx = AllocInt(index);
        using var dOut = AllocInt(outDims);
        using var dStr = AllocInt(xStrides);
        _gatherAxis((int)outp.Length, outDims.Length, x.View, outp.View,
            dIdx.View, dOut.View, dStr.View, axis, mode);
        Sync();
    }

    public void LaunchScatterAddAxis(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] index, int[] srcDims, int[] dstStrides, int axis, int mode)
    {
        using var dIdx = AllocInt(index);
        using var dSrc = AllocInt(srcDims);
        using var dDst = AllocInt(dstStrides);
        _scatterAddAxis((int)g.Length, srcDims.Length, g.View, gx.View,
            dIdx.View, dSrc.View, dDst.View, axis, mode);
        Sync();
    }

    public void LaunchRepeat(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        using var dOut = AllocInt(outDims);
        using var dInD = AllocInt(inDims);
        using var dInS = AllocInt(inStrides);
        _repeat((int)outp.Length, outDims.Length, x.View, outp.View,
            dOut.View, dInD.View, dInS.View);
        Sync();
    }

    public void LaunchRepeatGrad(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        using var dOut = AllocInt(outDims);
        using var dInD = AllocInt(inDims);
        using var dInS = AllocInt(inStrides);
        _repeatGrad((int)g.Length, outDims.Length, g.View, gx.View,
            dOut.View, dInD.View, dInS.View);
        Sync();
    }

    public void LaunchIm2Col(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> col,
        int[] cfg)
    {
        using var dCfg = AllocInt(cfg);
        _im2col((int)col.Length, x.View, col.View, dCfg.View);
        Sync();
    }

    public void LaunchCol2Im(
        MemoryBuffer1D<float, Stride1D.Dense> gcol,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] cfg)
    {
        using var dCfg = AllocInt(cfg);
        _col2im((int)gcol.Length, gcol.View, gx.View, dCfg.View);
        Sync();
    }

    // MaxPool forward returns the per-output argmax (flat input offsets) downloaded to the
    // host, so the backward closure can route gradients without holding a device buffer.
    public int[] LaunchMaxPool2d(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] cfg)
    {
        using var dCfg = AllocInt(cfg);
        using var dArg = Accelerator.Allocate1D<int>(outp.Length);
        _maxPool2d((int)outp.Length, x.View, outp.View, dArg.View, dCfg.View);
        Sync();
        return dArg.GetAsArray1D();
    }

    public void LaunchMaxPool2dGrad(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] argmax)
    {
        using var dArg = AllocInt(argmax);
        _maxPool2dGrad((int)g.Length, g.View, gx.View, dArg.View);
        Sync();
    }

    public void LaunchAvgPool2d(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> outp,
        int[] cfg)
    {
        using var dCfg = AllocInt(cfg);
        _avgPool2d((int)outp.Length, x.View, outp.View, dCfg.View);
        Sync();
    }

    public void LaunchAvgPool2dGrad(
        MemoryBuffer1D<float, Stride1D.Dense> g,
        MemoryBuffer1D<float, Stride1D.Dense> gx,
        int[] cfg)
    {
        using var dCfg = AllocInt(cfg);
        _avgPool2dGrad((int)g.Length, g.View, gx.View, dCfg.View);
        Sync();
    }

    public void Dispose()
    {
        Accelerator.Dispose();
        Context.Dispose();
    }
}
