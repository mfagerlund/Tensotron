using ILGPU;

namespace Tensotron;

/// <summary>
/// Hand-written CPU backend: tensors live in managed <c>float[]</c> (<see cref="HostStorage"/>) and
/// every op runs as a synchronous managed (scalar, then SIMD) kernel — no ILGPU, no per-op device
/// dispatch, no stream. This is the fast path for small-model CPU inference/training, where ILGPU's
/// scalar CPUAccelerator is dominated by dispatch overhead (measured ~274× vs hand-scalar at batch=1).
///
/// The math lives in <see cref="CpuKernels"/> (the managed twin of <see cref="Kernels"/>). Storage is
/// the managed array directly, so there is nothing to synchronize and no host↔device transfer.
/// </summary>
internal sealed class CpuSimdRuntime : TensorRuntime
{
    public override string DeviceName => "CPU-SIMD";
    public override bool IsGpu => false;

    // Lazily created only if device enumeration is requested (Cuda.* / Accelerators.*). Pure
    // CpuSimd compute never touches ILGPU.
    private Context? _ctx;
    public override Context Context => _ctx ??= Context.Create(b => b.Default());

    private static float[] H(TensorStorage s) => ((HostStorage)s).Data;

    // No stream: nothing is ever in flight, so Sync is a no-op and capture/replay buys nothing.
    public override void Sync() { }

    public override TensorStorage Allocate(int length)
    {
        Allocs++;
        return new HostStorage(new float[length]);
    }

    internal override void ReturnToPool(TensorStorage buf) { /* managed array: GC reclaims it */ }

    public override void ZeroBuffer(TensorStorage buf) => Array.Clear(H(buf));

    public override void DeviceCopy(TensorStorage src, TensorStorage dst)
    {
        var s = H(src); var d = H(dst);
        Array.Copy(s, d, s.Length);
    }

    // ---- compute surface (managed kernels in CpuKernels) ----

    public override void LaunchBinary<TOp>(TensorStorage a, TensorStorage b, TensorStorage outv,
        int[] outDims, int[] aStride, int[] bStride)
    {
        CpuKernels.Binary<TOp>(H(a), H(b), H(outv), outDims, aStride, bStride);
        Launches++;
    }

    public override void LaunchUnaryFwd<TOp>(TensorStorage x, TensorStorage outv)
    {
        CpuKernels.UnaryFwd<TOp>(H(x), H(outv));
        Launches++;
    }

    public override void LaunchUnaryBwd<TOp>(TensorStorage x, TensorStorage y, TensorStorage gy, TensorStorage gx)
    {
        CpuKernels.UnaryBwd<TOp>(H(x), H(y), H(gy), H(gx));
        Launches++;
    }

    public override void LaunchUnaryFwdP<TOp>(TOp op, TensorStorage x, TensorStorage outv)
    {
        CpuKernels.UnaryFwdP(op, H(x), H(outv));
        Launches++;
    }

    public override void LaunchUnaryBwdP<TOp>(TOp op, TensorStorage x, TensorStorage y, TensorStorage gy, TensorStorage gx)
    {
        CpuKernels.UnaryBwdP(op, H(x), H(y), H(gy), H(gx));
        Launches++;
    }

    public override void LaunchAddInto(TensorStorage target, TensorStorage source)
    {
        CpuKernels.AddInto(H(target), H(source));
        Launches++;
    }

    public override void LaunchReduceSum(TensorStorage inp, TensorStorage outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        CpuKernels.ReduceSum(H(inp), H(outp), inDims, inStrides, outDims, reduceMask);
        Launches++;
    }

    public override void LaunchReduce<TR>(TensorStorage inp, TensorStorage outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        CpuKernels.Reduce<TR>(H(inp), H(outp), inDims, inStrides, outDims, reduceMask);
        Launches++;
    }

    public override void LaunchReduceArg(TensorStorage inp, TensorStorage outIdx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        CpuKernels.ReduceArg(H(inp), H(outIdx), inDims, inStrides, outDims, reduceMask, isMax);
        Launches++;
    }

    public override void LaunchMatMul(TensorStorage a, TensorStorage b, TensorStorage c,
        int M, int N, int K, int aMs, int aKs, int bKs, int bNs)
    {
        CpuKernels.MatMul2D(H(a), H(b), H(c), M, N, K, aMs, aKs, bKs, bNs);
        Launches++;
    }

    public override void LaunchReduceArgGrad(TensorStorage inp, TensorStorage gout, TensorStorage gx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        CpuKernels.ReduceArgGrad(H(inp), H(gout), H(gx), inDims, inStrides, outDims, reduceMask, isMax);
        Launches++;
    }

    public override void LaunchProdGrad(TensorStorage inp, TensorStorage gout, TensorStorage gx,
        int[] inDims, int[] inStrides, int[] outStrides, int[] reduceMask)
    {
        CpuKernels.ProdGrad(H(inp), H(gout), H(gx), inDims, inStrides, outStrides, reduceMask);
        Launches++;
    }

    public override void LaunchStridedCopy(TensorStorage inp, TensorStorage outp,
        int[] outDims, int[] inStrides, int baseOff = 0)
    {
        CpuKernels.StridedCopy(H(inp), H(outp), outDims, inStrides, baseOff);
        Launches++;
    }

    public override void LaunchScatterAxisRange(TensorStorage src, TensorStorage dst,
        int[] srcDims, int[] dstStrides, int axis, int axisOffset)
    {
        CpuKernels.ScatterAxisRange(H(src), H(dst), srcDims, dstStrides, axis, axisOffset);
        Launches++;
    }

    public override void LaunchMatMulBatched(TensorStorage a, TensorStorage b, TensorStorage c,
        int batchCount, int M, int N, int K, int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBatchStrides, int[] bBatchStrides)
    {
        CpuKernels.MatMulBatched(H(a), H(b), H(c), batchCount, M, N, K, aMs, aKs, bKs, bNs,
            batchDims, aBatchStrides, bBatchStrides);
        Launches++;
    }

    public override void LaunchGatherAxis(TensorStorage x, TensorStorage outp,
        int[] index, int[] outDims, int[] xStrides, int axis, int mode)
    {
        CpuKernels.GatherAxis(H(x), H(outp), index, outDims, xStrides, axis, mode);
        Launches++;
    }

    public override void LaunchScatterAddAxis(TensorStorage g, TensorStorage gx,
        int[] index, int[] srcDims, int[] dstStrides, int axis, int mode)
    {
        CpuKernels.ScatterAddAxis(H(g), H(gx), index, srcDims, dstStrides, axis, mode);
        Launches++;
    }

    public override void LaunchRepeat(TensorStorage x, TensorStorage outp,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        CpuKernels.Repeat(H(x), H(outp), outDims, inDims, inStrides);
        Launches++;
    }

    public override void LaunchRepeatGrad(TensorStorage g, TensorStorage gx,
        int[] outDims, int[] inDims, int[] inStrides)
    {
        CpuKernels.RepeatGrad(H(g), H(gx), outDims, inDims, inStrides);
        Launches++;
    }

    public override void LaunchIm2Col(TensorStorage x, TensorStorage col, int[] cfg)
    {
        CpuKernels.Im2Col(H(x), H(col), cfg);
        Launches++;
    }

    public override void LaunchCol2Im(TensorStorage gcol, TensorStorage gx, int[] cfg)
    {
        CpuKernels.Col2Im(H(gcol), H(gx), cfg);
        Launches++;
    }

    public override object? LaunchMaxPool2d(TensorStorage x, TensorStorage outp, int[] cfg, bool keepArgmax)
    {
        var argmax = CpuKernels.MaxPool2d(H(x), H(outp), cfg);
        Launches++;
        return keepArgmax ? argmax : null;
    }

    public override void LaunchMaxPool2dGrad(TensorStorage g, TensorStorage gx, object argmax)
    {
        CpuKernels.MaxPool2dGrad(H(g), H(gx), (int[])argmax);
        Launches++;
    }

    public override void LaunchAvgPool2d(TensorStorage x, TensorStorage outp, int[] cfg)
    {
        CpuKernels.AvgPool2d(H(x), H(outp), cfg);
        Launches++;
    }

    public override void LaunchAvgPool2dGrad(TensorStorage g, TensorStorage gx, int[] cfg)
    {
        CpuKernels.AvgPool2dGrad(H(g), H(gx), cfg);
        Launches++;
    }

    public override void LaunchAdam(TensorStorage p, TensorStorage g, TensorStorage m, TensorStorage v,
        float b1, float oneMinusB1, float b2, float oneMinusB2,
        float lr, float eps, float invBc1, float invBc2, float coupledWd, float decoupledFactor)
    {
        CpuKernels.AdamStep(H(p), H(g), H(m), H(v),
            b1, oneMinusB1, b2, oneMinusB2, lr, eps, invBc1, invBc2, coupledWd, decoupledFactor);
        Launches++;
    }

    public override void LaunchSgd(TensorStorage p, TensorStorage g, TensorStorage buf,
        float lr, float momentum, float weightDecay, float dampening, float nesterov, float hasBuf)
    {
        CpuKernels.SgdStep(H(p), H(g), H(buf), lr, momentum, weightDecay, dampening, nesterov, hasBuf);
        Launches++;
    }

    public override void Dispose() => _ctx?.Dispose();
}
