using ILGPU;

namespace Tensotron;

/// <summary>
/// Raw ILGPU kernels. Kept dumb and explicit; all autograd/broadcast choreography
/// lives in the host layer. These are the only device-specific code in the library.
/// </summary>
internal static class Kernels
{
    /// <summary>
    /// Arbitrary-rank strided elementwise binary op with broadcasting.
    /// Output is contiguous. Each input is gathered via broadcast strides
    /// (stride 0 on a broadcast axis), so equal-shape and broadcast cases share
    /// one kernel. Generic over the op struct so each op inlines.
    /// </summary>
    public static void BinaryEltwise<TOp>(
        Index1D i,
        int rank,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> outv,
        ArrayView<int> outDims,
        ArrayView<int> aStride,
        ArrayView<int> bStride)
        where TOp : struct, IBinaryOp
    {
        int rem = i;
        int aOff = 0;
        int bOff = 0;
        // Decompose the flat (contiguous) output index into a multi-index,
        // last axis fastest, accumulating input offsets as we go.
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = outDims[ax];
            int idx = rem % d;
            rem /= d;
            aOff += idx * aStride[ax];
            bOff += idx * bStride[ax];
        }

        outv[i] = default(TOp).Apply(a[aOff], b[bOff]);
    }

    /// <summary>Struct-generic unary forward: out[i] = op.Forward(x[i]).</summary>
    public static void UnaryFwd<TOp>(Index1D i, ArrayView<float> x, ArrayView<float> outv)
        where TOp : struct, IUnaryOp
    {
        outv[i] = default(TOp).Forward(x[i]);
    }

    /// <summary>Struct-generic unary backward: gx[i] = op.Backward(x[i], y[i], gy[i]).</summary>
    public static void UnaryBwd<TOp>(
        Index1D i, ArrayView<float> x, ArrayView<float> y, ArrayView<float> gy, ArrayView<float> gx)
        where TOp : struct, IUnaryOp
    {
        gx[i] = default(TOp).Backward(x[i], y[i], gy[i]);
    }

    /// <summary>As <see cref="UnaryFwd{TOp}"/>, but the op carries runtime parameters (e.g.
    /// LeakyRelu slope, ELU alpha) passed by value rather than via <c>default(TOp)</c>.</summary>
    public static void UnaryFwdP<TOp>(Index1D i, TOp op, ArrayView<float> x, ArrayView<float> outv)
        where TOp : struct, IUnaryOp
    {
        outv[i] = op.Forward(x[i]);
    }

    /// <summary>As <see cref="UnaryBwd{TOp}"/>, but the op carries runtime parameters.</summary>
    public static void UnaryBwdP<TOp>(
        Index1D i, TOp op, ArrayView<float> x, ArrayView<float> y, ArrayView<float> gy, ArrayView<float> gx)
        where TOp : struct, IUnaryOp
    {
        gx[i] = op.Backward(x[i], y[i], gy[i]);
    }

    /// <summary>In-place accumulation: target[i] += source[i] (equal shape).</summary>
    public static void AddInto(Index1D i, ArrayView<float> target, ArrayView<float> source)
    {
        target[i] += source[i];
    }

    /// <summary>
    /// General axis reduction (sum). One thread per OUTPUT element; it loops over
    /// the cartesian product of the reduced axes. Output dims have reduced axes = 1
    /// (keepdim). Correct for any axis set; not the fastest (no tree reduction) —
    /// fine for correctness-first and small backward reductions. TODO: tree-reduce.
    /// </summary>
    public static void ReduceSum(
        Index1D oi,
        int rank,
        ArrayView<float> inp,
        ArrayView<float> outp,
        ArrayView<int> inDims,
        ArrayView<int> inStrides,
        ArrayView<int> outDims,
        ArrayView<int> reduceMask) // 1 if axis is reduced
    {
        // Base input offset from the kept axes (decompose oi over outDims).
        int rem = oi;
        int baseOff = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int od = outDims[ax];
            int idx = rem % od;
            rem /= od;
            if (reduceMask[ax] == 0)
                baseOff += idx * inStrides[ax];
        }

        // Number of elements to reduce = product of reduced input dims.
        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++)
            if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        float acc = 0f;
        for (int r = 0; r < reducedCount; r++)
        {
            int rr = r;
            int off = baseOff;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                if (reduceMask[ax] == 1)
                {
                    int rd = inDims[ax];
                    int idx = rr % rd;
                    rr /= rd;
                    off += idx * inStrides[ax];
                }
            }
            acc += inp[off];
        }

        outp[oi] = acc;
    }

    /// <summary>Generic axis reduction (struct-generic over the reduce op). Mirrors ReduceSum.</summary>
    public static void Reduce<TR>(
        Index1D oi,
        int rank,
        ArrayView<float> inp,
        ArrayView<float> outp,
        ArrayView<int> inDims,
        ArrayView<int> inStrides,
        ArrayView<int> outDims,
        ArrayView<int> reduceMask)
        where TR : struct, IReduceOp
    {
        int rem = oi;
        int baseOff = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int od = outDims[ax];
            int idx = rem % od;
            rem /= od;
            if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
        }

        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++)
            if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        var op = default(TR);
        float acc = op.Identity;
        for (int r = 0; r < reducedCount; r++)
        {
            int rr = r;
            int off = baseOff;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                if (reduceMask[ax] == 1)
                {
                    int rd = inDims[ax];
                    int idx = rr % rd;
                    rr /= rd;
                    off += idx * inStrides[ax];
                }
            }
            acc = op.Combine(acc, inp[off]);
        }
        outp[oi] = acc;
    }

    /// <summary>
    /// Argmax/argmin over a single reduced axis: outputs the index within the reduced
    /// product (= index along that axis). isMax != 0 selects max, else min. First-wins
    /// on ties (matches torch).
    /// </summary>
    public static void ReduceArg(
        Index1D oi,
        int rank,
        ArrayView<float> inp,
        ArrayView<float> outIdx,
        ArrayView<int> inDims,
        ArrayView<int> inStrides,
        ArrayView<int> outDims,
        ArrayView<int> reduceMask,
        int isMax)
    {
        int rem = oi;
        int baseOff = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int od = outDims[ax];
            int idx = rem % od;
            rem /= od;
            if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
        }

        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++)
            if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        float best = isMax != 0 ? float.NegativeInfinity : float.PositiveInfinity;
        int bestIdx = 0;
        for (int r = 0; r < reducedCount; r++)
        {
            int rr = r;
            int off = baseOff;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                if (reduceMask[ax] == 1)
                {
                    int rd = inDims[ax];
                    int idx = rr % rd;
                    rr /= rd;
                    off += idx * inStrides[ax];
                }
            }
            float v = inp[off];
            bool better = isMax != 0 ? v > best : v < best;
            if (better) { best = v; bestIdx = r; }
        }
        outIdx[oi] = bestIdx;
    }

    /// <summary>
    /// Backward for max/min reduction: routes the upstream gradient to the SINGLE
    /// first-winning element of each reduced group (matches torch.max(dim).values,
    /// which routes through the selected index — not torch.amax, which splits ties).
    /// One thread per OUTPUT group; gx must be pre-zeroed. Input is contiguous, so the
    /// winning offset is a true flat index into gx.
    /// </summary>
    public static void ReduceArgGrad(
        Index1D oi,
        int rank,
        ArrayView<float> inp,
        ArrayView<float> gout,
        ArrayView<float> gx,
        ArrayView<int> inDims,
        ArrayView<int> inStrides,
        ArrayView<int> outDims,
        ArrayView<int> reduceMask,
        int isMax)
    {
        int rem = oi;
        int baseOff = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int od = outDims[ax];
            int idx = rem % od;
            rem /= od;
            if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
        }

        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++)
            if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        float best = isMax != 0 ? float.NegativeInfinity : float.PositiveInfinity;
        int bestOff = baseOff;
        for (int r = 0; r < reducedCount; r++)
        {
            int rr = r;
            int off = baseOff;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                if (reduceMask[ax] == 1)
                {
                    int rd = inDims[ax];
                    int idx = rr % rd;
                    rr /= rd;
                    off += idx * inStrides[ax];
                }
            }
            float v = inp[off];
            bool better = isMax != 0 ? v > best : v < best; // strict => first-wins on ties
            if (better) { best = v; bestOff = off; }
        }
        gx[bestOff] = gout[oi];
    }

    /// <summary>
    /// Backward for product reduction: gx[i] = gout[group(i)] * (product of all OTHER
    /// elements in i's reduced group). Computed by an explicit skip-self product, so it
    /// is exact even when the group contains zeros (where g*prod/x would give NaN/Inf).
    /// One thread per INPUT element. Input is contiguous (ii is its flat index).
    /// </summary>
    public static void ProdGrad(
        Index1D ii,
        int rank,
        ArrayView<float> inp,
        ArrayView<float> gout,
        ArrayView<float> gx,
        ArrayView<int> inDims,
        ArrayView<int> inStrides,
        ArrayView<int> outStrides,
        ArrayView<int> reduceMask)
    {
        // Decompose this input element's flat index; build the group's base offset
        // (reduced coords zeroed) and the output group index oi.
        int rem = ii;
        int baseOff = 0;
        int oi = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = inDims[ax];
            int idx = rem % d;
            rem /= d;
            if (reduceMask[ax] == 0)
            {
                baseOff += idx * inStrides[ax];
                oi += idx * outStrides[ax];
            }
        }

        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++)
            if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        float prod = 1f;
        for (int r = 0; r < reducedCount; r++)
        {
            int rr = r;
            int off = baseOff;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                if (reduceMask[ax] == 1)
                {
                    int rd = inDims[ax];
                    int idx = rr % rd;
                    rr /= rd;
                    off += idx * inStrides[ax];
                }
            }
            if (off != ii) prod *= inp[off]; // skip self => product of the others
        }
        gx[ii] = gout[oi] * prod;
    }

    /// <summary>
    /// Strided gather copy: out[i] (contiguous) = in[offset], where offset is built
    /// from the output multi-index via per-output-axis input strides. This is the
    /// workhorse for transpose/permute (and expand, via stride 0): the output is
    /// always materialized contiguous, so the rest of the library keeps assuming
    /// contiguous storage.
    /// </summary>
    public static void StridedCopy(
        Index1D i,
        int rank,
        ArrayView<float> inp,
        ArrayView<float> outp,
        ArrayView<int> outDims,
        ArrayView<int> inStrides,
        int baseOff)
    {
        int rem = i;
        int off = baseOff;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = outDims[ax];
            int idx = rem % d;
            rem /= d;
            off += idx * inStrides[ax];
        }
        outp[i] = inp[off];
    }

    /// <summary>
    /// Scatter a contiguous source block into a region of a contiguous destination,
    /// offset by <paramref name="axisOffset"/> along <paramref name="axis"/>. One
    /// thread per source element. This is the dual of a base-offset StridedCopy gather
    /// and powers cat (forward) / narrow (backward).
    /// </summary>
    public static void ScatterAxisRange(
        Index1D i,
        int rank,
        ArrayView<float> src,
        ArrayView<float> dst,
        ArrayView<int> srcDims,
        ArrayView<int> dstStrides,
        int axis,
        int axisOffset)
    {
        int rem = i;
        int dstOff = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = srcDims[ax];
            int idx = rem % d;
            rem /= d;
            int dstIdx = ax == axis ? idx + axisOffset : idx;
            dstOff += dstIdx * dstStrides[ax];
        }
        dst[dstOff] = src[i];
    }

    /// <summary>
    /// Broadcast-batched matmul. One thread per output element of a (batch..., M, N)
    /// result. The batch index is decomposed over <paramref name="batchDims"/> and
    /// mapped to each operand via its batch strides (stride 0 = broadcast that batch
    /// axis), so (B,M,K)@(B,K,N), (B,M,K)@(K,N), and multi-axis broadcasts all share
    /// this kernel. Matrix access uses explicit strides, so backward transposes are
    /// stride swaps. cuBLAS strided-batched GEMM replaces this on CUDA later.
    /// Computes c[bb,m,n] = sum_k a[aOff + m*aMs + k*aKs] * b[bOff + n*bNs + k*bKs].
    /// </summary>
    public static void MatMulBatched(
        Index1D idx,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int batchRank,
        int M, int N, int K,
        int aMs, int aKs,
        int bKs, int bNs,
        ArrayView<int> batchDims,
        ArrayView<int> aBatchStrides,
        ArrayView<int> bBatchStrides)
    {
        int mn = M * N;
        int bb = idx / mn;
        int rem = idx % mn;
        int m = rem / N;
        int n = rem % N;

        int aOff = 0;
        int bOff = 0;
        int br = bb;
        for (int ax = batchRank - 1; ax >= 0; ax--)
        {
            int d = batchDims[ax];
            int bi = br % d;
            br /= d;
            aOff += bi * aBatchStrides[ax];
            bOff += bi * bBatchStrides[ax];
        }

        float acc = 0f;
        int aBase = aOff + m * aMs;
        int bBase = bOff + n * bNs;
        for (int k = 0; k < K; k++)
            acc += a[aBase + k * aKs] * b[bBase + k * bKs];

        c[idx] = acc;
    }

    /// <summary>
    /// Gather forward along one axis. Output is contiguous; one thread per output
    /// element. The index value used for <paramref name="axis"/> is:
    ///   mode 0 (index_select): idx[axisCoord]  — idx is 1D, length outDims[axis];
    ///   mode 1 (gather):       idx[i]           — idx has the full output shape.
    /// All other axes pass their coordinate straight through.
    /// </summary>
    public static void GatherAxis(
        Index1D i,
        int rank,
        ArrayView<float> x,
        ArrayView<float> outp,
        ArrayView<int> idx,
        ArrayView<int> outDims,
        ArrayView<int> xStrides,
        int axis,
        int mode)
    {
        int rem = i;
        int off = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = outDims[ax];
            int coord = rem % d;
            rem /= d;
            int realCoord = ax == axis ? (mode == 0 ? idx[coord] : idx[i]) : coord;
            off += realCoord * xStrides[ax];
        }
        outp[i] = x[off];
    }

    /// <summary>
    /// Scatter-add dual of <see cref="GatherAxis"/>: gx[off] += g[i], where off is built
    /// the same way (so it routes index_select/gather backward, and powers scatter_add
    /// forward). Atomic because many sources can map to one destination. gx pre-zeroed
    /// (or pre-seeded with self for scatter_add forward).
    /// </summary>
    public static void ScatterAddAxis(
        Index1D i,
        int rank,
        ArrayView<float> g,
        ArrayView<float> gx,
        ArrayView<int> idx,
        ArrayView<int> srcDims,
        ArrayView<int> dstStrides,
        int axis,
        int mode)
    {
        int rem = i;
        int off = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = srcDims[ax];
            int coord = rem % d;
            rem /= d;
            int realCoord = ax == axis ? (mode == 0 ? idx[coord] : idx[i]) : coord;
            off += realCoord * dstStrides[ax];
        }
        Atomic.Add(ref gx[off], g[i]);
    }

    /// <summary>
    /// Tiling forward (torch.Tensor.repeat): out[i] = x[off], where each output axis
    /// coordinate is taken modulo the input dim. New leading axes have inDim 1 / stride 0.
    /// </summary>
    public static void Repeat(
        Index1D i,
        int rank,
        ArrayView<float> x,
        ArrayView<float> outp,
        ArrayView<int> outDims,
        ArrayView<int> inDims,
        ArrayView<int> inStrides)
    {
        int rem = i;
        int off = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = outDims[ax];
            int coord = rem % d;
            rem /= d;
            off += (coord % inDims[ax]) * inStrides[ax];
        }
        outp[i] = x[off];
    }

    /// <summary>Backward dual of <see cref="Repeat"/>: every tile that read x[off]
    /// adds its gradient back (atomic; gx pre-zeroed). One thread per OUTPUT element.</summary>
    public static void RepeatGrad(
        Index1D i,
        int rank,
        ArrayView<float> g,
        ArrayView<float> gx,
        ArrayView<int> outDims,
        ArrayView<int> inDims,
        ArrayView<int> inStrides)
    {
        int rem = i;
        int off = 0;
        for (int ax = rank - 1; ax >= 0; ax--)
        {
            int d = outDims[ax];
            int coord = rem % d;
            rem /= d;
            off += (coord % inDims[ax]) * inStrides[ax];
        }
        Atomic.Add(ref gx[off], g[i]);
    }

    // im2col / col2im config layout (single int buffer, avoids a long param list):
    //   [0]N [1]C [2]H [3]W [4]kh [5]kw [6]sh [7]sw [8]ph [9]pw [10]dh [11]dw [12]Hout [13]Wout [14]K [15]L
    private static void DecodeColIndex(int i, ArrayView<int> cfg,
        out int n, out int c, out int ki, out int kj, out int oh, out int ow)
    {
        int kw = cfg[5], kh = cfg[4], K = cfg[14], L = cfg[15], Wo = cfg[13];
        int l = i % L;
        int tmp = i / L;
        int k = tmp % K;
        n = tmp / K;
        ow = l % Wo;
        oh = l / Wo;
        kj = k % kw;
        int t2 = k / kw;
        ki = t2 % kh;
        c = t2 / kh;
    }

    /// <summary>
    /// im2col: gather each conv receptive field into a column. Output is contiguous
    /// (N, C·kh·kw, Hout·Wout). Out-of-bounds (padding) positions read 0. One thread per
    /// output element. Conv forward is then a batched matmul of the weight against this.
    /// </summary>
    public static void Im2Col(Index1D i, ArrayView<float> x, ArrayView<float> col, ArrayView<int> cfg)
    {
        DecodeColIndex(i, cfg, out int n, out int c, out int ki, out int kj, out int oh, out int ow);
        int C = cfg[1], H = cfg[2], W = cfg[3];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        int ih = oh * sh - ph + ki * dh;
        int iw = ow * sw - pw + kj * dw;
        float v = 0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            v = x[((n * C + c) * H + ih) * W + iw];
        col[i] = v;
    }

    /// <summary>Backward dual of <see cref="Im2Col"/>: scatter-add each column gradient
    /// back to its input pixel (atomic; overlapping receptive fields accumulate). gx pre-zeroed.</summary>
    public static void Col2Im(Index1D i, ArrayView<float> gcol, ArrayView<float> gx, ArrayView<int> cfg)
    {
        DecodeColIndex(i, cfg, out int n, out int c, out int ki, out int kj, out int oh, out int ow);
        int C = cfg[1], H = cfg[2], W = cfg[3];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        int ih = oh * sh - ph + ki * dh;
        int iw = ow * sw - pw + kj * dw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            Atomic.Add(ref gx[((n * C + c) * H + ih) * W + iw], gcol[i]);
    }

    /// <summary>
    /// 2D matmul core via explicit strides, so transposes are expressed as stride
    /// swaps (no transpose copies in backward). Output C is contiguous (M,N).
    /// Computes C[m,n] = sum_k A[m,k] * B[k,n] using:
    ///   a index = m*aMs + k*aKs ,  b index = n*bNs + k*bKs.
    /// cuBLAS GEMM replaces this on CUDA later; same interface.
    /// </summary>
    public static void MatMul2D(
        Index1D idx,
        ArrayView<float> a,
        ArrayView<float> b,
        ArrayView<float> c,
        int M, int N, int K,
        int aMs, int aKs,
        int bKs, int bNs)
    {
        int total = M * N;
        if (idx >= total) return;
        int m = idx / N;
        int n = idx % N;

        float acc = 0f;
        int aBase = m * aMs;
        int bBase = n * bNs;
        for (int k = 0; k < K; k++)
            acc += a[aBase + k * aKs] * b[bBase + k * bKs];

        c[m * N + n] = acc;
    }

    // pool config layout (single int buffer):
    //   [0]N [1]C [2]H [3]W [4]kh [5]kw [6]sh [7]sw [8]ph [9]pw [10]dh [11]dw [12]Hout [13]Wout
    private static void DecodePoolIndex(int i, ArrayView<int> cfg,
        out int n, out int c, out int oh, out int ow)
    {
        int C = cfg[1], Ho = cfg[12], Wo = cfg[13];
        ow = i % Wo;
        int t = i / Wo;
        oh = t % Ho;
        t /= Ho;
        c = t % C;
        n = t / C;
    }

    /// <summary>
    /// 2D max pooling forward (torch.nn.functional.max_pool2d). One thread per output
    /// element. Padded cells are treated as -inf (never selected). Records the flat input
    /// offset of the winning cell into <paramref name="argmax"/> for the backward pass.
    /// </summary>
    public static void MaxPool2d(Index1D i, ArrayView<float> x, ArrayView<float> outp,
        ArrayView<int> argmax, ArrayView<int> cfg)
    {
        DecodePoolIndex(i, cfg, out int n, out int c, out int oh, out int ow);
        int C = cfg[1], H = cfg[2], W = cfg[3], kh = cfg[4], kw = cfg[5];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        float best = float.NegativeInfinity;
        int bestOff = -1;
        for (int ki = 0; ki < kh; ki++)
        {
            int ih = oh * sh - ph + ki * dh;
            if (ih < 0 || ih >= H) continue;
            for (int kj = 0; kj < kw; kj++)
            {
                int iw = ow * sw - pw + kj * dw;
                if (iw < 0 || iw >= W) continue;
                int off = ((n * C + c) * H + ih) * W + iw;
                float v = x[off];
                if (v > best) { best = v; bestOff = off; }
            }
        }
        outp[i] = bestOff >= 0 ? best : 0f;
        argmax[i] = bestOff;
    }

    /// <summary>Backward dual of <see cref="MaxPool2d"/>: route each output gradient to its
    /// argmax input cell (atomic; overlapping windows accumulate). gx pre-zeroed.</summary>
    public static void MaxPool2dGrad(Index1D i, ArrayView<float> g, ArrayView<float> gx,
        ArrayView<int> argmax)
    {
        int off = argmax[i];
        if (off >= 0) Atomic.Add(ref gx[off], g[i]);
    }

    /// <summary>
    /// 2D average pooling forward (torch.nn.functional.avg_pool2d, count_include_pad=True):
    /// each output is the window sum divided by the full window size kh·kw (padded cells
    /// count as zero). One thread per output element.
    /// </summary>
    public static void AvgPool2d(Index1D i, ArrayView<float> x, ArrayView<float> outp, ArrayView<int> cfg)
    {
        DecodePoolIndex(i, cfg, out int n, out int c, out int oh, out int ow);
        int C = cfg[1], H = cfg[2], W = cfg[3], kh = cfg[4], kw = cfg[5];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        float sum = 0f;
        for (int ki = 0; ki < kh; ki++)
        {
            int ih = oh * sh - ph + ki * dh;
            if (ih < 0 || ih >= H) continue;
            for (int kj = 0; kj < kw; kj++)
            {
                int iw = ow * sw - pw + kj * dw;
                if (iw < 0 || iw >= W) continue;
                sum += x[((n * C + c) * H + ih) * W + iw];
            }
        }
        outp[i] = sum / (kh * kw);
    }

    /// <summary>Backward dual of <see cref="AvgPool2d"/>: spread each output gradient evenly
    /// (÷ kh·kw) across its window's in-bounds cells (atomic). gx pre-zeroed.</summary>
    public static void AvgPool2dGrad(Index1D i, ArrayView<float> g, ArrayView<float> gx, ArrayView<int> cfg)
    {
        DecodePoolIndex(i, cfg, out int n, out int c, out int oh, out int ow);
        int C = cfg[1], H = cfg[2], W = cfg[3], kh = cfg[4], kw = cfg[5];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        float share = g[i] / (kh * kw);
        for (int ki = 0; ki < kh; ki++)
        {
            int ih = oh * sh - ph + ki * dh;
            if (ih < 0 || ih >= H) continue;
            for (int kj = 0; kj < kw; kj++)
            {
                int iw = ow * sw - pw + kj * dw;
                if (iw < 0 || iw >= W) continue;
                Atomic.Add(ref gx[((n * C + c) * H + ih) * W + iw], share);
            }
        }
    }
}
