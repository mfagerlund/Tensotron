using System.Numerics;
using System.Threading.Tasks;

namespace Tensotron;

/// <summary>
/// Managed (CPU) twin of <see cref="Kernels"/>: the same strided index math, expressed as plain
/// loops over <c>float[]</c>, reusing the very same struct-generic <see cref="IBinaryOp"/> /
/// <see cref="IUnaryOp"/> / <see cref="IReduceOp"/> ops (so the arithmetic — including the XMath
/// transcendentals — is identical to the GPU path, which is what makes parity near-automatic).
///
/// Scalar baseline first (correctness + the parity gate). SIMD (<c>Vector&lt;float&gt;</c>) fast
/// paths layer on top, gated by the same fixtures. GPU atomics become plain <c>+=</c> here (and are
/// deterministic in this fixed loop order). The only multi-threaded op is matmul, and only when
/// opted in (<see cref="MatMulThreads"/> &gt; 1): it splits output *rows* across workers, which is
/// bit-identical to the serial result (each row computed by one worker, same K-reduction order), so
/// it does not relax Tensotron's parity law. Tensotron's single-thread rule still forbids calling
/// ops concurrently from *different* threads — this is parallelism strictly *within* one op.
/// </summary>
internal static class CpuKernels
{
    // ---- elementwise ----

    public static void Binary<TOp>(float[] a, float[] b, float[] outv,
        int[] outDims, int[] aStride, int[] bStride) where TOp : struct, IBinaryOp
    {
        int rank = outDims.Length;
        var op = default(TOp);
        for (int i = 0; i < outv.Length; i++)
        {
            int rem = i, aOff = 0, bOff = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int d = outDims[ax];
                int idx = rem % d;
                rem /= d;
                aOff += idx * aStride[ax];
                bOff += idx * bStride[ax];
            }
            outv[i] = op.Apply(a[aOff], b[bOff]);
        }
    }

    public static void UnaryFwd<TOp>(float[] x, float[] outv) where TOp : struct, IUnaryOp
    {
        var op = default(TOp);
        for (int i = 0; i < outv.Length; i++) outv[i] = op.Forward(x[i]);
    }

    public static void UnaryBwd<TOp>(float[] x, float[] y, float[] gy, float[] gx) where TOp : struct, IUnaryOp
    {
        var op = default(TOp);
        for (int i = 0; i < gx.Length; i++) gx[i] = op.Backward(x[i], y[i], gy[i]);
    }

    public static void UnaryFwdP<TOp>(TOp op, float[] x, float[] outv) where TOp : struct, IUnaryOp
    {
        for (int i = 0; i < outv.Length; i++) outv[i] = op.Forward(x[i]);
    }

    public static void UnaryBwdP<TOp>(TOp op, float[] x, float[] y, float[] gy, float[] gx) where TOp : struct, IUnaryOp
    {
        for (int i = 0; i < gx.Length; i++) gx[i] = op.Backward(x[i], y[i], gy[i]);
    }

    public static void AddInto(float[] target, float[] source)
    {
        int n = target.Length;
        int vw = Vector<float>.Count;
        int i = 0;
        for (int last = n - vw; i <= last; i += vw)
            (new Vector<float>(target, i) + new Vector<float>(source, i)).CopyTo(target, i);
        for (; i < n; i++) target[i] += source[i];
    }

    // ---- reductions ----

    public static void ReduceSum(float[] inp, float[] outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask)
    {
        int rank = inDims.Length;
        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++) if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        for (int oi = 0; oi < outp.Length; oi++)
        {
            int rem = oi, baseOff = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int od = outDims[ax];
                int idx = rem % od;
                rem /= od;
                if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
            }
            float acc = 0f;
            for (int r = 0; r < reducedCount; r++)
            {
                int rr = r, off = baseOff;
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
    }

    public static void Reduce<TR>(float[] inp, float[] outp,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask) where TR : struct, IReduceOp
    {
        int rank = inDims.Length;
        var op = default(TR);
        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++) if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        for (int oi = 0; oi < outp.Length; oi++)
        {
            int rem = oi, baseOff = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int od = outDims[ax];
                int idx = rem % od;
                rem /= od;
                if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
            }
            float acc = op.Identity;
            for (int r = 0; r < reducedCount; r++)
            {
                int rr = r, off = baseOff;
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
    }

    public static void ReduceArg(float[] inp, float[] outIdx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        int rank = inDims.Length;
        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++) if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        for (int oi = 0; oi < outIdx.Length; oi++)
        {
            int rem = oi, baseOff = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int od = outDims[ax];
                int idx = rem % od;
                rem /= od;
                if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
            }
            float best = isMax ? float.NegativeInfinity : float.PositiveInfinity;
            int bestIdx = 0;
            for (int r = 0; r < reducedCount; r++)
            {
                int rr = r, off = baseOff;
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
                bool better = isMax ? v > best : v < best;
                if (better) { best = v; bestIdx = r; }
            }
            outIdx[oi] = bestIdx;
        }
    }

    public static void ReduceArgGrad(float[] inp, float[] gout, float[] gx,
        int[] inDims, int[] inStrides, int[] outDims, int[] reduceMask, bool isMax)
    {
        int rank = inDims.Length;
        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++) if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        for (int oi = 0; oi < gout.Length; oi++)
        {
            int rem = oi, baseOff = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int od = outDims[ax];
                int idx = rem % od;
                rem /= od;
                if (reduceMask[ax] == 0) baseOff += idx * inStrides[ax];
            }
            float best = isMax ? float.NegativeInfinity : float.PositiveInfinity;
            int bestOff = baseOff;
            for (int r = 0; r < reducedCount; r++)
            {
                int rr = r, off = baseOff;
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
                bool better = isMax ? v > best : v < best;   // strict => first-wins on ties
                if (better) { best = v; bestOff = off; }
            }
            gx[bestOff] = gout[oi];
        }
    }

    public static void ProdGrad(float[] inp, float[] gout, float[] gx,
        int[] inDims, int[] inStrides, int[] outStrides, int[] reduceMask)
    {
        int rank = inDims.Length;
        int reducedCount = 1;
        for (int ax = 0; ax < rank; ax++) if (reduceMask[ax] == 1) reducedCount *= inDims[ax];

        for (int ii = 0; ii < inp.Length; ii++)
        {
            int rem = ii, baseOff = 0, oi = 0;
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
            float prod = 1f;
            for (int r = 0; r < reducedCount; r++)
            {
                int rr = r, off = baseOff;
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
                if (off != ii) prod *= inp[off];
            }
            gx[ii] = gout[oi] * prod;
        }
    }

    // ---- movement / structure ----

    public static void StridedCopy(float[] inp, float[] outp, int[] outDims, int[] inStrides, int baseOff)
    {
        int rank = outDims.Length;
        for (int i = 0; i < outp.Length; i++)
        {
            int rem = i, off = baseOff;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int d = outDims[ax];
                int idx = rem % d;
                rem /= d;
                off += idx * inStrides[ax];
            }
            outp[i] = inp[off];
        }
    }

    public static void ScatterAxisRange(float[] src, float[] dst,
        int[] srcDims, int[] dstStrides, int axis, int axisOffset)
    {
        int rank = srcDims.Length;
        for (int i = 0; i < src.Length; i++)
        {
            int rem = i, dstOff = 0;
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
    }

    public static void GatherAxis(float[] x, float[] outp, int[] idx,
        int[] outDims, int[] xStrides, int axis, int mode)
    {
        int rank = outDims.Length;
        for (int i = 0; i < outp.Length; i++)
        {
            int rem = i, off = 0;
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
    }

    public static void ScatterAddAxis(float[] g, float[] gx, int[] idx,
        int[] srcDims, int[] dstStrides, int axis, int mode)
    {
        int rank = srcDims.Length;
        for (int i = 0; i < g.Length; i++)
        {
            int rem = i, off = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int d = srcDims[ax];
                int coord = rem % d;
                rem /= d;
                int realCoord = ax == axis ? (mode == 0 ? idx[coord] : idx[i]) : coord;
                off += realCoord * dstStrides[ax];
            }
            gx[off] += g[i];
        }
    }

    public static void Repeat(float[] x, float[] outp, int[] outDims, int[] inDims, int[] inStrides)
    {
        int rank = outDims.Length;
        for (int i = 0; i < outp.Length; i++)
        {
            int rem = i, off = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int d = outDims[ax];
                int coord = rem % d;
                rem /= d;
                off += (coord % inDims[ax]) * inStrides[ax];
            }
            outp[i] = x[off];
        }
    }

    public static void RepeatGrad(float[] g, float[] gx, int[] outDims, int[] inDims, int[] inStrides)
    {
        int rank = outDims.Length;
        for (int i = 0; i < g.Length; i++)
        {
            int rem = i, off = 0;
            for (int ax = rank - 1; ax >= 0; ax--)
            {
                int d = outDims[ax];
                int coord = rem % d;
                rem /= d;
                off += (coord % inDims[ax]) * inStrides[ax];
            }
            gx[off] += g[i];
        }
    }

    // ---- matmul ----
    // C[m,n] = sum_k A[m*aMs + k*aKs] * B[n*bNs + k*bKs]; C contiguous (M,N). Same stride contract
    // as Kernels.MatMul2D, so backward transposes stay stride swaps. When both operands are
    // contiguous along the contraction axis (aKs==bKs==1 — the Linear-forward / MatMulNT layout),
    // the k-reduction is a SIMD dot product; otherwise fall back to the strided scalar loop
    // (covers the transposed backward matmuls, which contract along a non-unit stride).
    // ---- multi-threading knobs (within-op row parallelism only — see ParallelRows) ----
    // Off by default (1). The runtime sets these from TENSOTRON_CPU_THREADS / the CpuMatMulThreads
    // property. Row-parallel matmul is bit-identical to serial (each output row is computed by exactly
    // one worker, same K-reduction order), so it does NOT relax the parity contract.
    public static int MatMulThreads = 1;
    // Below this FLOP count (2*M*N*K) the thread-dispatch overhead outweighs the win — stay serial.
    public static long MatMulParallelMinFlops = 1_000_000;

    public static void MatMul2D(float[] a, float[] b, float[] c,
        int M, int N, int K, int aMs, int aKs, int bKs, int bNs)
    {
        int threads = MatMulThreads;
        if (threads > 1 && M >= 2 && 2L * M * N * K >= MatMulParallelMinFlops)
        {
            int chunks = Math.Min(threads, M);
            ParallelRows(M, chunks, threads,
                (s, e) => MatMul2DRange(a, b, c, N, K, aMs, aKs, bKs, bNs, s, e));
            return;
        }
        MatMul2DRange(a, b, c, N, K, aMs, aKs, bKs, bNs, 0, M);
    }

    // Compute output rows [mStart, mEnd). Each path's body for a row touches only that row's slice of
    // c (c[m*N .. m*N+N)), reads of a/b are disjoint or shared-read — so distinct row ranges never
    // race, and the result equals the serial [0,M) computation element-for-element.
    private static void MatMul2DRange(float[] a, float[] b, float[] c, int N, int K,
        int aMs, int aKs, int bKs, int bNs, int mStart, int mEnd)
    {
        if (aKs == 1 && bKs == 1) { MatMul2DDot(a, b, c, N, K, aMs, bNs, mStart, mEnd); return; }
        if (bNs == 1) { MatMul2DAxpy(a, b, c, N, K, aMs, aKs, bKs, mStart, mEnd); return; }
        for (int m = mStart; m < mEnd; m++)
        {
            int aBase = m * aMs;
            int cBase = m * N;
            for (int n = 0; n < N; n++)
            {
                int bBase = n * bNs;
                float acc = 0f;
                for (int k = 0; k < K; k++)
                    acc += a[aBase + k * aKs] * b[bBase + k * bKs];
                c[cBase + n] = acc;
            }
        }
    }

    // Split [0,M) into `chunks` contiguous row blocks and run them on the thread pool, capped at
    // `degree`. One Parallel.For iteration per block (not per row) keeps scheduling overhead flat and
    // each worker's rows contiguous (cache-friendly). Only reached for large GEMMs (the FLOP gate),
    // where this overhead is negligible against the work.
    private static void ParallelRows(int M, int chunks, int degree, Action<int, int> body)
    {
        int per = (M + chunks - 1) / chunks;
        var po = new ParallelOptions { MaxDegreeOfParallelism = degree };
        Parallel.For(0, chunks, po, ci =>
        {
            int s = ci * per;
            if (s >= M) return;
            int e = Math.Min(M, s + per);
            body(s, e);
        });
    }

    // Contraction-contiguous GEMM: each output is a vectorized dot product over K. Four independent
    // vector accumulators hide the FMA-latency dependency chain (the single-accumulator form is
    // latency-bound, not throughput-bound, for K in the tens-to-hundreds).
    private static void MatMul2DDot(float[] a, float[] b, float[] c, int N, int K, int aMs, int bNs,
        int mStart, int mEnd)
    {
        int vw = Vector<float>.Count;
        int step = vw * 4;
        for (int m = mStart; m < mEnd; m++)
        {
            int aBase = m * aMs;
            int cBase = m * N;
            for (int n = 0; n < N; n++)
            {
                int bBase = n * bNs;
                var a0 = Vector<float>.Zero; var a1 = Vector<float>.Zero;
                var a2 = Vector<float>.Zero; var a3 = Vector<float>.Zero;
                int k = 0;
                for (; k <= K - step; k += step)
                {
                    a0 += new Vector<float>(a, aBase + k) * new Vector<float>(b, bBase + k);
                    a1 += new Vector<float>(a, aBase + k + vw) * new Vector<float>(b, bBase + k + vw);
                    a2 += new Vector<float>(a, aBase + k + 2 * vw) * new Vector<float>(b, bBase + k + 2 * vw);
                    a3 += new Vector<float>(a, aBase + k + 3 * vw) * new Vector<float>(b, bBase + k + 3 * vw);
                }
                var vacc = (a0 + a1) + (a2 + a3);
                for (; k <= K - vw; k += vw)
                    vacc += new Vector<float>(a, aBase + k) * new Vector<float>(b, bBase + k);
                float acc = Vector.Dot(vacc, Vector<float>.One);   // horizontal sum
                for (; k < K; k++) acc += a[aBase + k] * b[bBase + k];
                c[cBase + n] = acc;
            }
        }
    }

    // Output-stationary (AXPY) GEMM for the bNs==1 layout (B contiguous along the output column —
    // the transposed backward matmuls). For each (m,k) we broadcast A[m,k] and FMA the contiguous
    // B[k,:] row into the contiguous C[m,:] row, so the SIMD runs over N. C is accumulated, so it is
    // cleared first (don't rely on the allocator zeroing).
    private static void MatMul2DAxpy(float[] a, float[] b, float[] c, int N, int K, int aMs, int aKs, int bKs,
        int mStart, int mEnd)
    {
        Array.Clear(c, mStart * N, (mEnd - mStart) * N);   // clear only this worker's row block
        int vw = Vector<float>.Count;
        int nVec = N - (N % vw);
        for (int m = mStart; m < mEnd; m++)
        {
            int aBase = m * aMs;
            int cBase = m * N;
            for (int k = 0; k < K; k++)
            {
                // No `av==0 → skip`: that would turn 0·NaN into 0 instead of NaN, diverging from
                // torch/the GPU path. Faithfulness over the small speedup on sparse (ReLU'd) grads.
                float av = a[aBase + k * aKs];
                var vav = new Vector<float>(av);
                int bBase = k * bKs;
                int n = 0;
                for (; n < nVec; n += vw)
                {
                    var vc = new Vector<float>(c, cBase + n)
                           + vav * new Vector<float>(b, bBase + n);
                    vc.CopyTo(c, cBase + n);
                }
                for (; n < N; n++) c[cBase + n] += av * b[bBase + n];
            }
        }
    }

    public static void MatMulBatched(float[] a, float[] b, float[] c,
        int batchCount, int M, int N, int K, int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBatchStrides, int[] bBatchStrides)
    {
        int batchRank = batchDims.Length;
        int mn = M * N;
        for (int idx = 0; idx < batchCount * mn; idx++)
        {
            int bb = idx / mn;
            int rem = idx % mn;
            int m = rem / N;
            int n = rem % N;

            int aOff = 0, bOff = 0, br = bb;
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
    }

    // ---- conv (im2col / col2im) ----
    // cfg layout: [0]N [1]C [2]H [3]W [4]kh [5]kw [6]sh [7]sw [8]ph [9]pw [10]dh [11]dw [12]Hout [13]Wout [14]K [15]L
    private static void DecodeColIndex(int i, int[] cfg,
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

    public static void Im2Col(float[] x, float[] col, int[] cfg)
    {
        int C = cfg[1], H = cfg[2], W = cfg[3];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        for (int i = 0; i < col.Length; i++)
        {
            DecodeColIndex(i, cfg, out int n, out int c, out int ki, out int kj, out int oh, out int ow);
            int ih = oh * sh - ph + ki * dh;
            int iw = ow * sw - pw + kj * dw;
            float v = 0f;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                v = x[((n * C + c) * H + ih) * W + iw];
            col[i] = v;
        }
    }

    public static void Col2Im(float[] gcol, float[] gx, int[] cfg)
    {
        int C = cfg[1], H = cfg[2], W = cfg[3];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        for (int i = 0; i < gcol.Length; i++)
        {
            DecodeColIndex(i, cfg, out int n, out int c, out int ki, out int kj, out int oh, out int ow);
            int ih = oh * sh - ph + ki * dh;
            int iw = ow * sw - pw + kj * dw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                gx[((n * C + c) * H + ih) * W + iw] += gcol[i];
        }
    }

    // ---- pooling ----
    // cfg layout: [0]N [1]C [2]H [3]W [4]kh [5]kw [6]sh [7]sw [8]ph [9]pw [10]dh [11]dw [12]Hout [13]Wout
    private static void DecodePoolIndex(int i, int[] cfg, out int n, out int c, out int oh, out int ow)
    {
        int C = cfg[1], Ho = cfg[12], Wo = cfg[13];
        ow = i % Wo;
        int t = i / Wo;
        oh = t % Ho;
        t /= Ho;
        c = t % C;
        n = t / C;
    }

    /// <summary>Max pool forward; returns the winning flat input offset per output (argmax) for backward.</summary>
    public static int[] MaxPool2d(float[] x, float[] outp, int[] cfg)
    {
        var argmax = new int[outp.Length];
        int C = cfg[1], H = cfg[2], W = cfg[3], kh = cfg[4], kw = cfg[5];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        for (int i = 0; i < outp.Length; i++)
        {
            DecodePoolIndex(i, cfg, out int n, out int c, out int oh, out int ow);
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
        return argmax;
    }

    public static void MaxPool2dGrad(float[] g, float[] gx, int[] argmax)
    {
        for (int i = 0; i < g.Length; i++)
        {
            int off = argmax[i];
            if (off >= 0) gx[off] += g[i];
        }
    }

    public static void AvgPool2d(float[] x, float[] outp, int[] cfg)
    {
        int C = cfg[1], H = cfg[2], W = cfg[3], kh = cfg[4], kw = cfg[5];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        for (int i = 0; i < outp.Length; i++)
        {
            DecodePoolIndex(i, cfg, out int n, out int c, out int oh, out int ow);
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
    }

    public static void AvgPool2dGrad(float[] g, float[] gx, int[] cfg)
    {
        int C = cfg[1], H = cfg[2], W = cfg[3], kh = cfg[4], kw = cfg[5];
        int sh = cfg[6], sw = cfg[7], ph = cfg[8], pw = cfg[9], dh = cfg[10], dw = cfg[11];
        for (int i = 0; i < g.Length; i++)
        {
            DecodePoolIndex(i, cfg, out int n, out int c, out int oh, out int ow);
            float share = g[i] / (kh * kw);
            for (int ki = 0; ki < kh; ki++)
            {
                int ih = oh * sh - ph + ki * dh;
                if (ih < 0 || ih >= H) continue;
                for (int kj = 0; kj < kw; kj++)
                {
                    int iw = ow * sw - pw + kj * dw;
                    if (iw < 0 || iw >= W) continue;
                    gx[((n * C + c) * H + ih) * W + iw] += share;
                }
            }
        }
    }

    // ---- optimizers (fused, in place) ----

    public static void AdamStep(float[] p, float[] g, float[] m, float[] v,
        float b1, float oneMinusB1, float b2, float oneMinusB2,
        float lr, float eps, float invBc1, float invBc2, float coupledWd, float decoupledFactor)
    {
        for (int i = 0; i < p.Length; i++)
        {
            float pi = p[i];
            float gi = g[i] + coupledWd * pi;
            float mi = b1 * m[i] + oneMinusB1 * gi;
            float vi = b2 * v[i] + oneMinusB2 * gi * gi;
            m[i] = mi;
            v[i] = vi;
            float step = (mi * invBc1) / (MathF.Sqrt(vi * invBc2) + eps);
            p[i] = pi * decoupledFactor - lr * step;
        }
    }

    public static void SgdStep(float[] p, float[] g, float[] buf,
        float lr, float momentum, float weightDecay, float dampening, float nesterov, float hasBuf)
    {
        for (int i = 0; i < p.Length; i++)
        {
            float gi = g[i] + weightDecay * p[i];
            if (momentum != 0f)
            {
                float bb = hasBuf != 0f ? momentum * buf[i] + (1f - dampening) * gi : gi;
                buf[i] = bb;
                gi = nesterov != 0f ? gi + momentum * bb : bb;
            }
            p[i] = p[i] - lr * gi;
        }
    }
}
