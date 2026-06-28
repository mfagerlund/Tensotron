namespace Tensotron;

public static partial class TensorOps
{
    // ---------------- batched / N-D matmul ----------------
    //
    // PyTorch matmul on rank>2 treats the last two axes as the matrix and broadcasts
    // all leading (batch) axes. Tensors are contiguous, so a batch axis's element
    // stride is just its row-major stride; broadcast axes get stride 0.

    private static (int[] dims, int[] aBs, int[] bBs) BroadcastBatch(
        int[] aDims, int[] aStr, int[] bDims, int[] bStr)
    {
        int ra = aDims.Length, rb = bDims.Length, r = Math.Max(ra, rb);
        var dims = new int[r];
        var aBs = new int[r];
        var bBs = new int[r];
        for (int j = 0; j < r; j++)
        {
            int axa = j - (r - ra), axb = j - (r - rb);
            int da = axa >= 0 ? aDims[axa] : 1;
            int db = axb >= 0 ? bDims[axb] : 1;
            if (da != db && da != 1 && db != 1)
                throw new InvalidOperationException($"Cannot broadcast batch dims [{string.Join(",", aDims)}] with [{string.Join(",", bDims)}].");
            int od = Math.Max(da, db);
            dims[j] = od;
            aBs[j] = (axa >= 0 && da == od) ? aStr[axa] : 0;
            bBs[j] = (axb >= 0 && db == od) ? bStr[axb] : 0;
        }
        return (dims, aBs, bBs);
    }

    // No-grad batched matmul: result is contiguous (batchDims..., M, N).
    private static Tensor RawBmm(
        Tensor a, Tensor b, int M, int N, int K,
        int aMs, int aKs, int bKs, int bNs,
        int[] batchDims, int[] aBs, int[] bBs)
    {
        int batchCount = 1;
        foreach (var d in batchDims) batchCount *= d;

        var outBuf = Runtime.Allocate(batchCount * M * N);
        Runtime.LaunchMatMulBatched(a.Buffer, b.Buffer, outBuf, batchCount, M, N, K,
            aMs, aKs, bKs, bNs, batchDims, aBs, bBs);

        var outDims = new int[batchDims.Length + 2];
        Array.Copy(batchDims, outDims, batchDims.Length);
        outDims[^2] = M;
        outDims[^1] = N;
        return new Tensor(new Shape(outDims), outBuf);
    }

    private static Tensor MatMulND(Tensor a, Tensor b)
    {
        int ar = a.Rank, br = b.Rank;
        int M = a.Shape.Dims[ar - 2], K = a.Shape.Dims[ar - 1];
        int K2 = b.Shape.Dims[br - 2], N = b.Shape.Dims[br - 1];
        if (K != K2)
            throw new InvalidOperationException($"MatMul inner dims differ: {a.Shape} @ {b.Shape}.");

        var aBatchDims = a.Shape.Dims[..(ar - 2)];
        var aBatchStr = a.Shape.Strides[..(ar - 2)];
        var bBatchDims = b.Shape.Dims[..(br - 2)];
        var bBatchStr = b.Shape.Strides[..(br - 2)];
        var (batchDims, aBs, bBs) = BroadcastBatch(aBatchDims, aBatchStr, bBatchDims, bBatchStr);

        // forward: a (.,M,K) contiguous -> aMs=K,aKs=1 ; b (.,K,N) -> bKs=N,bNs=1
        var result = RawBmm(a, b, M, N, K, K, 1, N, 1, batchDims, aBs, bBs);

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(a) || Tensor.NeedsGrad(b)))
        {
            result.Node = new GradNode("MatMulND", new[] { a, b }, g =>
            {
                // g is contiguous, shape batchDims...,M,N
                var gBatchDims = g.Shape.Dims[..(g.Rank - 2)];
                var gBatchStr = g.Shape.Strides[..(g.Rank - 2)];
                int gMs = g.Shape.Strides[g.Rank - 2]; // = N
                int gNs = g.Shape.Strides[g.Rank - 1]; // = 1

                if (Tensor.NeedsGrad(a))
                {
                    // dA[.,m,k] = sum_n g[.,m,n] * B[.,k,n]   (Mo=M, No=K, inner=N)
                    //   a=g:  aMs=gMs, aKs=gNs
                    //   b=B:  out-n is k -> bNs=B k-stride ; inner is n -> bKs=B n-stride
                    var (bd, gBs2, bBs2) = BroadcastBatch(gBatchDims, gBatchStr, bBatchDims, bBatchStr);
                    var dA = RawBmm(g, b, M, K, N,
                        gMs, gNs,
                        b.Shape.Strides[br - 1], b.Shape.Strides[br - 2],
                        bd, gBs2, bBs2);
                    a.AddGrad(ReduceGradToShape(dA, a.Shape));
                }
                if (Tensor.NeedsGrad(b))
                {
                    // dB[.,k,n] = sum_m A[.,m,k] * g[.,m,n]   (Mo=K, No=N, inner=M)
                    //   a=A:  out-m is k -> aMs=A k-stride ; inner is m -> aKs=A m-stride
                    //   b=g:  inner is m -> bKs=gMs ; out-n is n -> bNs=gNs
                    var (bd, aBs2, gBs2) = BroadcastBatch(aBatchDims, aBatchStr, gBatchDims, gBatchStr);
                    var dB = RawBmm(a, g, K, N, M,
                        a.Shape.Strides[ar - 1], a.Shape.Strides[ar - 2],
                        gMs, gNs,
                        bd, aBs2, gBs2);
                    b.AddGrad(ReduceGradToShape(dB, b.Shape));
                }
            });
        }

        return result;
    }

    /// <summary>Batched matmul, strict 3D with equal batch (PyTorch <c>torch.bmm</c>).</summary>
    public static Tensor Bmm(Tensor a, Tensor b)
    {
        if (a.Rank != 3 || b.Rank != 3)
            throw new InvalidOperationException($"bmm expects 3D tensors, got {a.Shape} and {b.Shape}.");
        if (a.Shape.Dims[0] != b.Shape.Dims[0])
            throw new InvalidOperationException($"bmm batch sizes differ: {a.Shape.Dims[0]} vs {b.Shape.Dims[0]}.");
        return MatMulND(a, b);
    }

    /// <summary>Matrix-vector product (PyTorch <c>torch.mv</c>).</summary>
    public static Tensor Mv(Tensor mat, Tensor vec) => MatMul(mat, vec);

    /// <summary>Vector dot product (PyTorch <c>torch.dot</c>).</summary>
    public static Tensor Dot(Tensor a, Tensor b) => MatMul(a, b);

    /// <summary>Outer product of two 1D vectors → (M,N).</summary>
    public static Tensor Outer(Tensor a, Tensor b)
    {
        if (a.Rank != 1 || b.Rank != 1)
            throw new InvalidOperationException($"outer expects 1D tensors, got {a.Shape} and {b.Shape}.");
        return MatMul(a.Reshape(a.Shape.Dims[0], 1), b.Reshape(1, b.Shape.Dims[0]));
    }
}

public sealed partial class Tensor
{
    public Tensor Bmm(Tensor other) => TensorOps.Bmm(this, other);
}
