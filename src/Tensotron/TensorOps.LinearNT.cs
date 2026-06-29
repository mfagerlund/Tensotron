namespace Tensotron;

public static partial class TensorOps
{
    /// <summary>
    /// y = a · bᵀ computed directly from the explicit-stride matmul, WITHOUT materializing bᵀ.
    /// This is the matmul `torch.nn.functional.linear` needs (a:(M,K) inputs, b:(N,K) weight →
    /// y:(M,N)). Both operands are read through strides, so no transpose copy is ever
    /// materialized. Backward mirrors MatMul2D's stride-only transposes:
    ///   da = g · b      (M,K)
    ///   db = gᵀ · a     (N,K)
    /// </summary>
    public static Tensor MatMulNT(Tensor a, Tensor b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new InvalidOperationException($"MatMulNT expects 2D operands, got {a.Shape} and {b.Shape}.");
        int M = a.Shape.Dims[0], K = a.Shape.Dims[1];
        int N = b.Shape.Dims[0], K2 = b.Shape.Dims[1];
        if (K != K2)
            throw new InvalidOperationException($"MatMulNT inner dims differ: {a.Shape} · {b.Shape}ᵀ.");

        var outBuf = Runtime.Allocate(M * N);
        // y[m,n] = Σ_k a[m,k]·b[n,k]:  a row-major (aMs=K,aKs=1); b read as [n,k] (bNs=K,bKs=1).
        Runtime.LaunchMatMul(a.Buffer, b.Buffer, outBuf, M, N, K, K, 1, 1, K);
        var result = new Tensor(new Shape(M, N), outBuf);

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(a) || Tensor.NeedsGrad(b)))
        {
            result.Node = new GradNode("MatMulNT", new[] { a, b }, g =>
            {
                if (Tensor.NeedsGrad(a))
                {
                    // da (M,K) = g (M,N) · b (N,K).  c[m,k]=Σ_n g[m,n]·b[n,k]
                    //   a=g: aMs=N,aKs=1 ; b=b read [k,n]→b[n,k]: bNs=1,bKs=K
                    var dABuf = Runtime.Allocate(M * K);
                    Runtime.LaunchMatMul(g.Buffer, b.Buffer, dABuf, M, K, N, N, 1, K, 1);
                    a.AddGrad(new Tensor(new Shape(M, K), dABuf));
                }
                if (Tensor.NeedsGrad(b))
                {
                    // db (N,K) = gᵀ (N,M) · a (M,K).  c[n,k]=Σ_m g[m,n]·a[m,k]
                    //   a=g read [n,m]→g[m,n]: aMs=1,aKs=N ; b=a: bNs=1,bKs=K
                    var dBBuf = Runtime.Allocate(N * K);
                    Runtime.LaunchMatMul(g.Buffer, a.Buffer, dBBuf, N, K, M, 1, N, K, 1);
                    b.AddGrad(new Tensor(new Shape(N, K), dBBuf));
                }
            });
        }

        return result;
    }
}
