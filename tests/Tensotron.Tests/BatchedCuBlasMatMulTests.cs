using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Exercises the per-batch cuBLAS SGEMM path in <c>LaunchMatMulBatched</c> — taken when a batched
/// product's per-matrix work (M·N·K) is large. Conv rides this path (im2col + batched matmul), so a
/// silent stride/offset error here would corrupt conv gradients. Checks forward + dA + dB against a
/// CPU reference for (1) a fully-batched bmm (both operands batched, transposed backward operands)
/// and (2) the broadcast 2D@3D case conv actually uses (weight shared over the batch → dA reduces
/// over it). Dims give per-matrix work 64·64·80 = 327680 ≥ 2^18 so the cuBLAS path is taken on CUDA;
/// on a CPU-only box _blas is null and the same assertions validate the naive fallback.
/// </summary>
public class BatchedCuBlasMatMulTests
{
    private const int Bt = 3, M = 64, N = 64, K = 80;

    private static float[] Mm(float[] a, float[] b, int m, int k, int n)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                for (int t = 0; t < k; t++) acc += a[i * k + t] * b[t * n + j];
                c[i * n + j] = acc;
            }
        return c;
    }

    private static float[] T(float[] a, int r, int c)
    {
        var t = new float[r * c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) t[j * r + i] = a[i * c + j];
        return t;
    }

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        return a;
    }

    private static void Close(string tag, float[] got, float[] exp, float tol)
    {
        Assert.Equal(exp.Length, got.Length);
        for (int i = 0; i < exp.Length; i++)
            Assert.True(MathF.Abs(got[i] - exp[i]) <= tol, $"{tag}[{i}] {got[i]} != {exp[i]}");
    }

    [Fact]
    public void BatchedBmm_ForwardAndGrads_MatchCpu()
    {
        var rng = new Random(7);
        var aData = Rand(Bt * M * K, rng);
        var bData = Rand(Bt * K * N, rng);
        var gData = Rand(Bt * M * N, rng);

        var A = Tensor.FromShaped(aData, new[] { Bt, M, K }).RequireGrad();
        var B = Tensor.FromShaped(bData, new[] { Bt, K, N }).RequireGrad();
        var C = TensorOps.Bmm(A, B);
        C.Backward(Tensor.FromShaped(gData, new[] { Bt, M, N }));

        var cGot = C.ToArray();
        var dAGot = A.Grad!.ToArray();
        var dBGot = B.Grad!.ToArray();
        for (int e = 0; e < Bt; e++)
        {
            var ae = aData[(e * M * K)..((e + 1) * M * K)];
            var be = bData[(e * K * N)..((e + 1) * K * N)];
            var ge = gData[(e * M * N)..((e + 1) * M * N)];
            Close($"C{e}", cGot[(e * M * N)..((e + 1) * M * N)], Mm(ae, be, M, K, N), 4e-3f);
            Close($"dA{e}", dAGot[(e * M * K)..((e + 1) * M * K)], Mm(ge, T(be, K, N), M, N, K), 4e-3f);
            Close($"dB{e}", dBGot[(e * K * N)..((e + 1) * K * N)], Mm(T(ae, M, K), ge, K, M, N), 4e-3f);
        }
    }

    [Fact]
    public void BroadcastMatMul_2Dx3D_ForwardAndGrads_MatchCpu()
    {
        // (M,K) @ (Bt,K,N): the conv pattern — weight A broadcast over the batch (batch stride 0),
        // so dA reduces over the batch. Validates the broadcast offset + the grad reduction.
        var rng = new Random(11);
        var aData = Rand(M * K, rng);
        var bData = Rand(Bt * K * N, rng);
        var gData = Rand(Bt * M * N, rng);

        var A = Tensor.FromShaped(aData, new[] { M, K }).RequireGrad();
        var B = Tensor.FromShaped(bData, new[] { Bt, K, N }).RequireGrad();
        var C = TensorOps.MatMul(A, B);                 // -> (Bt, M, N)
        C.Backward(Tensor.FromShaped(gData, new[] { Bt, M, N }));

        var cGot = C.ToArray();
        var dBGot = B.Grad!.ToArray();
        var dARef = new float[M * K];
        for (int e = 0; e < Bt; e++)
        {
            var be = bData[(e * K * N)..((e + 1) * K * N)];
            var ge = gData[(e * M * N)..((e + 1) * M * N)];
            Close($"C{e}", cGot[(e * M * N)..((e + 1) * M * N)], Mm(aData, be, M, K, N), 4e-3f);
            Close($"dB{e}", dBGot[(e * K * N)..((e + 1) * K * N)], Mm(T(aData, M, K), ge, K, M, N), 4e-3f);
            var dAe = Mm(ge, T(be, K, N), M, N, K);
            for (int i = 0; i < dARef.Length; i++) dARef[i] += dAe[i];
        }
        Close("dA", A.Grad!.ToArray(), dARef, 6e-3f);   // summed over Bt → looser tol
    }
}
