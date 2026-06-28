using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Verifies the no-transpose-copy matmul <see cref="TensorOps.MatMulNT"/> (y = a·bᵀ) that
/// backs <c>Linear</c>: forward and both gradients must equal the reference <c>a · bᵀ</c>
/// computed the old way, at a small size (naive kernel) and a large size (tiled kernel).
/// </summary>
public class MatMulNtOpTests
{
    [Theory]
    [InlineData(5, 7, 3)]       // small -> naive matmul path
    [InlineData(96, 80, 112)]   // large, non-multiple-of-16 -> tiled matmul path
    public void MatMulNT_ForwardAndGrads_MatchReferenceTranspose(int M, int K, int N)
    {
        var rng = new Random(7);
        float[] Rand(int n) { var d = new float[n]; for (int i = 0; i < n; i++) d[i] = (float)(rng.NextDouble() - 0.5); return d; }
        var aData = Rand(M * K);
        var bData = Rand(N * K);   // weight is (N=out, K=in)
        var wData = Rand(M * N);   // upstream grad

        // Reference path: y = a @ bᵀ via the existing MatMul(a, b.T()).
        var aRef = Tensor.FromArray((float[])aData.Clone(), M, K).RequireGrad();
        var bRef = Tensor.FromArray((float[])bData.Clone(), N, K).RequireGrad();
        var wRef = Tensor.FromArray((float[])wData.Clone(), M, N);
        var yRef = TensorOps.MatMul(aRef, bRef.T());
        TensorOps.Sum(TensorOps.Mul(yRef, wRef)).Backward();

        // Under test: y = MatMulNT(a, b).
        var a = Tensor.FromArray((float[])aData.Clone(), M, K).RequireGrad();
        var b = Tensor.FromArray((float[])bData.Clone(), N, K).RequireGrad();
        var w = Tensor.FromArray((float[])wData.Clone(), M, N);
        var y = TensorOps.MatMulNT(a, b);
        TensorOps.Sum(TensorOps.Mul(y, w)).Backward();

        void Same(string what, float[] got, float[] exp)
        {
            Assert.Equal(exp.Length, got.Length);
            for (int i = 0; i < got.Length; i++)
                Assert.True(MathF.Abs(got[i] - exp[i]) <= 2e-3f + 2e-3f * MathF.Abs(exp[i]),
                    $"{what}[{i}] {got[i]} != {exp[i]}");
        }

        Same("y", y.ToArray(), yRef.ToArray());
        Same("da", a.Grad!.ToArray(), aRef.Grad!.ToArray());
        Same("db", b.Grad!.ToArray(), bRef.Grad!.ToArray());
    }
}
