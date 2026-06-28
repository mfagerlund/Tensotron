using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Exercises the tiled shared-memory GEMM (only taken when M,N,K ≥ 64) in all three stride
/// configurations it must serve: the contiguous forward, and the two transposed-operand
/// products in MatMul backward (dA = W·Bᵀ, dB = Aᵀ·W). Compared against a CPU reference so the
/// tiling, boundary masking (dims not multiples of the tile), and strided access are all checked.
/// </summary>
public class TiledMatMulTests
{
    private static float[] CpuMatMul(float[] a, float[] b, int M, int K, int N)
    {
        var c = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++) acc += a[m * K + k] * b[k * N + n];
                c[m * N + n] = acc;
            }
        return c;
    }

    private static float[] Transpose(float[] a, int R, int C)
    {
        var t = new float[R * C];
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) t[c * R + r] = a[r * C + c];
        return t;
    }

    [Fact]
    public void TiledGemm_ForwardAndGradients_MatchCpu()
    {
        // Dims ≥ 64 (so the tiled path is taken) and deliberately NOT multiples of 16 (so
        // boundary masking is exercised).
        const int M = 96, K = 80, N = 112;
        var rng = new Random(1234);
        var aData = new float[M * K];
        var bData = new float[K * N];
        var wData = new float[M * N]; // fixed upstream gradient weights
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < wData.Length; i++) wData[i] = (float)(rng.NextDouble() - 0.5);

        var A = Tensor.FromArray(aData, M, K).RequireGrad();
        var B = Tensor.FromArray(bData, K, N).RequireGrad();
        var W = Tensor.FromArray(wData, M, N);

        var C = TensorOps.MatMul(A, B);                 // forward (tiled)
        var loss = TensorOps.Sum(TensorOps.Mul(C, W));  // L = Σ C∘W
        loss.Backward();

        // Forward reference.
        var cRef = CpuMatMul(aData, bData, M, K, N);
        var cGot = C.ToArray();
        for (int i = 0; i < cRef.Length; i++)
            Assert.True(MathF.Abs(cGot[i] - cRef[i]) <= 2e-3f, $"C[{i}] {cGot[i]} != {cRef[i]}");

        // dA = W · Bᵀ  (exercises tiled with the B operand stride-transposed).
        var bT = Transpose(bData, K, N);
        var dARef = CpuMatMul(wData, bT, M, N, K);
        var dAGot = A.Grad!.ToArray();
        for (int i = 0; i < dARef.Length; i++)
            Assert.True(MathF.Abs(dAGot[i] - dARef[i]) <= 3e-3f, $"dA[{i}] {dAGot[i]} != {dARef[i]}");

        // dB = Aᵀ · W  (exercises tiled with the A operand stride-transposed).
        var aT = Transpose(aData, M, K);
        var dBRef = CpuMatMul(aT, wData, K, M, N);
        var dBGot = B.Grad!.ToArray();
        for (int i = 0; i < dBRef.Length; i++)
            Assert.True(MathF.Abs(dBGot[i] - dBRef[i]) <= 3e-3f, $"dB[{i}] {dBGot[i]} != {dBRef[i]}");
    }
}
