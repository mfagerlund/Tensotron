using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// The caching allocator (size-bucketed pool fed by Tensor.Dispose) and the per-step
/// graph recycle (Tensor.DisposeGraph) must be numerically transparent: reusing device
/// buffers must not change any result. Trains the same net twice — once recycling each
/// step's graph into the pool, once not — and requires the final parameters to match.
/// Tolerance, not bit-equality: the parallel reduction accumulates with atomics (per the
/// "not bitwise-deterministic" design), so two runs of the same math differ at ~1e-7;
/// pooling corruption would diverge by orders of magnitude, which this still catches.
/// </summary>
public class AllocatorPoolTests
{
    private static (Tensor w, Tensor b) TrainXorish(bool recycle, int steps)
    {
        Init.Seed(123);
        var rng = new Random(123);
        var data = new float[64 * 8];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var x = Tensor.FromArray(data, 64, 8);
        var target = Tensor.FromArray(new float[64], 64, 1);

        var l1 = new Linear(8, 16);
        var l2 = new Linear(16, 1);
        var model = new Sequential(l1, Activation.Relu(), l2);
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-2f);

        for (int s = 0; s < steps; s++)
        {
            var loss = TensorOps.MseLoss(model.Forward(x), target);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (recycle) loss.DisposeGraph();
        }
        return (l1.Weight, l2.Weight);
    }

    [Fact]
    public void DisposeGraph_AndPooling_AreNumericallyTransparent()
    {
        var (w1Plain, w2Plain) = TrainXorish(recycle: false, steps: 25);
        var plain1 = w1Plain.ToArray();
        var plain2 = w2Plain.ToArray();

        var (w1Pool, w2Pool) = TrainXorish(recycle: true, steps: 25);
        var pool1 = w1Pool.ToArray();
        var pool2 = w2Pool.ToArray();

        Assert.Equal(plain1.Length, pool1.Length);
        for (int i = 0; i < plain1.Length; i++)
            Assert.True(MathF.Abs(plain1[i] - pool1[i]) <= 1e-3f, $"w1[{i}] {pool1[i]} vs {plain1[i]}");
        for (int i = 0; i < plain2.Length; i++)
            Assert.True(MathF.Abs(plain2[i] - pool2[i]) <= 1e-3f, $"w2[{i}] {pool2[i]} vs {plain2[i]}");
    }

    [Fact]
    public void Pool_ReusesBufferOfMatchingSize()
    {
        var rt = TensorRuntime.Instance;
        long before = rt.PoolHits;
        // Free a buffer of a distinctive size, then allocate the same size — must be served from pool.
        const int n = 9173;
        var a = Tensor.FromArray(new float[n], n);
        a.Dispose();                       // returns to pool
        var b = Tensor.Zeros(new Shape(n)); // should reuse the pooled buffer
        Assert.True(rt.PoolHits > before, "expected a pool hit for the matching-size reallocation");
        Assert.Equal(0f, b.ToArray()[0]);   // Zeros still initializes correctly over the reused buffer
        b.Dispose();
    }
}
