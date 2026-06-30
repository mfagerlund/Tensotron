using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Lifetime hazards around zero-copy views (Reshape/Squeeze/Flatten/Detach share the parent's
/// buffer). These reproduce the failures behaviorally — by forcing the caching allocator to
/// hand a recycled buffer back out and checking the still-live tensor was not corrupted —
/// rather than relying on GC timing.
/// </summary>
public class ViewLifetimeTests
{
    // Churn same-size allocations so any buffer that was returned to the pool gets handed back
    // out and overwritten; if a live view aliased it, the overwrite shows up.
    private static void ClobberPool(int n)
    {
        for (int k = 0; k < 4; k++)
        {
            var c = Tensor.Ones(new Shape(n)); // memset to a non-zero sentinel
            _ = c.ToArray();
            c.Dispose();
        }
    }

    [Fact]
    public void DisposeGraph_OnViewRoot_LeavesRootReadable()
    {
        const int n = 9173; // distinctive size, unlikely to collide with other pooled buckets
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = i * 0.5f + 1f;
        var x = Tensor.FromArray(d, n).RequireGrad();

        // Root of the graph is a zero-copy VIEW; its parent (x*x) owns the shared buffer.
        var root = (x * x).Reshape(n);
        var expected = root.ToArray();

        root.DisposeGraph();   // must not recycle the buffer the root view still points at
        ClobberPool(n);

        var got = root.ToArray();
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(expected[i] - got[i]) <= 1e-3f,
                $"root[{i}] = {got[i]} expected {expected[i]} — DisposeGraph recycled a live view buffer");
    }

    [Fact]
    public void GradAccumulation_ThroughReshape_IsCorrect()
    {
        const int n = 4096;
        var d = new float[n];
        for (int i = 0; i < n; i++) d[i] = 1f;
        var w = Tensor.FromArray(d, n).RequireGrad();

        // Two independent graphs, no ZeroGrad between them: the leaf grad accumulates.
        // w is consumed through Reshape, whose backward feeds AddGrad a non-owning view.
        var loss1 = w.Reshape(n).Sum();
        loss1.Backward();           // w.Grad <- ones (size n)
        loss1.DisposeGraph();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        ClobberPool(n);

        var loss2 = w.Reshape(n).Sum();
        loss2.Backward();           // AddInto(w.Grad, ones)
        loss2.DisposeGraph();

        var g = w.Grad!.ToArray();
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(g[i] - 2f) <= 1e-3f,
                $"grad[{i}] = {g[i]} expected 2 — reshape grad aliasing corrupted accumulation");
    }
}
