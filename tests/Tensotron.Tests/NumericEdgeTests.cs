using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Special-value parity the JSON golden fixtures can't carry: System.Text.Json rejects the bare
/// <c>NaN</c>/<c>Infinity</c> tokens Python emits, so these forward/backward expectations are
/// torch-derived (recorded via tools/fixtures/torch_probe.py) and asserted directly.
/// torch.maximum/minimum propagate NaN; torch.where / masked_fill are true selects that never
/// leak a NaN/Inf out of the unselected branch; x.pow(2) is differentiable at x == 0.
/// </summary>
public class NumericEdgeTests
{
    private const float NaN = float.NaN;
    private const float Inf = float.PositiveInfinity;

    [Fact]
    public void MaximumMinimum_PropagateNaN()
    {
        var a = Tensor.FromArray(new[] { NaN, 1f, 3f, NaN }, 4);
        var b = Tensor.FromArray(new[] { 1f, NaN, 2f, NaN }, 4);

        var mx = TensorOps.Maximum(a, b).ToArray();
        var mn = TensorOps.Minimum(a, b).ToArray();

        // A NaN on either side -> NaN; only the finite pair (3,2) resolves normally.
        Assert.True(float.IsNaN(mx[0]) && float.IsNaN(mx[1]) && mx[2] == 3f && float.IsNaN(mx[3]));
        Assert.True(float.IsNaN(mn[0]) && float.IsNaN(mn[1]) && mn[2] == 2f && float.IsNaN(mn[3]));
    }

    [Fact]
    public void Where_TrueSelect_NoNaNLeakFromUnselectedBranch()
    {
        // cond = a > 0. NaN/Inf sit in the UNSELECTED branch at every lane; a true select must
        // discard them (torch.where), unlike an arithmetic blend where 0*NaN = NaN would poison it.
        var a = Tensor.FromArray(new[] { 2f, -1f, 3f, -4f }, 4).RequireGrad();
        var b = Tensor.FromArray(new[] { NaN, 5f, Inf, 6f }, 4).RequireGrad();

        var y = TensorOps.Where(a > 0f, a, b);

        var f = y.ToArray();
        var expected = new[] { 2f, 5f, 3f, 6f };
        for (int i = 0; i < 4; i++)
            Assert.True(f[i] == expected[i], $"where[{i}] = {f[i]} expected {expected[i]}");

        y.Backward(Tensor.Ones(new Shape(4)));
        // Gradient flows to the selected side only (torch): da = [1,0,1,0], db = [0,1,0,1].
        Assert.Equal(new[] { 1f, 0f, 1f, 0f }, a.Grad!.ToArray());
        Assert.Equal(new[] { 0f, 1f, 0f, 1f }, b.Grad!.ToArray());
    }

    [Fact]
    public void MaskedFill_TrueSelect_NoNaNLeak()
    {
        // x carries NaN/Inf exactly where the mask replaces it; the fill must win (torch.masked_fill).
        var x = Tensor.FromArray(new[] { NaN, 2f, Inf, 4f }, 4).RequireGrad();
        var mask = Tensor.FromArray(new[] { 1f, 0f, 1f, 0f }, 4);

        var y = x.MaskedFill(mask, 0.5f);

        var f = y.ToArray();
        var expected = new[] { 0.5f, 2f, 0.5f, 4f };
        for (int i = 0; i < 4; i++)
            Assert.True(f[i] == expected[i], $"masked_fill[{i}] = {f[i]} expected {expected[i]}");

        y.Backward(Tensor.Ones(new Shape(4)));
        // Gradient reaches the kept positions only (torch): dx = [0,1,0,1].
        Assert.Equal(new[] { 0f, 1f, 0f, 1f }, x.Grad!.ToArray());
    }

    [Fact]
    public void Pow_ZeroBase_GradientIsFinite()
    {
        // x.pow(2) at x == 0 must give grad 0 (b·a^(b-1) = 2·0¹), not 0/0 = NaN.
        var x = Tensor.FromArray(new[] { 0f, 0f, 2f }, 3).RequireGrad();

        var y = x.Pow(2f).Sum();
        y.Backward();

        Assert.Equal(new[] { 0f, 0f, 4f }, x.Grad!.ToArray());   // d/dx x² = 2x
    }
}
