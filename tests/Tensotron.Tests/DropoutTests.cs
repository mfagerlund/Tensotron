using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Dropout is stochastic, so it can't be golden-fixtured against torch like the deterministic ops.
/// Instead we verify the properties torch guarantees for inverted dropout: identity in eval mode,
/// identity at p=0, kept units scaled by 1/(1-p) with ~p fraction zeroed in train mode, and the
/// gradient flowing only through the kept units.
/// </summary>
public class DropoutTests
{
    [Fact]
    public void Eval_IsIdentity()
    {
        var d = new Dropout(0.5f);
        d.Eval();
        var x = Tensor.FromArray(new[] { -2f, -0.5f, 0f, 1f, 3f }, 5);
        Assert.Equal(x.ToArray(), d.Forward(x).ToArray());
    }

    [Fact]
    public void ZeroP_IsIdentity_EvenInTrainMode()
    {
        var d = new Dropout(0f);
        Assert.True(d.Training);
        var x = Tensor.FromArray(new[] { 1f, 2f, 3f, 4f }, 4);
        Assert.Equal(x.ToArray(), d.Forward(x).ToArray());
    }

    [Fact]
    public void Train_ScalesKeptByInverseKeep_AndDropsAboutPFraction()
    {
        const float p = 0.5f;
        float scale = 1f / (1f - p);
        var d = new Dropout(p, seed: 1);
        var x = Tensor.FromShaped(Enumerable.Repeat(1f, 10000).ToArray(), new[] { 10000 });
        var y = d.Forward(x).ToArray();

        int zeros = 0;
        foreach (var v in y)
        {
            if (v == 0f) zeros++;
            else Assert.True(MathF.Abs(v - scale) < 1e-6f, $"kept value {v} != inverse-keep scale {scale}");
        }
        float dropRate = zeros / (float)y.Length;
        Assert.True(MathF.Abs(dropRate - p) < 0.03f, $"drop rate {dropRate} not near p={p}");
    }

    [Fact]
    public void Train_Backward_FlowsThroughKeptUnitsOnly()
    {
        var d = new Dropout(0.5f, seed: 2);
        var ones = Enumerable.Repeat(1f, 256).ToArray();
        var x = Tensor.FromShaped((float[])ones.Clone(), new[] { 256 }).RequireGrad();
        var y = d.Forward(x);
        y.Backward(Tensor.FromShaped((float[])ones.Clone(), new[] { 256 }));

        var mask = y.ToArray();         // x is all ones, so y == the inverted-dropout mask
        var g = x.Grad!.ToArray();
        // d(x*mask)/dx == mask, so the gradient must equal the forward mask exactly.
        for (int i = 0; i < g.Length; i++)
            Assert.True(MathF.Abs(g[i] - mask[i]) < 1e-6f, $"grad[{i}]={g[i]} != mask {mask[i]}");
    }
}
