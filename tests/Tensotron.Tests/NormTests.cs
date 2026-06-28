using Tensotron;

namespace Tensotron.Tests;

public class NormTests
{
    [Fact]
    public void Norm_MatchTorch()
    {
        var fix = Fixtures.Load("norm");
        foreach (var c in fix.Cases)
        {
            var m = c.Meta!;
            if (m.Op == "layer_norm")
            {
                var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var w = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
                var b = Fixtures.ToTensor(c.Inputs[2]).RequireGrad();
                var y = TensorOps.LayerNorm(x, m.Dims!, w, b);

                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad x", c.Grads[0], x.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad w", c.Grads[1], w.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad b", c.Grads[2], b.Grad!);
            }
            else if (m.Op == "batch_norm")
            {
                int cFeat = m.Dims![0];
                var bn = new BatchNorm1d(cFeat);
                bn.Weight.Copy_(Fixtures.ToTensor(c.Inputs[1]));
                bn.Bias.Copy_(Fixtures.ToTensor(c.Inputs[2]));

                var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var y = bn.Forward(x);

                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad x", c.Grads[0], x.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad weight", c.Grads[1], bn.Weight.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad bias", c.Grads[2], bn.Bias.Grad!);
                Fixtures.AssertMatches($"{c.Name} running_mean", c.RunningMean!, bn.RunningMean);
                Fixtures.AssertMatches($"{c.Name} running_var", c.RunningVar!, bn.RunningVar);
            }
            else if (m.Op == "group_norm")
            {
                var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var w = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
                var b = Fixtures.ToTensor(c.Inputs[2]).RequireGrad();
                var y = TensorOps.GroupNorm(x, m.Dim, w, b);

                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad x", c.Grads[0], x.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad w", c.Grads[1], w.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad b", c.Grads[2], b.Grad!);
            }
            else if (m.Op == "batch_norm_2d")
            {
                int cFeat = m.Dims![0];
                var bn = new BatchNorm2d(cFeat);
                bn.Weight.Copy_(Fixtures.ToTensor(c.Inputs[1]));
                bn.Bias.Copy_(Fixtures.ToTensor(c.Inputs[2]));

                var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var y = bn.Forward(x);

                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad x", c.Grads[0], x.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad weight", c.Grads[1], bn.Weight.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad bias", c.Grads[2], bn.Bias.Grad!);
                Fixtures.AssertMatches($"{c.Name} running_mean", c.RunningMean!, bn.RunningMean);
                Fixtures.AssertMatches($"{c.Name} running_var", c.RunningVar!, bn.RunningVar);
            }
            else
            {
                throw new InvalidOperationException(m.Op);
            }
        }
    }

    [Fact]
    public void BatchNorm_EvalUsesRunningStats()
    {
        var bn = new BatchNorm1d(3);
        bn.RunningMean.Copy_(Tensor.FromShaped(new[] { 1f, 2f, 3f }, new[] { 3 }));
        bn.RunningVar.Copy_(Tensor.FromShaped(new[] { 4f, 4f, 4f }, new[] { 3 }));
        bn.Eval();

        // (x - mean)/sqrt(var+eps); gamma=1, beta=0. eps negligible vs var=4.
        var x = Tensor.FromShaped(new[] { 1f, 2f, 3f, 3f, 6f, 9f }, new[] { 2, 3 });
        var y = bn.Forward(x).ToArray();
        Assert.Equal(0f, y[0], 3);   // (1-1)/2
        Assert.Equal(0f, y[1], 3);   // (2-2)/2
        Assert.Equal(0f, y[2], 3);   // (3-3)/2
        Assert.Equal(1f, y[3], 3);   // (3-1)/2
        Assert.Equal(2f, y[4], 3);   // (6-2)/2
        Assert.Equal(3f, y[5], 3);   // (9-3)/2
    }
}
