using Tensotron;

namespace Tensotron.Tests;

public class ConvTests
{
    [Fact]
    public void Conv2d_MatchTorch()
    {
        var fix = Fixtures.Load("conv");
        foreach (var c in fix.Cases)
        {
            var cfg = c.Meta!.Config!;
            int stride = (int)cfg["stride"], pad = (int)cfg["padding"], dil = (int)cfg["dilation"];
            bool bias = cfg["bias"] != 0f;

            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var w = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
            Tensor? b = bias ? Fixtures.ToTensor(c.Inputs[2]).RequireGrad() : null;

            var y = TensorOps.Conv2d(x, w, b, stride, pad, dil);
            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad x", c.Grads[0], x.Grad!);
            Fixtures.AssertMatches($"{c.Name} grad weight", c.Grads[1], w.Grad!);
            if (bias)
                Fixtures.AssertMatches($"{c.Name} grad bias", c.Grads[2], b!.Grad!);
        }
    }
}
