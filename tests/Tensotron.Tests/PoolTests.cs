using Tensotron;

namespace Tensotron.Tests;

public class PoolTests
{
    [Fact]
    public void Pool_MatchTorch()
    {
        var fix = Fixtures.Load("pool");
        foreach (var c in fix.Cases)
        {
            var cfg = c.Meta!.Config!;
            int k = (int)cfg["kernel"], stride = (int)cfg["stride"], pad = (int)cfg["padding"], dil = (int)cfg["dilation"];

            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var y = c.Meta.Op == "max_pool2d"
                ? TensorOps.MaxPool2d(x, k, stride, pad, dil)
                : TensorOps.AvgPool2d(x, k, stride, pad);

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad x", c.Grads[0], x.Grad!);
        }
    }
}
