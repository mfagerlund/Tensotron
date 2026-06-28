using Tensotron;

namespace Tensotron.Tests;

public class SelectTests
{
    [Fact]
    public void Select_MatchTorch()
    {
        var fix = Fixtures.Load("select");
        foreach (var c in fix.Cases)
        {
            if (c.Meta!.Op == "where")
            {
                var a = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var b = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
                // cond = a > 0 (no grad), matching the fixture's torch.where(a > 0, a, b).
                var y = TensorOps.Where(a > 0f, a, b);

                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad a", c.Grads[0], a.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad b", c.Grads[1], b.Grad!);
            }
            else if (c.Meta.Op == "masked_fill")
            {
                var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var mask = Fixtures.ToTensor(c.Inputs[1]); // 0/1 float, no grad
                var y = TensorOps.MaskedFill(x, mask, c.Meta.Params![0]);

                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
            }
            else
            {
                throw new InvalidOperationException(c.Meta.Op);
            }
        }
    }
}
