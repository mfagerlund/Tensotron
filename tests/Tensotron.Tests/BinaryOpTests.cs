using Tensotron;

namespace Tensotron.Tests;

public class BinaryOpTests
{
    private static readonly Dictionary<string, Func<Tensor, Tensor, Tensor>> Ops = new()
    {
        ["div"] = TensorOps.Div,
        ["pow"] = TensorOps.Pow,
        ["maximum"] = TensorOps.Maximum,
        ["minimum"] = TensorOps.Minimum,
    };

    [Fact]
    public void Binary_MatchTorch()
    {
        var fix = Fixtures.Load("binary");
        foreach (var c in fix.Cases)
        {
            var op = Ops[c.Meta!.Op!];
            var a = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var b = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
            var y = op(a, b);

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad(a)", c.Grads[0], a.Grad!);
            Fixtures.AssertMatches($"{c.Name} grad(b)", c.Grads[1], b.Grad!);
        }
    }

    [Fact]
    public void Composite_MatchTorch()
    {
        var fix = Fixtures.Load("composite");
        foreach (var c in fix.Cases)
        {
            var p = c.Meta!.Params!;
            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            Tensor y = c.Meta.Op switch
            {
                "clamp" => TensorOps.Clamp(x, p[0], p[1]),
                "leaky_relu" => TensorOps.LeakyRelu(x, p[0]),
                "elu" => TensorOps.Elu(x, p[0]),
                _ => throw new InvalidOperationException(c.Meta.Op),
            };

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
        }
    }
}
