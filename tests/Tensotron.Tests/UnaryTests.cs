using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// All unary ops in one fixture, dispatched by name. The template (shape passthrough
/// + grad) is proven once; each op just needs its value-parity case.
/// </summary>
public class UnaryTests
{
    private static readonly Dictionary<string, Func<Tensor, Tensor>> Ops = new()
    {
        ["neg"] = TensorOps.Neg,
        ["abs"] = TensorOps.Abs,
        ["sign"] = TensorOps.Sign,
        ["reciprocal"] = TensorOps.Reciprocal,
        ["square"] = TensorOps.Square,
        ["sqrt"] = TensorOps.Sqrt,
        ["rsqrt"] = TensorOps.Rsqrt,
        ["exp"] = TensorOps.Exp,
        ["log"] = TensorOps.Log,
        ["log1p"] = TensorOps.Log1p,
        ["sin"] = TensorOps.Sin,
        ["cos"] = TensorOps.Cos,
        ["tanh"] = TensorOps.Tanh,
        ["sigmoid"] = TensorOps.Sigmoid,
        ["relu"] = TensorOps.Relu,
        // unary.json's "gelu" case is the tanh approximation; the exact-erf default is in gelu.json
        // (see GeluTests).
        ["gelu"] = x => TensorOps.Gelu(x, approximateTanh: true),
        ["softplus"] = TensorOps.Softplus,
    };

    [Fact]
    public void AllUnary_MatchTorch()
    {
        var fix = Fixtures.Load("unary");
        foreach (var c in fix.Cases)
        {
            var op = Ops[c.Meta!.Op!];
            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var y = op(x);

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
        }
    }
}
