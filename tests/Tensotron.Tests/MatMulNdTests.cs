using Tensotron;

namespace Tensotron.Tests;

public class MatMulNdTests
{
    [Fact]
    public void BatchedMatMul_MatchTorch()
    {
        var fix = Fixtures.Load("matmul_nd");
        foreach (var c in fix.Cases)
        {
            var a = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var b = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();

            Tensor y = c.Meta!.Op switch
            {
                "bmm" => TensorOps.Bmm(a, b),
                "matmul" => TensorOps.MatMul(a, b),
                "outer" => TensorOps.Outer(a, b),
                _ => throw new InvalidOperationException(c.Meta.Op),
            };

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad a", c.Grads[0], a.Grad!);
            Fixtures.AssertMatches($"{c.Name} grad b", c.Grads[1], b.Grad!);
        }
    }
}
