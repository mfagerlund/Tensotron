using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Every op is validated forward AND backward against torch-generated fixtures.
/// No torch at test time — just the committed JSON. Regenerate with
/// `python tools/fixtures/gen.py` (see each fixture's embedded `source`).
/// </summary>
public class ParityTests
{
    [Fact]
    public void Add_MatchesTorch() => RunBinary("add", TensorOps.Add);

    [Fact]
    public void Matmul_MatchesTorch() => RunBinary("matmul", TensorOps.MatMul);

    [Fact]
    public void Sum_MatchesTorch()
    {
        var fix = Fixtures.Load("sum");
        foreach (var c in fix.Cases)
        {
            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var y = TensorOps.Sum(x, c.Meta!.Dims, c.Meta.Keepdim);

            Fixtures.AssertMatches($"sum[{c.Name}] forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"sum[{c.Name}] grad(x)", c.Grads[0], x.Grad!);
        }
    }

    private static void RunBinary(string op, Func<Tensor, Tensor, Tensor> fn)
    {
        var fix = Fixtures.Load(op);
        foreach (var c in fix.Cases)
        {
            var a = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var b = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
            var y = fn(a, b);

            Fixtures.AssertMatches($"{op}[{c.Name}] forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{op}[{c.Name}] grad(a)", c.Grads[0], a.Grad!);
            Fixtures.AssertMatches($"{op}[{c.Name}] grad(b)", c.Grads[1], b.Grad!);
        }
    }
}
