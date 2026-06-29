using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Exact (erf) GELU — torch's DEFAULT F.gelu(approximate='none') / nn.GELU(), which is what
/// Tensotron's Gelu() uses by default. The tanh approximation is covered separately by the "gelu"
/// case in unary.json (Tensotron's Gelu(approximateTanh: true)).
/// </summary>
public class GeluTests
{
    [Fact]
    public void GeluErf_MatchesTorch()
    {
        var fix = Fixtures.Load("gelu");
        Assert.NotEmpty(fix.Cases);
        foreach (var c in fix.Cases)
        {
            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var y = TensorOps.Gelu(x);   // default = exact erf

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
        }
    }
}
