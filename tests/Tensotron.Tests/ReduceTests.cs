using Tensotron;

namespace Tensotron.Tests;

public class ReduceTests
{
    [Fact]
    public void Reductions_MatchTorch() => Run("reduce");

    // Deterministic edge cases: zeros under prod, ties under max/min (first-winner).
    [Fact]
    public void EdgeReductions_MatchTorch() => Run("reduce_edge");

    private static void Run(string fixture)
    {
        var fix = Fixtures.Load(fixture);
        foreach (var c in fix.Cases)
        {
            var m = c.Meta!;
            int[]? dims = m.Dims;
            int dim0 = dims is { Length: > 0 } ? dims[0] : 0;
            var x = Fixtures.ToTensor(c.Inputs[0]);
            bool noGrad = c.Grads.Count == 0;
            if (!noGrad) x.RequireGrad();

            Tensor y = m.Op switch
            {
                "mean" => TensorOps.Mean(x, dims, m.Keepdim),
                "max" => TensorOps.Max(x, dims, m.Keepdim),
                "min" => TensorOps.Min(x, dims, m.Keepdim),
                "prod" => TensorOps.Prod(x, dims, m.Keepdim),
                "var" => TensorOps.Var(x, dims, m.Keepdim),
                "std" => TensorOps.Std(x, dims, m.Keepdim),
                "logsumexp" => TensorOps.LogSumExp(x, dim0, m.Keepdim),
                "softmax" => TensorOps.Softmax(x, dim0),
                "log_softmax" => TensorOps.LogSoftmax(x, dim0),
                "argmax" => TensorOps.Argmax(x, dim0, m.Keepdim),
                "argmin" => TensorOps.Argmin(x, dim0, m.Keepdim),
                _ => throw new InvalidOperationException(m.Op),
            };

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            if (!noGrad)
            {
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
            }
        }
    }
}
