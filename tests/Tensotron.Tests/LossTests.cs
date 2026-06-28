using Tensotron;

namespace Tensotron.Tests;

public class LossTests
{
    private static Reduction Parse(string? r) => r switch
    {
        "mean" => Reduction.Mean,
        "sum" => Reduction.Sum,
        "none" => Reduction.None,
        _ => throw new InvalidOperationException($"reduction {r}"),
    };

    [Fact]
    public void Losses_MatchTorch()
    {
        var fix = Fixtures.Load("losses");
        foreach (var c in fix.Cases)
        {
            var m = c.Meta!;
            var red = Parse(m.Reduction);
            var input = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();

            Tensor y;
            switch (m.Op)
            {
                case "mse": y = TensorOps.MseLoss(input, Fixtures.ToTensor(c.Inputs[1]), red); break;
                case "l1": y = TensorOps.L1Loss(input, Fixtures.ToTensor(c.Inputs[1]), red); break;
                case "huber": y = TensorOps.HuberLoss(input, Fixtures.ToTensor(c.Inputs[1]), m.Params![0], red); break;
                case "bce_with_logits": y = TensorOps.BceWithLogits(input, Fixtures.ToTensor(c.Inputs[1]), red); break;
                case "nll": y = TensorOps.NllLoss(input, m.Index!, red); break;
                case "cross_entropy": y = TensorOps.CrossEntropy(input, m.Index!, red); break;
                case "kl_div": y = TensorOps.KlDiv(input, Fixtures.ToTensor(c.Inputs[1]), red); break;
                default: throw new InvalidOperationException(m.Op);
            }

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad input", c.Grads[0], input.Grad!);
        }
    }
}
