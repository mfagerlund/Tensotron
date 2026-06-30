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

    // The fused single-node MseLoss must stay behaviourally identical to the composed
    // Sub→Square→reduce expression (which losses.json already pins to torch). This nails
    // forward + BOTH input and target gradients across every reduction.
    [Theory]
    [InlineData("mean")]
    [InlineData("sum")]
    [InlineData("none")]
    public void MseLoss_FusedMatchesComposed(string reduction)
    {
        var red = Parse(reduction);
        var rng = new Random(17);
        const int N = 4, D = 5;
        float[] Rand(int c) { var a = new float[c]; for (int i = 0; i < c; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var xData = Rand(N * D);
        var tData = Rand(N * D);
        var gData = red == Reduction.None ? Rand(N * D) : new[] { 1.3f };

        var xf = Tensor.FromShaped((float[])xData.Clone(), new[] { N, D }).RequireGrad();
        var tf = Tensor.FromShaped((float[])tData.Clone(), new[] { N, D }).RequireGrad();
        var yf = TensorOps.MseLoss(xf, tf, red);

        var xc = Tensor.FromShaped((float[])xData.Clone(), new[] { N, D }).RequireGrad();
        var tc = Tensor.FromShaped((float[])tData.Clone(), new[] { N, D }).RequireGrad();
        var sq = TensorOps.Square(TensorOps.Sub(xc, tc));
        var yc = red switch
        {
            Reduction.Mean => TensorOps.Mean(sq),
            Reduction.Sum => TensorOps.Sum(sq),
            _ => sq,
        };

        yf.Backward(Tensor.FromShaped((float[])gData.Clone(), yf.Shape.Dims));
        yc.Backward(Tensor.FromShaped((float[])gData.Clone(), yc.Shape.Dims));

        Close("forward", yf.ToArray(), yc.ToArray());
        Close("dInput", xf.Grad!.ToArray(), xc.Grad!.ToArray());
        Close("dTarget", tf.Grad!.ToArray(), tc.Grad!.ToArray());
    }

    // A broadcast target (N,1 against N,D) makes dTarget reduce over the broadcast axis —
    // the fused path routes through ReduceGradToShape exactly like the composed Sub does.
    [Fact]
    public void MseLoss_Fused_BroadcastTarget_MatchesComposed()
    {
        var rng = new Random(23);
        const int N = 4, D = 5;
        float[] Rand(int c) { var a = new float[c]; for (int i = 0; i < c; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var xData = Rand(N * D);
        var tData = Rand(N);

        var xf = Tensor.FromShaped((float[])xData.Clone(), new[] { N, D }).RequireGrad();
        var tf = Tensor.FromShaped((float[])tData.Clone(), new[] { N, 1 }).RequireGrad();
        var yf = TensorOps.MseLoss(xf, tf, Reduction.Mean);

        var xc = Tensor.FromShaped((float[])xData.Clone(), new[] { N, D }).RequireGrad();
        var tc = Tensor.FromShaped((float[])tData.Clone(), new[] { N, 1 }).RequireGrad();
        var yc = TensorOps.Mean(TensorOps.Square(TensorOps.Sub(xc, tc)));

        yf.Backward(Tensor.FromShaped(new[] { 1f }, yf.Shape.Dims));
        yc.Backward(Tensor.FromShaped(new[] { 1f }, yc.Shape.Dims));

        Close("forward", yf.ToArray(), yc.ToArray());
        Close("dInput", xf.Grad!.ToArray(), xc.Grad!.ToArray());
        Close("dTarget", tf.Grad!.ToArray(), tc.Grad!.ToArray());
    }

    private static void Close(string tag, float[] got, float[] exp, float tol = 1e-5f)
    {
        Assert.Equal(exp.Length, got.Length);
        for (int i = 0; i < exp.Length; i++)
            Assert.True(MathF.Abs(got[i] - exp[i]) <= tol, $"{tag}[{i}] {got[i]} != {exp[i]}");
    }
}
