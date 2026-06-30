using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// The <c>capturable</c> optimizer path reads the learning rate from a device scalar (so the step can
/// be folded into a captured graph). It must be numerically identical to the by-value path it mirrors,
/// under an annealed LR schedule — these run on BOTH backends (no capture involved), so they pin the
/// capturable kernels (GPU <c>AdamStepCapturable</c>/<c>SgdStepCapturable</c>, and the SIMD twin) to the
/// canonical fused optimizers. The capture/replay half (LR honoured across replays) is in TraceReplayTests.
/// </summary>
public class OptimizerCapturableTests
{
    private static float[] Rand(Random r, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() - 0.5);
        return a;
    }

    private static float[] Train(Func<IReadOnlyList<Tensor>, Optimizer> makeOpt,
                                 float[] xData, float[] tData, int batch, int inDim, int outDim, int steps)
    {
        Init.Seed(5);
        var m = new Sequential(new Linear(inDim, 8), Activation.Tanh(), new Linear(8, outDim));
        var x = Tensor.FromArray(xData, batch, inDim);
        var t = Tensor.FromArray(tData, batch, outDim);
        var ps = m.Parameters().ToList();
        var opt = makeOpt(ps);
        for (int k = 0; k < steps; k++)
        {
            opt.LearningRate = 1e-2f * (0.55f + 0.45f * MathF.Cos(k / (float)steps));   // anneal each step
            opt.ZeroGrad();
            var loss = TensorOps.MseLoss(m.Forward(x), t);
            loss.Backward();
            opt.Step();
            loss.DisposeGraph();
        }
        return ps[0].ToArray();
    }

    [Theory]
    [InlineData("adam")]
    [InlineData("adamw")]
    [InlineData("sgd")]
    public void Capturable_matches_byvalue_under_lr_schedule(string which)
    {
        var rng = new Random(7);
        const int batch = 8, inDim = 4, outDim = 2, steps = 15;
        var xData = Rand(rng, batch * inDim);
        var tData = Rand(rng, batch * outDim);

        Optimizer Make(IReadOnlyList<Tensor> ps, bool capturable) => which switch
        {
            "adam" => new Adam(ps, lr: 1e-2f, weightDecay: 0.01f, capturable: capturable),
            "adamw" => new AdamW(ps, lr: 1e-2f, weightDecay: 0.05f, capturable: capturable),
            "sgd" => new Sgd(ps, lr: 1e-2f, momentum: 0.9f, weightDecay: 0.01f, capturable: capturable),
            _ => throw new ArgumentException(which),
        };

        var byValue = Train(ps => Make(ps, false), xData, tData, batch, inDim, outDim, steps);
        var capturable = Train(ps => Make(ps, true), xData, tData, batch, inDim, outDim, steps);

        Assert.Equal(byValue.Length, capturable.Length);
        for (int i = 0; i < byValue.Length; i++)
            Assert.True(MathF.Abs(byValue[i] - capturable[i]) <= 1e-5f,
                $"{which} param[{i}] capturable {capturable[i]} != by-value {byValue[i]}");
    }
}
