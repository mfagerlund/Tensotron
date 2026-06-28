using Tensotron;

namespace Tensotron.Tests;

public class OptimizerTests
{
    private static Optimizer Build(string kind, Tensor[] ps, Dictionary<string, float> c)
    {
        float G(string k, float d) => c.TryGetValue(k, out var v) ? v : d;
        return kind switch
        {
            "sgd" => new Sgd(ps, G("lr", 0.1f), G("momentum", 0f), G("weight_decay", 0f),
                             G("dampening", 0f), G("nesterov", 0f) != 0f),
            "adam" => new Adam(ps, G("lr", 1e-3f), G("beta1", 0.9f), G("beta2", 0.999f),
                               G("eps", 1e-8f), G("weight_decay", 0f)),
            "adamw" => new AdamW(ps, G("lr", 1e-3f), G("beta1", 0.9f), G("beta2", 0.999f),
                                 G("eps", 1e-8f), G("weight_decay", 0.01f)),
            "rmsprop" => new RmsProp(ps, G("lr", 1e-2f), G("alpha", 0.99f), G("eps", 1e-8f),
                                     G("weight_decay", 0f), G("momentum", 0f), G("centered", 0f) != 0f),
            _ => throw new InvalidOperationException(kind),
        };
    }

    [Fact]
    public void Optimizers_MatchTorch()
    {
        var fix = Fixtures.Load("optim");
        foreach (var c in fix.Cases)
        {
            var m = c.Meta!;
            var cfg = m.Config!;
            int steps = (int)cfg["steps"];

            var p = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var target = Fixtures.ToTensor(c.Inputs[1]);
            var opt = Build(m.Op!, new[] { p }, cfg);

            for (int s = 0; s < steps; s++)
            {
                opt.ZeroGrad();
                // loss = 0.5 * sum((p - target)^2) — same convex replay as the fixture.
                var loss = 0.5f * (p - target).Square().Sum();
                loss.Backward();
                opt.Step();
            }

            // 5–6 steps of float accumulation through pow/sqrt — loosen tolerance a touch.
            Fixtures.AssertMatches($"{c.Name} final param", c.Output, p, atol: 1e-3f, rtol: 1e-3f);
        }
    }

    [Fact]
    public void ClipGradNorm_ScalesToMaxNorm()
    {
        var p = Tensor.FromShaped(new[] { 3f, 4f }, new[] { 2 }).RequireGrad();
        // grad = [3,4] -> norm 5. Clip to 1 => scale 0.2 => [0.6,0.8], new norm 1.
        var loss = TensorOps.Sum(TensorOps.Mul(p, Tensor.FromShaped(new[] { 3f, 4f }, new[] { 2 })));
        loss.Backward();
        float norm = GradUtils.ClipGradNorm(new[] { p }, maxNorm: 1f);
        Assert.Equal(5f, norm, 3);
        var g = p.Grad!.ToArray();
        Assert.Equal(0.6f, g[0], 3);
        Assert.Equal(0.8f, g[1], 3);
    }
}
