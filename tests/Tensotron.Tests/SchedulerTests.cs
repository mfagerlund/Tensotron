using Tensotron;

namespace Tensotron.Tests;

public class SchedulerTests
{
    [Fact]
    public void Schedulers_MatchTorch()
    {
        var fix = Fixtures.Load("sched");
        foreach (var c in fix.Cases)
        {
            var cfg = c.Meta!.Config!;
            float G(string k, float d) => cfg.TryGetValue(k, out var v) ? v : d;

            var p = Tensor.Zeros(new Shape(1)).RequireGrad();
            var opt = new Sgd(new[] { p }, G("lr", 0.1f));
            LrScheduler sch = c.Meta.Op switch
            {
                "step" => new StepLR(opt, (int)cfg["step_size"], cfg["gamma"]),
                "exp" => new ExponentialLR(opt, cfg["gamma"]),
                "cosine" => new CosineAnnealingLR(opt, (int)cfg["t_max"], G("eta_min", 0f)),
                "linear" => new LinearLR(opt, cfg["start"], cfg["end"], (int)cfg["total"]),
                _ => throw new InvalidOperationException(c.Meta.Op),
            };

            var expected = c.Output.Data;
            var got = new float[expected.Length];
            got[0] = sch.CurrentLr;
            for (int i = 1; i < expected.Length; i++)
            {
                sch.Step();
                got[i] = sch.CurrentLr;
            }

            for (int i = 0; i < expected.Length; i++)
                Assert.True(MathF.Abs(got[i] - expected[i]) <= 1e-5f + 1e-4f * MathF.Abs(expected[i]),
                    $"{c.Name} step {i}: lr {got[i]} != torch {expected[i]}");
        }
    }
}
