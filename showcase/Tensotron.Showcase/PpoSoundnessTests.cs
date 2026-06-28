using Tensotron;
using Tensotron.Showcase.Environments;
using Tensotron.Showcase.Rl;
using Xunit.Abstractions;

namespace Tensotron.Showcase;

/// <summary>
/// PPO *correctness* benchmark against a problem with a known closed-form optimum (see
/// <see cref="PointReachEnv"/>: reward -(a-t)^2, optimum a* = t, optimal return 0). Unlike the
/// pole-cart showcase ("did the pole stay up", slow, threshold-noisy, conflates PPO with env
/// dynamics), this isolates the actor-critic + GAE + clipped-surrogate machinery and asserts it
/// drives the policy to the analytic optimum. One-step episodes keep it sub-second, so it's an
/// always-on gate (NOT Category=Showcase): if PPO is unsound, this fails fast on every run.
/// </summary>
public class PpoSoundnessTests
{
    private readonly ITestOutputHelper _out;
    public PpoSoundnessTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Ppo_ConvergesToKnownOptimum_PointReach()
    {
        Init.Seed(0);
        var rng = new Random(0);

        var ac = new ActorCritic(stateSize: 1, actionSize: 1, hidden: 32, initLogStd: -0.5f);
        var ppo = new ContinuousPpo(ac, () => new PointReachEnv(rng), rng)
        {
            NumEnvs = 64, Horizon = 8, Epochs = 4, MinibatchSize = 512, LearningRate = 3e-3f,
        };

        // Fixed evaluation set so before/after are comparable. Mean reward = -MSE(a, t); the
        // optimum is 0, an untrained policy is well below it.
        var evalTargets = new float[256];
        var er = new Random(123);
        for (int i = 0; i < evalTargets.Length; i++)
            evalTargets[i] = (float)(er.NextDouble() * 1.6 - 0.8);

        float before = MeanReward(ac, evalTargets);
        ppo.Train(80, (i, ret) => { if (i % 20 == 0 || i == 79) _out.WriteLine($"iter {i,3}: rolloutReturn={ret:0.000}"); });
        float after = MeanReward(ac, evalTargets);

        _out.WriteLine($"meanReward before={before:0.000}, after={after:0.000} (optimum 0)");

        // Converged near the analytic optimum (mean |a-t| ~ 0.1), and a clear, large improvement.
        Assert.True(after > -0.02f, $"PPO did not reach the known optimum (meanReward={after:0.000}, optimum 0).");
        Assert.True(after > before + 0.1f, $"PPO showed no learning toward the optimum (before={before:0.000}, after={after:0.000}).");
    }

    // Greedy (mean-action) mean reward over a fixed target set: -(a - t)^2 averaged.
    private static float MeanReward(ActorCritic ac, float[] targets)
    {
        using (Tensor.NoGradScope())
        {
            var st = Tensor.FromShaped(targets, new[] { targets.Length, 1 });
            var actions = ac.PolicyMean(st).ToArray();
            float sum = 0f;
            for (int i = 0; i < targets.Length; i++)
            {
                float d = actions[i] - targets[i];
                sum += -(d * d);
            }
            return sum / targets.Length;
        }
    }
}
