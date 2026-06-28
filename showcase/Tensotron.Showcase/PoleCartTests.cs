using Tensotron;
using Tensotron.Showcase.Environments;
using Tensotron.Showcase.Rl;
using Xunit.Abstractions;

namespace Tensotron.Showcase;

/// <summary>
/// "Tensotron can be used for RL": train a continuous PPO controller, built on Tensotron
/// tensors, to balance pole-carts — and assert it actually learns. Emits an SVG replay of
/// the trained controller's run.
///
/// Tagged <c>Category=Showcase</c>: these run full-strength convergence (minutes-to-tens-of-
/// minutes on a GPU; impractically slow on the CPU accelerator) and are EXCLUDED from the
/// normal suite. Run them explicitly — on a CUDA GPU — via <c>tools/run-tests.ps1 -Showcase</c>
/// or <c>dotnet test --filter "Category=Showcase"</c>. The always-on <see cref="ShowcaseSmokeTests"/>
/// guard the same code paths cheaply on every run.
/// </summary>
[Trait("Category", "Showcase")]
public class PoleCartTests
{
    private readonly ITestOutputHelper _out;
    public PoleCartTests(ITestOutputHelper output) => _out = output;

    private static string OutputDir
    {
        get
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "ShowcaseOutput");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    [SkippableFact]
    public void SinglePoleCart_PpoLearnsToBalance()
    {
        Skip.IfNot(Cuda.IsAvailable(),
            "Showcase convergence runs on a CUDA GPU; none detected (CPU is too slow for full training).");
        _out.WriteLine($"Device: {Accelerators.Active().Name}");

        const int maxSteps = 500;
        Init.Seed(1234);
        var rng = new Random(1234);

        var ac = new ActorCritic(stateSize: 4, actionSize: 1, hidden: 64);
        var ppo = new ContinuousPpo(ac, () => new SinglePoleCart(rng, maxSteps), rng)
        {
            NumEnvs = 16, Horizon = 256, Epochs = 10, MinibatchSize = 512, LearningRate = 2e-3f,
        };

        ppo.Train(40, (i, ret) => { if (i % 5 == 0 || i == 39) _out.WriteLine($"iter {i,3}: meanReturn={ret:0.0}"); });

        float meanSteps = ppo.EvaluateMeanSteps(episodes: 10, maxSteps: maxSteps);
        _out.WriteLine($"FINAL greedy meanSteps={meanSteps:0.0} / {maxSteps}");

        // Render a replay of the trained controller.
        var demo = new SinglePoleCart(rng, maxSteps);
        RunGreedyEpisode(ac, demo, maxSteps);
        demo.RenderToSvg(Path.Combine(OutputDir, "single_pole.svg"));

        Assert.True(meanSteps >= 450f, $"PPO failed to learn single pole balancing (meanSteps={meanSteps}).");
    }

    [SkippableFact]
    public void DoublePoleCart_PpoLearnsToBalance()
    {
        Skip.IfNot(Cuda.IsAvailable(),
            "Showcase convergence runs on a CUDA GPU; none detected (CPU is too slow for full training).");
        _out.WriteLine($"Device: {Accelerators.Active().Name}");

        const int maxSteps = 1000;
        Init.Seed(7);
        var rng = new Random(7);

        var ac = new ActorCritic(stateSize: 6, actionSize: 1, hidden: 64);
        var ppo = new ContinuousPpo(ac, () => new DoublePoleCart(maxSteps), rng)
        {
            NumEnvs = 16, Horizon = 256, Epochs = 10, MinibatchSize = 512, LearningRate = 2e-3f,
        };

        ppo.Train(60, (i, ret) => { if (i % 5 == 0 || i == 59) _out.WriteLine($"iter {i,3}: meanReturn={ret:0.0}"); });

        float meanSteps = ppo.EvaluateMeanSteps(episodes: 5, maxSteps: maxSteps);
        _out.WriteLine($"FINAL greedy meanSteps={meanSteps:0.0} / {maxSteps}");

        var demo = new DoublePoleCart(maxSteps);
        RunGreedyEpisode(ac, demo, maxSteps);
        demo.RenderToSvg(Path.Combine(OutputDir, "double_pole.svg"));

        // Deterministic start tips immediately; an untrained/poor controller falls within
        // a few dozen steps. Require clear, sustained balancing.
        Assert.True(meanSteps >= 300f, $"PPO failed to learn double pole balancing (meanSteps={meanSteps}).");
    }

    private static void RunGreedyEpisode(ActorCritic ac, IEnvironment env, int maxSteps)
    {
        using (Tensor.NoGradScope())
        {
            var s = env.Reset();
            for (int i = 0; i < maxSteps; i++)
            {
                var st = Tensor.FromShaped(s, new[] { 1, ac.StateSize });
                var mean = ac.PolicyMean(st).ToArray();
                var (_, done) = env.Step(Math.Clamp(mean[0], -1f, 1f));
                s = env.GetState();
                if (done) break;
            }
        }
    }
}
