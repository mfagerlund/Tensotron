using Tensotron;
using Tensotron.Showcase.Environments;
using Tensotron.Showcase.Rl;
using Xunit.Abstractions;

namespace Tensotron.Showcase;

/// <summary>
/// "Tensotron can be used for RL II": train a continuous PPO controller — built entirely on
/// Tensotron tensors — to drive a point-car around a closed wavy corridor using five raycast
/// whiskers, and assert it actually learns. Emits an SVG replay of the trained controller's
/// lap (a time-graded trajectory over the track walls).
///
/// Tagged <c>Category=Showcase</c> (excluded from the normal suite). Unlike the pole-cart
/// demos this is light enough to converge on the managed/SIMD CPU backend in a couple of
/// minutes, so it is not GPU-gated — run it with
/// <c>TENSOTRON_BACKEND=simd dotnet test --filter "FullyQualifiedName~CorridorFollower"</c>
/// or on a GPU via <c>tools/run-tests.ps1 -Showcase</c>.
/// </summary>
[Trait("Category", "Showcase")]
public class CorridorTests
{
    private readonly ITestOutputHelper _out;
    public CorridorTests(ITestOutputHelper output) => _out = output;

    private static string OutputDir
    {
        get
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "ShowcaseOutput");
            Directory.CreateDirectory(dir);
            return dir;
        }
    }

    [Fact]
    public void CorridorFollower_PpoLearnsToDrive()
    {
        _out.WriteLine($"Device: {Accelerators.Active().Name}");

        const int maxSteps = 800;
        Init.Seed(20);
        var rng = new Random(20);

        var ac = new ActorCritic(stateSize: 5, actionSize: 1, hidden: 64);
        var ppo = new ContinuousPpo(ac, () => new CorridorEnv(rng, maxSteps), rng)
        {
            NumEnvs = 16, Horizon = 256, Epochs = 10, MinibatchSize = 512, LearningRate = 2e-3f,
        };

        ppo.Train(120, (i, ret) => { if (i % 10 == 0 || i == 119) _out.WriteLine($"iter {i,3}: meanReturn={ret:0.0}"); });

        float meanSteps = ppo.EvaluateMeanSteps(episodes: 10, maxSteps: maxSteps);
        _out.WriteLine($"FINAL greedy meanSteps={meanSteps:0.0} / {maxSteps}");

        // Render a clean, deterministic lap of the trained controller.
        var demo = new CorridorEnv(new Random(123), maxSteps);
        RunGreedyEpisode(ac, demo, maxSteps);
        var path = Path.Combine(OutputDir, "corridor.svg");
        demo.RenderToSvg(path);
        _out.WriteLine($"Rendered {path} ({demo.Trajectory.Count} steps)");

        Assert.True(meanSteps >= maxSteps * 0.85f,
            $"PPO failed to learn corridor following (meanSteps={meanSteps} / {maxSteps}).");
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
