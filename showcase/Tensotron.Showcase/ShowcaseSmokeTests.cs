using Tensotron;
using Tensotron.Showcase.Environments;
using Tensotron.Showcase.Rl;
using Xunit.Abstractions;

namespace Tensotron.Showcase;

/// <summary>
/// Fast, always-on guards for the showcase code paths. Unlike the full
/// <c>Category=Showcase</c> convergence tests (which take minutes-to-tens-of-minutes and only
/// run on demand / on a GPU), these are sized to finish quickly on the CPU accelerator while
/// still exercising the same machinery end to end:
///   * the continuous-PPO + actor-critic + autograd + optimizer loop, and
///   * the Conv2d → MaxPool2d → Linear stack training under cross-entropy.
/// They assert *improvement* (learning happened) rather than a hard performance bar, so they
/// catch regressions in the training stack without depending on full convergence or downloads.
/// </summary>
public class ShowcaseSmokeTests
{
    private readonly ITestOutputHelper _out;
    public ShowcaseSmokeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void Ppo_LearningSignal_SinglePole()
    {
        const int maxSteps = 150;
        Init.Seed(1);
        var rng = new Random(1);

        var ac = new ActorCritic(stateSize: 4, actionSize: 1, hidden: 32);
        var ppo = new ContinuousPpo(ac, () => new SinglePoleCart(rng, maxSteps), rng)
        {
            NumEnvs = 8, Horizon = 64, Epochs = 3, MinibatchSize = 256, LearningRate = 3e-3f,
        };

        // Baseline before any training, then a handful of cheap iterations.
        float before = ppo.EvaluateMeanSteps(episodes: 3, maxSteps: maxSteps);
        ppo.Train(5, (i, ret) => _out.WriteLine($"iter {i}: meanReturn={ret:0.0}"));
        float after = ppo.EvaluateMeanSteps(episodes: 3, maxSteps: maxSteps);

        _out.WriteLine($"meanSteps before={before:0.0}, after={after:0.0} (/{maxSteps})");
        // A few iterations on the easy single pole must measurably improve balancing.
        // Margin is generous so the smoke is robust to seed/scheduler noise.
        Assert.True(after > before + 10f,
            $"PPO showed no learning signal (before={before:0.0}, after={after:0.0}).");
    }

    [Fact]
    public void Ppo_ValueTargetNormalization_KeepsCriticBounded()
    {
        const int maxSteps = 150;
        Init.Seed(2);
        var rng = new Random(2);

        var ac = new ActorCritic(stateSize: 4, actionSize: 1, hidden: 32);
        var ppo = new ContinuousPpo(ac, () => new SinglePoleCart(rng, maxSteps), rng)
        {
            NumEnvs = 8, Horizon = 64, Epochs = 3, MinibatchSize = 256, LearningRate = 3e-3f,
        };

        float maxLoss = 0f;
        int totalSkipped = 0;
        ppo.Train(20, (i, ret) =>
        {
            if (float.IsFinite(ppo.LastLoss)) maxLoss = MathF.Max(maxLoss, ppo.LastLoss);
            totalSkipped += ppo.LastSkippedUpdates;
            if (i % 5 == 0)
                _out.WriteLine($"iter {i,2}: return={ret:0.0} loss={ppo.LastLoss:0.000} σ_ret={ppo.ReturnScale:0.00} skipped={ppo.LastSkippedUpdates}");
        });

        _out.WriteLine($"maxLoss={maxLoss:0.000}, ReturnScale={ppo.ReturnScale:0.00}, totalSkipped={totalSkipped}");

        // The value head regresses onto ret/σ, so σ_ret calibrates to the real return scale (rewards are
        // +1/step → RMS ≫ 1). Without this normalization σ stays exactly 1.0 and the raw-target MSE is
        // free to random-walk into the 1e8 range; with it the (unit-scale) value targets keep the total
        // loss O(1). Both assertions fail if the normalization is removed.
        Assert.True(ppo.ReturnScale > 2f, $"return-scale normalization did not engage (σ_ret={ppo.ReturnScale}).");
        Assert.True(maxLoss < 25f, $"value loss not bounded by normalization (maxLoss={maxLoss}).");
        // A healthy run never trips the non-finite crash guard — a non-zero count means a minibatch went NaN.
        Assert.Equal(0, totalSkipped);
    }

    [Fact]
    public void Cnn_LossDecreases_OnSyntheticImages()
    {
        Init.Seed(0);
        var rng = new Random(0);

        // Tiny 2-class synthetic image set: class 0 = bright top half, class 1 = bright bottom
        // half, plus noise. Linearly trivial for a human, but it forces the conv/pool/linear
        // stack and its gradients to actually do work. No download, runs in seconds.
        const int n = 48, side = 12, area = side * side;
        var images = new float[n * area];
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            int cls = i % 2;
            labels[i] = cls;
            for (int r = 0; r < side; r++)
                for (int c = 0; c < side; c++)
                {
                    bool topHalf = r < side / 2;
                    float baseVal = (cls == 0 ? topHalf : !topHalf) ? 0.9f : 0.1f;
                    images[i * area + r * side + c] = Math.Clamp(baseVal + (float)(rng.NextDouble() - 0.5) * 0.2f, 0f, 1f);
                }
        }
        var x = Tensor.FromShaped(images, new[] { n, 1, side, side });

        var model = new Sequential(
            new Conv2d(1, 4, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Conv2d(4, 8, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Activation(t => TensorOps.Flatten(t, 1)),
            new Linear(8 * 3 * 3, 16), Activation.Relu(),
            new Linear(16, 2));
        var opt = new Adam(model.Parameters().ToList(), lr: 3e-3f);

        float firstLoss = 0f, lastLoss = 0f;
        const int steps = 12;
        for (int step = 0; step < steps; step++)
        {
            var logits = model.Forward(x);
            var loss = TensorOps.CrossEntropy(logits, labels);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            lastLoss = loss.ToArray()[0];
            if (step == 0) firstLoss = lastLoss;
            _out.WriteLine($"step {step}: loss={lastLoss:0.0000}");
        }

        _out.WriteLine($"loss first={firstLoss:0.0000}, last={lastLoss:0.0000}");
        // The conv/pool/linear stack must drive the loss clearly down on a trivially separable set.
        Assert.True(lastLoss < firstLoss * 0.6f,
            $"CNN loss did not decrease enough (first={firstLoss:0.0000}, last={lastLoss:0.0000}).");
    }
}
