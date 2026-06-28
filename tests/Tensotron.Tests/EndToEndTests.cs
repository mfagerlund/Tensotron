using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

public class EndToEndTests
{
    // Full stack: Module/Linear/ReLU + Adam + cosine LR + MSE, trained on a fixed toy
    // regression. Proves forward -> loss -> backward -> optimizer.step actually learns.
    [Fact]
    public void TrainsMlpToFitNonlinearFunction()
    {
        const int n = 64, inDim = 4;
        var rng = new Random(2024);

        // Inputs in [-1,1]; target is a smooth nonlinear function of them.
        var xData = new float[n * inDim];
        var yData = new float[n];
        for (int i = 0; i < n; i++)
        {
            float a = (float)(rng.NextDouble() * 2 - 1);
            float b = (float)(rng.NextDouble() * 2 - 1);
            float c = (float)(rng.NextDouble() * 2 - 1);
            float d = (float)(rng.NextDouble() * 2 - 1);
            xData[i * inDim + 0] = a; xData[i * inDim + 1] = b;
            xData[i * inDim + 2] = c; xData[i * inDim + 3] = d;
            yData[i] = a * b + c * c - d;          // needs a hidden layer to fit
        }
        var x = Tensor.FromShaped(xData, new[] { n, inDim });
        var y = Tensor.FromShaped(yData, new[] { n, 1 });

        Init.Seed(123);
        var model = new Sequential(
            new Linear(inDim, 32), Activation.Relu(),
            new Linear(32, 32), Activation.Relu(),
            new Linear(32, 1));

        var opt = new Adam(model.Parameters().ToList(), lr: 0.02f);
        const int steps = 400;
        var sched = new CosineAnnealingLR(opt, tMax: steps, etaMin: 1e-4f);

        float Loss() => TensorOps.MseLoss(model.Forward(x), y).Item();
        float initial = Loss();

        for (int s = 0; s < steps; s++)
        {
            model.ZeroGrad();
            var loss = TensorOps.MseLoss(model.Forward(x), y);
            loss.Backward();
            opt.Step();
            sched.Step();
        }

        float final = Loss();

        // The network should drive the loss down by well over an order of magnitude,
        // and the cosine schedule should have wound the LR down to near eta_min.
        Assert.True(final < initial * 0.05f, $"loss did not converge: {initial} -> {final}");
        Assert.True(final < 0.05f, $"final loss too high: {final}");
        Assert.True(opt.LearningRate < 0.02f, $"LR not annealed: {opt.LearningRate}");
    }

    // Train, snapshot to disk, reload into a fresh model -> identical predictions.
    [Fact]
    public void TrainedModelSurvivesSaveLoad()
    {
        const int n = 32, inDim = 3;
        var rng = new Random(7);
        var xData = new float[n * inDim];
        var yData = new float[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < inDim; j++) xData[i * inDim + j] = (float)(rng.NextDouble() * 2 - 1);
            yData[i] = xData[i * inDim] - xData[i * inDim + 2];
        }
        var x = Tensor.FromShaped(xData, new[] { n, inDim });
        var y = Tensor.FromShaped(yData, new[] { n, 1 });

        Init.Seed(5);
        var model = new Sequential(new Linear(inDim, 16), Activation.Tanh(), new Linear(16, 1));
        var opt = new AdamW(model.Parameters().ToList(), lr: 0.01f);
        for (int s = 0; s < 100; s++)
        {
            model.ZeroGrad();
            TensorOps.MseLoss(model.Forward(x), y).Backward();
            opt.Step();
        }

        var before = model.Forward(x).ToArray();

        var path = Path.Combine(Path.GetTempPath(), $"tensotron_e2e_{Guid.NewGuid():N}.tns");
        try
        {
            Serialization.Save(model, path);
            Init.Seed(999);
            var reloaded = new Sequential(new Linear(inDim, 16), Activation.Tanh(), new Linear(16, 1));
            Serialization.Load(reloaded, path);
            var after = reloaded.Forward(x).ToArray();
            Assert.Equal(before, after);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
