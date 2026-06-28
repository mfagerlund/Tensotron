using Tensotron;
using Tensotron.Showcase.Mnist;
using Xunit.Abstractions;

namespace Tensotron.Showcase;

/// <summary>
/// "Tensotron can be used for vision": train a small CNN (Conv2d + MaxPool2d + Linear) on
/// MNIST and assert it reaches a real accuracy. Exercises the conv/pool stack end to end.
/// MNIST is downloaded+cached on first run; if it can't be fetched (offline) the test skips.
///
/// Tagged <c>Category=Showcase</c>: full-strength training, excluded from the normal suite
/// (very slow on the CPU accelerator). Run via <c>tools/run-tests.ps1 -Showcase</c> or
/// <c>dotnet test --filter "Category=Showcase"</c>. The always-on <see cref="ShowcaseSmokeTests"/>
/// exercise the conv/pool/autograd stack cheaply on every run.
/// </summary>
[Trait("Category", "Showcase")]
public class MnistTests
{
    private readonly ITestOutputHelper _out;
    public MnistTests(ITestOutputHelper output) => _out = output;

    // Subset sizes keep the showcase fast while still proving real learning.
    private const int TrainSubset = 4000;
    private const int TestSubset = 1000;
    private const int BatchSize = 64;
    private const int Epochs = 4;

    [SkippableFact]
    public void Cnn_LearnsMnistDigits()
    {
        Skip.IfNot(Cuda.IsAvailable(),
            "Showcase convergence runs on a CUDA GPU; none detected (CPU is too slow for full training).");
        _out.WriteLine($"Device: {Accelerators.Active().Name}");

        var loaded = MnistData.TryLoad(_out.WriteLine);
        Skip.If(loaded is null, "MNIST data unavailable (offline?).");
        var data = loaded!;

        Init.Seed(0);
        var model = new Sequential(
            new Conv2d(1, 8, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Conv2d(8, 16, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Activation(t => TensorOps.Flatten(t, 1)),
            new Linear(16 * 7 * 7, 64), Activation.Relu(),
            new Linear(64, 10));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);

        int nTrain = Math.Min(TrainSubset, data.TrainCount);
        int batches = nTrain / BatchSize;

        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            float lossSum = 0f;
            for (int b = 0; b < batches; b++)
            {
                var (x, y) = Batch(data.TrainImages, data.TrainLabels, b * BatchSize, BatchSize);
                var logits = model.Forward(x);
                var loss = TensorOps.CrossEntropy(logits, y);
                opt.ZeroGrad();
                loss.Backward();
                opt.Step();
                lossSum += loss.ToArray()[0];
            }
            float acc = Accuracy(model, data.TestImages, data.TestLabels, Math.Min(TestSubset, data.TestCount));
            _out.WriteLine($"epoch {epoch}: avgLoss={lossSum / batches:0.000}, testAcc={acc:0.000}");
        }

        float finalAcc = Accuracy(model, data.TestImages, data.TestLabels, Math.Min(TestSubset, data.TestCount));
        _out.WriteLine($"FINAL testAcc={finalAcc:0.000}");
        // 4 epochs on the 4k-image subset lands comfortably above this; 3 epochs reached 0.886
        // on the 4090, so the bar is set to be a real "it learned" signal without seed flakiness.
        Assert.True(finalAcc >= 0.88f, $"CNN failed to learn MNIST (testAcc={finalAcc}).");
    }

    private static (Tensor x, int[] y) Batch(float[] images, int[] labels, int start, int count)
    {
        var xs = new float[count * MnistData.ImageSize];
        Array.Copy(images, start * MnistData.ImageSize, xs, 0, count * MnistData.ImageSize);
        var ys = new int[count];
        Array.Copy(labels, start, ys, 0, count);
        return (Tensor.FromShaped(xs, new[] { count, 1, 28, 28 }), ys);
    }

    private static float Accuracy(Sequential model, float[] images, int[] labels, int count)
    {
        using (Tensor.NoGradScope())
        {
            int correct = 0;
            for (int start = 0; start < count; start += BatchSize)
            {
                int n = Math.Min(BatchSize, count - start);
                var (x, _) = Batch(images, labels, start, n);
                var logits = model.Forward(x).ToArray(); // (n,10) row-major
                for (int i = 0; i < n; i++)
                {
                    int best = 0;
                    for (int c = 1; c < 10; c++)
                        if (logits[i * 10 + c] > logits[i * 10 + best]) best = c;
                    if (best == labels[start + i]) correct++;
                }
            }
            return (float)correct / count;
        }
    }
}
