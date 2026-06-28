using Tensotron;

namespace Tensotron.Examples;

/// <summary>
/// The CS231n classic: K intertwined spirals, one class per arm. The classes wind around
/// each other so a linear boundary is hopeless — it forces the MLP to learn a genuinely
/// curved decision surface. Trains a small ReLU MLP, prints accuracy, and writes
/// <c>spiral.svg</c> showing the learned decision regions with the data overlaid.
/// </summary>
public static class SpiralExample
{
    private const int Classes = 3;
    private const int PerClass = 200;

    // Soft fills for the decision regions; saturated dots for the data.
    private static readonly string[] Fill = { "#f6c9c9", "#c9d9f6", "#c9f0cd" };
    private static readonly string[] Dot = { "#cc3333", "#3366cc", "#2e9e44" };

    public static void Run()
    {
        Console.WriteLine("== Spiral ==");
        Init.Seed(0);
        var rng = new Random(0);

        var (data, labels) = MakeSpiral(rng);
        var x = Tensor.FromArray(data, Classes * PerClass, 2);

        var model = new Sequential(
            new Linear(2, 64), Activation.Relu(),
            new Linear(64, 64), Activation.Relu(),
            new Linear(64, Classes));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-2f, weightDecay: 1e-4f);

        for (int epoch = 1; epoch <= 400; epoch++)
        {
            var loss = TensorOps.CrossEntropy(model.Forward(x), labels);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (epoch % 100 == 0)
                Console.WriteLine($"  epoch {epoch,4}: loss={loss.Item():0.0000}, acc={Accuracy(model, x, labels):0.000}");
        }

        Console.WriteLine($"  final accuracy: {Accuracy(model, x, labels):0.000}");
        RenderSvg(model, data, labels, "spiral.svg");
        Console.WriteLine();
    }

    // Two-feature spiral: each class is an arm sweeping radius 0..1 while the angle advances.
    private static (float[] data, int[] labels) MakeSpiral(Random rng)
    {
        var data = new float[Classes * PerClass * 2];
        var labels = new int[Classes * PerClass];
        int k = 0;
        for (int c = 0; c < Classes; c++)
            for (int i = 0; i < PerClass; i++)
            {
                float r = (float)i / PerClass;                                  // radius 0..1
                float t = c * 4f + r * 5f + (float)(rng.NextDouble() - 0.5) * 0.4f; // angle + noise
                data[k * 2] = r * MathF.Sin(t);
                data[k * 2 + 1] = r * MathF.Cos(t);
                labels[k] = c;
                k++;
            }
        return (data, labels);
    }

    private static float Accuracy(Sequential model, Tensor x, int[] labels)
    {
        using (Tensor.NoGradScope())
        {
            float[] logits = model.Forward(x).ToArray(); // (N, Classes)
            int correct = 0;
            for (int i = 0; i < labels.Length; i++)
                if (ArgMax(logits, i * Classes, Classes) == labels[i]) correct++;
            return (float)correct / labels.Length;
        }
    }

    private static int ArgMax(float[] a, int offset, int n)
    {
        int best = 0;
        for (int j = 1; j < n; j++) if (a[offset + j] > a[offset + best]) best = j;
        return best;
    }

    // Sweep a grid through the model in one batch, paint each cell by predicted class,
    // then drop the training points on top.
    private static void RenderSvg(Sequential model, float[] data, int[] labels, string file)
    {
        var plot = new Plot();
        const float lo = -1.2f, hi = 1.2f, step = 0.04f;
        int side = (int)MathF.Round((hi - lo) / step) + 1;

        var grid = new float[side * side * 2];
        int g = 0;
        for (int iy = 0; iy < side; iy++)
            for (int ix = 0; ix < side; ix++)
            {
                grid[g * 2] = lo + ix * step;
                grid[g * 2 + 1] = lo + iy * step;
                g++;
            }

        using (Tensor.NoGradScope())
        {
            float[] logits = model.Forward(Tensor.FromArray(grid, side * side, 2)).ToArray();
            for (int i = 0; i < side * side; i++)
                plot.FilledCell(grid[i * 2], grid[i * 2 + 1], step / 2f, Fill[ArgMax(logits, i * Classes, Classes)]);
        }

        for (int i = 0; i < labels.Length; i++)
            plot.Dot(data[i * 2], data[i * 2 + 1], 0.012f, Dot[labels[i]]);

        plot.Save(file);
    }
}
