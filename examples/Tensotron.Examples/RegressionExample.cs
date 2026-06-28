using Tensotron;

namespace Tensotron.Examples;

/// <summary>
/// 1D curve fitting — the regression counterpart to the classifiers. Fits noisy samples of
/// a sine with an MLP under <see cref="TensorOps.MseLoss"/> (the regression path: continuous
/// targets, no softmax). Writes <c>regression.svg</c> with the noisy data and the learned
/// curve drawn through it.
/// </summary>
public static class RegressionExample
{
    private const int N = 64;

    public static void Run()
    {
        Console.WriteLine("== Regression (sin) ==");
        Init.Seed(0);
        var rng = new Random(0);

        // Inputs x in [-3, 3]; targets y = sin(1.5x) + small noise.
        var xs = new float[N];
        var ys = new float[N];
        for (int i = 0; i < N; i++)
        {
            float xi = -3f + 6f * i / (N - 1);
            xs[i] = xi;
            ys[i] = MathF.Sin(1.5f * xi) + (float)(rng.NextDouble() - 0.5) * 0.2f;
        }
        var x = Tensor.FromArray(xs, N, 1);
        var target = Tensor.FromArray(ys, N, 1);

        // 1 -> 64 -> 64 -> 1, tanh hidden units (smooth, good for fitting smooth functions).
        var model = new Sequential(
            new Linear(1, 64), Activation.Tanh(),
            new Linear(64, 64), Activation.Tanh(),
            new Linear(64, 1));
        var opt = new Adam(model.Parameters().ToList(), lr: 5e-3f);

        for (int step = 1; step <= 3000; step++)
        {
            var loss = TensorOps.MseLoss(model.Forward(x), target);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (step % 500 == 0) Console.WriteLine($"  step {step,4}: mse={loss.Item():0.0000}");
        }

        RenderSvg(model, xs, ys, "regression.svg");
        Console.WriteLine();
    }

    private static void RenderSvg(Sequential model, float[] xs, float[] ys, string file)
    {
        var plot = new Plot();
        for (int i = 0; i < xs.Length; i++) plot.Dot(xs[i], ys[i], 0.04f, "#cc3333");

        // Dense sweep of the fitted function for a smooth curve.
        const int samples = 200;
        var gx = new float[samples];
        for (int i = 0; i < samples; i++) gx[i] = -3f + 6f * i / (samples - 1);

        using (Tensor.NoGradScope())
        {
            float[] gy = model.Forward(Tensor.FromArray(gx, samples, 1)).ToArray();
            var curve = new (float, float)[samples];
            for (int i = 0; i < samples; i++) curve[i] = (gx[i], gy[i]);
            plot.Curve(curve, "#3366cc", 0.03f);
        }

        plot.Save(file);
    }
}
