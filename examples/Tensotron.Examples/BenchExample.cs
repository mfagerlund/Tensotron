using System.Diagnostics;
using Tensotron;

namespace Tensotron.Examples;

/// <summary>
/// Micro-benchmark for the training hot path: a fixed MLP run for many steps, reporting
/// ms/step. Most of the cost in a small net is per-op launch + host-sync overhead rather
/// than arithmetic, so this is a sensitive probe for runtime changes (async batching,
/// device-side parameter updates, allocation churn). Not a test — just a stopwatch.
/// </summary>
public static class BenchExample
{
    public static void Run()
    {
        Console.WriteLine("== Bench (training hot path) ==");
        Init.Seed(0);
        var rng = new Random(0);

        const int batch = 256, inDim = 32, steps = 300, warmup = 30;
        var data = new float[batch * inDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var x = Tensor.FromArray(data, batch, inDim);
        var target = Tensor.FromArray(new float[batch], batch, 1);

        var model = new Sequential(
            new Linear(inDim, 128), Activation.Relu(),
            new Linear(128, 128), Activation.Relu(),
            new Linear(128, 1));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);

        var sw = new Stopwatch();
        for (int step = 0; step < steps; step++)
        {
            if (step == warmup) sw.Start(); // exclude JIT / kernel-compile warmup
            var loss = TensorOps.MseLoss(model.Forward(x), target);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (step == steps - 1) _ = loss.Item(); // force a final host sync so timing is honest
        }
        sw.Stop();

        int timed = steps - warmup;
        Console.WriteLine($"  {timed} steps in {sw.ElapsedMilliseconds} ms  =>  {sw.Elapsed.TotalMilliseconds / timed:0.00} ms/step");
        Console.WriteLine();
    }
}
