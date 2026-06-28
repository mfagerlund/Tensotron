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

        // Headline: the default MLP, measured the way training actually runs (async loop,
        // single host sync at the end). Plus per-step launch/alloc/host-upload counts and a
        // serialized phase attribution (fwd/bwd/opt) so we can see WHERE the time goes.
        var r = Measure(batch: 256, inDim: 32, width: 128, depth: 2, steps: 300, warmup: 30, attribute: true);
        Console.WriteLine($"  MLP {r.InDim}->{r.Width}x{r.Depth}->1, batch {r.Batch}:");
        Console.WriteLine($"    {r.Timed} steps in {r.TotalMs:0} ms  =>  {r.MsPerStep:0.00} ms/step");
        Console.WriteLine($"    per step: {r.LaunchesPerStep:0.0} launches, {r.AllocsPerStep:0.0} device allocs, {r.HostUploadsPerStep:0.0} host uploads");
        Console.WriteLine($"    attribution (serialized, sums > async total): " +
                          $"fwd {r.FwdMs:0.00}  bwd {r.BwdMs:0.00}  opt {r.OptMs:0.00} ms/step");
        Console.WriteLine();
    }

    /// <summary>Sweep batch size and width to expose the overhead-bound → compute-bound crossover.</summary>
    public static void Sweep()
    {
        Console.WriteLine("== Bench sweep (ms/step) ==");
        Console.WriteLine("  batch  width  depth | ms/step | laun/step alloc/step");
        int[] batches = { 64, 256, 1024, 4096 };
        int[] widths = { 128, 512, 2048 };
        foreach (var w in widths)
            foreach (var b in batches)
            {
                int steps = w >= 2048 || b >= 4096 ? 120 : 250;
                var r = Measure(batch: b, inDim: 64, width: w, depth: 2, steps: steps, warmup: 20, attribute: false);
                Console.WriteLine($"  {b,5}  {w,5}  {r.Depth,5} | {r.MsPerStep,7:0.00} | {r.LaunchesPerStep,9:0.0} {r.AllocsPerStep,10:0.0}");
            }
        Console.WriteLine();
    }

    public readonly record struct Result(
        int Batch, int InDim, int Width, int Depth, int Timed,
        double TotalMs, double MsPerStep,
        double FwdMs, double BwdMs, double OptMs,
        double LaunchesPerStep, double AllocsPerStep, double HostUploadsPerStep);

    private static Result Measure(int batch, int inDim, int width, int depth, int steps, int warmup, bool attribute)
    {
        Init.Seed(0);
        var rng = new Random(0);

        var data = new float[batch * inDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var x = Tensor.FromArray(data, batch, inDim);
        var target = Tensor.FromArray(new float[batch], batch, 1);

        var layers = new List<Module>();
        int d = inDim;
        for (int l = 0; l < depth; l++) { layers.Add(new Linear(d, width)); layers.Add(Activation.Relu()); d = width; }
        layers.Add(new Linear(d, 1));
        var model = new Sequential(layers.ToArray());
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);

        var sw = new Stopwatch();
        long launches0 = 0, allocs0 = 0, uploads0 = 0;
        var rt = TensorRuntime.Instance;

        for (int step = 0; step < steps; step++)
        {
            if (step == warmup)
            {
                rt.ResetCounters();
                launches0 = rt.Launches; allocs0 = rt.Allocs; uploads0 = rt.HostUploads;
                sw.Start();
            }
            var loss = TensorOps.MseLoss(model.Forward(x), target);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (step == steps - 1) _ = loss.Item(); // force a final host sync so timing is honest
        }
        sw.Stop();
        int timed = steps - warmup;
        double total = sw.Elapsed.TotalMilliseconds;
        double launchesPer = (double)(rt.Launches - launches0) / timed;
        double allocsPer = (double)(rt.Allocs - allocs0) / timed;
        double uploadsPer = (double)(rt.HostUploads - uploads0) / timed;

        // Serialized phase attribution: sync after each phase so the stopwatch captures GPU time
        // for that phase. This serializes (defeats async batching), so the per-phase sum is HIGHER
        // than the async headline above — it is for attribution only, not a headline number.
        double fwd = 0, bwd = 0, optMs = 0;
        if (attribute)
        {
            const int aSteps = 60, aWarm = 10;
            var swp = new Stopwatch();
            for (int step = 0; step < aSteps; step++)
            {
                bool timeIt = step >= aWarm;
                swp.Restart(); var loss = TensorOps.MseLoss(model.Forward(x), target); rt.Sync(); swp.Stop();
                if (timeIt) fwd += swp.Elapsed.TotalMilliseconds;
                opt.ZeroGrad();
                swp.Restart(); loss.Backward(); rt.Sync(); swp.Stop();
                if (timeIt) bwd += swp.Elapsed.TotalMilliseconds;
                swp.Restart(); opt.Step(); rt.Sync(); swp.Stop();
                if (timeIt) optMs += swp.Elapsed.TotalMilliseconds;
            }
            int an = aSteps - aWarm;
            fwd /= an; bwd /= an; optMs /= an;
        }

        return new Result(batch, inDim, width, depth, timed, total, total / timed,
            fwd, bwd, optMs, launchesPer, allocsPer, uploadsPer);
    }
}
