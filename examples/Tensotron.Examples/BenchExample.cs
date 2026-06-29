using System.Diagnostics;
using System.Globalization;
using System.Threading;
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

    /// <summary>Caching-allocator probe: heavy (large-activation) configs run with the per-step
    /// graph recycled (Tensor.DisposeGraph → pooled buffers) vs not, exposing the cudaMalloc cliff.</summary>
    public static void Pool()
    {
        Console.WriteLine("== Bench: caching allocator on large activations ==");
        Console.WriteLine("  config            | pool-off (no recycle)     | pool-on (DisposeGraph)");
        (int b, int w)[] configs = { (1024, 2048), (4096, 2048) };
        foreach (var (b, w) in configs)
        {
            var off = Measure(b, 64, w, 2, 80, 20, attribute: false, freeGraph: false);
            var on = Measure(b, 64, w, 2, 80, 20, attribute: false, freeGraph: true);
            Console.WriteLine($"  batch {b,4} width {w,4} | {off.MsPerStep,8:0.0} ms {off.AllocsPerStep,5:0} alloc/s | " +
                              $"{on.MsPerStep,8:0.0} ms {on.AllocsPerStep,5:0} alloc/s  ({off.MsPerStep / on.MsPerStep:0.0}× )");
        }
        Console.WriteLine();
    }

    /// <summary>
    /// Cross-library ladder vs PyTorch. Emits machine-readable <c>RESULT …</c> lines (identical
    /// schema to tools/bench/torch_bench.py) that get merged into docs/PERFORMANCE_VS_PYTORCH.md.
    /// Tensotron recycles each step's graph (DisposeGraph) so the caching allocator is in steady
    /// state — the fair analogue to PyTorch's always-on caching allocator.
    /// </summary>
    public static void Ladder()
    {
        // RESULT lines must use '.' decimals so they merge with the PyTorch side and parse cleanly.
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.WriteLine("== Ladder: Tensotron (ms/step, fwd+bwd+Adam; GEMM is fwd-only) ==");

        // Rung 1 — MLP hot path (overhead-bound; our recent work targets exactly this).
        foreach (var (b, inD, w, d) in new[] { (256, 32, 128, 2), (1024, 128, 512, 2) })
        {
            var r = Measure(batch: b, inDim: inD, width: w, depth: d, steps: 300, warmup: 30,
                            attribute: false, freeGraph: true);
            Console.WriteLine($"RESULT mlp config=tensotron batch={b} in={inD} width={w} depth={d} " +
                              $"ms_per_step={r.MsPerStep:0.000} launches={r.LaunchesPerStep:0.0} allocs={r.AllocsPerStep:0.0}");
        }

        // Rung 2 — MNIST-CNN (the showcase arch; stresses conv + the maxpool host-sync).
        foreach (var b in new[] { 64, 256 })
        {
            var c = MeasureCnn(batch: b, steps: 150, warmup: 20);
            Console.WriteLine($"RESULT cnn config=tensotron batch={b} " +
                              $"ms_per_step={c.ms:0.000} launches={c.launches:0.0} allocs={c.allocs:0.0}");
        }

        // Rung 3 — large GEMM forward (compute-bound; exercises the cuBLAS path).
        foreach (var n in new[] { 1024, 2048, 4096 })
        {
            var g = MeasureGemm(n, n, n, iters: 200, warmup: 30);
            Console.WriteLine($"RESULT gemm config=tensotron m={n} n={n} k={n} ms={g.ms:0.000} gflops={g.gflops:0.0}");
        }
        Console.WriteLine();
    }

    /// <summary>TF32 probe: same forward GEMM under cuBLAS math modes FP32 / TensorOp / TF32, so we
    /// see which (if any) engages the tensor cores on this cuBLAS build. Storage stays FP32 throughout.</summary>
    public static void GemmTf32()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.WriteLine("== GEMM math-mode probe (fwd-only; FP32 vs TensorOp vs TF32) ==");
        var rt = TensorRuntime.Instance;
        foreach (var (tag, mode) in new[] { ("fp32", 0), ("tensorop", 1), ("tf32", 3) })
        {
            rt.SetCuBlasMathMode(mode);
            foreach (var n in new[] { 1024, 2048, 4096 })
            {
                var g = MeasureGemm(n, n, n, iters: 200, warmup: 30);
                Console.WriteLine($"RESULT gemm config=tensotron-{tag} m={n} n={n} k={n} ms={g.ms:0.000} gflops={g.gflops:0.0}");
            }
        }
        rt.SetCuBlasMathMode(0);
        Console.WriteLine();
    }

    /// <summary>
    /// Trace/replay vs eager for a PPO-scale MLP training step. Captures one fixed-shape step
    /// (forward + backward + clip + Adam) and replays it buffer-to-buffer with no host-side autograd
    /// graph rebuild, vs the eager loop that rebuilds the graph every step. The step is ~95% host-bound
    /// (see StepBreakdown), so this is where the small-model speedup lives.
    /// </summary>
    public static void ReplayBench()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        var rt = TensorRuntime.Instance;
        Console.WriteLine("== Trace replay vs eager (PPO-scale MLP train step) ==");
        Console.WriteLine($"  device: {rt.DeviceName}");

        Init.Seed(0);
        var rng = new Random(0);
        const int batch = 512, inDim = 8, width = 64, outDim = 2;
        var data = new float[batch * inDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var input = Tensor.FromArray(data, batch, inDim);
        var target = Tensor.FromArray(new float[batch * outDim], batch, outDim);

        var model = new Sequential(new Linear(inDim, width), Activation.Tanh(),
                                   new Linear(width, width), Activation.Tanh(),
                                   new Linear(width, outDim));
        var ps = model.Parameters().ToList();
        var opt = new Adam(ps, lr: 1e-3f);

        Tensor StepBody()
        {
            var loss = TensorOps.MseLoss(model.Forward(input), target);
            opt.ZeroGrad();
            loss.Backward();
            GradUtils.ClipGradNorm(ps, 0.5f, returnTotalNorm: false);
            opt.Step();
            return loss;
        }

        double toUs = 1e6 / Stopwatch.Frequency;
        const int warm = 100, iters = 1000;
        var sw = new Stopwatch();

        // Eager: rebuild the graph every step (recycle activations so the allocator is steady).
        for (int i = 0; i < warm; i++) { var l = StepBody(); l.DisposeGraph(); }
        rt.Sync();
        sw.Restart();
        for (int i = 0; i < iters; i++) { var l = StepBody(); l.DisposeGraph(); }
        rt.Sync();
        sw.Stop();
        double eagerUs = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

        // Capture once (advances opt past warmup so frozen Adam bias-correction ≈ 1), then replay.
        var graph = rt.Capture(StepBody);
        for (int i = 0; i < warm; i++) graph.Replay();
        rt.Sync();
        sw.Restart();
        for (int i = 0; i < iters; i++) graph.Replay();
        rt.Sync();
        sw.Stop();
        double replayUs = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

        Console.WriteLine($"  {graph.LaunchCount} launches/step (replay re-fires these, no graph rebuild)");
        Console.WriteLine($"  eager  : {eagerUs,8:0.0} us/step");
        Console.WriteLine($"  replay : {replayUs,8:0.0} us/step");
        Console.WriteLine($"  speedup: {eagerUs / replayUs,8:0.00}x");
        Console.WriteLine();
    }

    /// <summary>
    /// Decider probe for "what gives the best small-model boost": splits one PPO-scale training
    /// step into host-dispatch wall (the host thread building the autograd graph + dispatching
    /// kernels) vs device-tail (GPU drain after the last op is issued). Per step we drain, then
    /// issue forward+loss+backward+clip+Adam with NO host pull (clip uses returnTotalNorm:false),
    /// timestamp the host return, then Sync and timestamp the device drain. If host-dispatch
    /// dominates, the lever is trace/replay (kill the per-step C# graph rebuild); if the device
    /// tail dominates, it's kernel fusion (fewer/bigger launches). The raw single-op dispatch
    /// floor (one cached elementwise launch in a tight loop) estimates how much of the per-launch
    /// host cost is ILGPU dispatch (which only fusion removes) vs C# graph build (which
    /// trace/replay removes).
    /// </summary>
    public static void StepBreakdown()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        var rt = TensorRuntime.Instance;
        Console.WriteLine("== Step breakdown: host-dispatch wall vs device-tail (PPO-scale MLP) ==");
        Console.WriteLine($"  device: {rt.DeviceName}");

        Init.Seed(0);
        var rng = new Random(0);
        const int batch = 512, inDim = 8, width = 64, outDim = 2;
        var data = new float[batch * inDim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        var x = Tensor.FromArray(data, batch, inDim);
        var target = Tensor.FromArray(new float[batch * outDim], batch, outDim);

        var model = new Sequential(new Linear(inDim, width), Activation.Tanh(),
                                   new Linear(width, width), Activation.Tanh(),
                                   new Linear(width, outDim));
        var ps = model.Parameters().ToList();
        var opt = new Adam(ps, lr: 1e-3f);

        void Step()
        {
            var loss = TensorOps.MseLoss(model.Forward(x), target);
            opt.ZeroGrad();
            loss.Backward();
            GradUtils.ClipGradNorm(ps, 0.5f, returnTotalNorm: false);
            opt.Step();
            loss.DisposeGraph();
        }

        double toUs = 1e6 / Stopwatch.Frequency;
        const int warm = 100, iters = 500;
        for (int i = 0; i < warm; i++) Step();
        rt.Sync();
        long launch0 = rt.Launches;

        double hostUs = 0, devUs = 0;
        for (int i = 0; i < iters; i++)
        {
            rt.Sync();
            long tA = Stopwatch.GetTimestamp();
            Step();
            long tB = Stopwatch.GetTimestamp();
            rt.Sync();
            long tC = Stopwatch.GetTimestamp();
            hostUs += (tB - tA) * toUs;
            devUs += (tC - tB) * toUs;
        }
        long launchDelta = rt.Launches - launch0;
        double launchesPer = (double)launchDelta / iters;
        hostUs /= iters; devUs /= iters;
        double total = hostUs + devUs;

        // Issue-only host throughput of a single elementwise op, NoGrad (no GradNode, no backward
        // graph) vs WITH grad (GradNode + retained graph). Issue N ops back-to-back, sync ONCE at
        // the end, divide the issue wall by N — so this is pure host cost per op (the device runs
        // behind). The gap between the two isolates the autograd-graph-build cost (what trace/
        // replay removes) from the Tensor-alloc + ILGPU-dispatch cost (what only fusion removes).
        using var probe = Tensor.FromArray(new float[batch * width], batch, width);
        double NoGradOpUs(int n)
        {
            using (Tensor.NoGradScope())
            {
                for (int i = 0; i < n / 4; i++) { using var w = probe.Relu(); }   // warm
                rt.Sync();
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < n; i++) { using var r = probe.Relu(); }
                long t1 = Stopwatch.GetTimestamp();   // host issue wall (device runs behind)
                rt.Sync();
                return (t1 - t0) * toUs / n;
            }
        }
        double GradOpUs(int n)
        {
            var leaf = probe.Detach().RequireGrad();
            for (int i = 0; i < n / 4; i++) { _ = leaf.Relu(); }   // warm
            rt.Sync();
            long t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < n; i++) { _ = leaf.Relu(); }       // builds GradNode + retains graph
            long t1 = Stopwatch.GetTimestamp();
            rt.Sync();
            return (t1 - t0) * toUs / n;
        }
        double noGradUs = NoGradOpUs(20000);
        double gradUs = GradOpUs(20000);

        Console.WriteLine($"  {launchesPer:0} launches/step ({launchDelta} over {iters} steps)");
        Console.WriteLine($"  host-dispatch wall : {hostUs,8:0.0} us/step  ({100 * hostUs / total:0}%)");
        Console.WriteLine($"  device-tail (drain): {devUs,8:0.0} us/step  ({100 * devUs / total:0}%)");
        Console.WriteLine($"  total              : {total,8:0.0} us/step");
        Console.WriteLine($"  per-launch host    : {hostUs / launchesPer,8:0.0} us/launch");
        Console.WriteLine($"  1 op issue, NoGrad : {noGradUs,8:0.0} us  (Tensor alloc + ILGPU dispatch only)");
        Console.WriteLine($"  1 op issue, w/grad : {gradUs,8:0.0} us  (+ GradNode + retained graph)");
        Console.WriteLine($"  autograd build tax : {gradUs - noGradUs,8:0.0} us/op  (what trace/replay removes)");
        Console.WriteLine();
    }

    /// <summary>
    /// Inference latency for a small PPO-style policy net (8→64→64→2, tanh), forward-only / NoGrad,
    /// the realistic deployment path: obs float[] in → action float[] out (includes input marshaling
    /// and output readback). Compared against a faithful hand-rolled *scalar* C# forward using the
    /// SAME weights — so we both validate equivalence (max abs diff) and see the per-op overhead
    /// multiple. Run under TENSOTRON_BACKEND=cpu vs =cuda to compare backends. This is the number
    /// that decides whether ILGPU's (scalar) CPU path is acceptable for a control net on the CPU.
    /// </summary>
    public static void InferenceLatency()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        const int obs = 8, h1n = 64, h2n = 64, act = 2;
        Console.WriteLine($"== Inference latency (policy {obs}->{h1n}->{h2n}->{act}, tanh, NoGrad) ==");
        Console.WriteLine($"  device: {TensorRuntime.Instance.DeviceName}");

        Init.Seed(0);
        var l1 = new Linear(obs, h1n);
        var l2 = new Linear(h1n, h2n);
        var l3 = new Linear(h2n, act);
        var model = new Sequential(l1, Activation.Tanh(), l2, Activation.Tanh(), l3);

        // Pull weights for the hand-rolled baseline (Linear.Weight is (out,in) row-major, Bias is (out,)).
        float[] w1 = l1.Weight.ToArray(), b1 = l1.Bias!.ToArray();
        float[] w2 = l2.Weight.ToArray(), b2 = l2.Bias!.ToArray();
        float[] w3 = l3.Weight.ToArray(), b3 = l3.Bias!.ToArray();

        var rt = TensorRuntime.Instance;
        var rng = new Random(1);

        foreach (var B in new[] { 1, 8, 64 })
        {
          try
          {
            var xdata = new float[B * obs];
            for (int i = 0; i < xdata.Length; i++) xdata[i] = (float)(rng.NextDouble() - 0.5);

            // Validate the hand-rolled baseline matches Tensotron (sample 0 of the batch).
            float[] tOut;
            using (Tensor.NoGradScope())
            {
                using var xt = Tensor.FromArray(xdata, B, obs);
                using var ot = model.Forward(xt);
                tOut = ot.ToArray();
            }
            var hOut0 = HandFwd(xdata, 0, obs, w1, b1, h1n, w2, b2, h2n, w3, b3, act);
            float maxDiff = 0;
            for (int a = 0; a < act; a++) maxDiff = MathF.Max(maxDiff, MathF.Abs(tOut[a] - hOut0[a]));

            int iters = B == 1 ? 20000 : B == 8 ? 10000 : 4000;
            int warm = iters / 10;

            // Tensotron — full realistic path: marshal obs in, forward, read action out.
            var sw = new Stopwatch();
            using (Tensor.NoGradScope())
            {
                for (int i = 0; i < iters + warm; i++)
                {
                    if (i == warm) { rt.Sync(); sw.Restart(); }
                    using var xt = Tensor.FromArray(xdata, B, obs);
                    using var ot = model.Forward(xt);
                    _ = ot.ToArray();
                }
                rt.Sync();
            }
            sw.Stop();
            double tUs = sw.Elapsed.TotalMilliseconds * 1000.0 / iters;

            // Hand-rolled scalar C# — same weights, no library machinery.
            var sw2 = new Stopwatch();
            var sink = new float[act];
            for (int i = 0; i < iters + warm; i++)
            {
                if (i == warm) sw2.Restart();
                for (int b = 0; b < B; b++)
                    sink = HandFwd(xdata, b, obs, w1, b1, h1n, w2, b2, h2n, w3, b3, act);
            }
            sw2.Stop();
            double hUs = sw2.Elapsed.TotalMilliseconds * 1000.0 / iters;
            GC.KeepAlive(sink);

            Console.WriteLine($"  batch={B,-3} tensotron {tUs,8:0.00} µs/fwd | hand-scalar {hUs,7:0.00} µs/fwd | " +
                              $"{tUs / hUs,6:0.0}x  (validate maxdiff {maxDiff:0.0e+0})");
          }
          catch (Exception e)
          {
            Console.WriteLine($"  batch={B,-3} FAILED on this backend: {e.GetType().Name}: {e.Message}");
          }
        }
        Console.WriteLine();
    }

    // Faithful scalar forward of the policy net for sample `s` of a (B,obs) row-major batch.
    private static float[] HandFwd(float[] x, int s, int obs,
        float[] w1, float[] b1, int h1n, float[] w2, float[] b2, int h2n, float[] w3, float[] b3, int act)
    {
        int xoff = s * obs;
        var a1 = new float[h1n];
        for (int o = 0; o < h1n; o++)
        {
            float acc = b1[o]; int wo = o * obs;
            for (int i = 0; i < obs; i++) acc += x[xoff + i] * w1[wo + i];
            a1[o] = MathF.Tanh(acc);
        }
        var a2 = new float[h2n];
        for (int o = 0; o < h2n; o++)
        {
            float acc = b2[o]; int wo = o * h1n;
            for (int i = 0; i < h1n; i++) acc += a1[i] * w2[wo + i];
            a2[o] = MathF.Tanh(acc);
        }
        var y = new float[act];
        for (int o = 0; o < act; o++)
        {
            float acc = b3[o]; int wo = o * h2n;
            for (int i = 0; i < h2n; i++) acc += a2[i] * w3[wo + i];
            y[o] = acc;
        }
        return y;
    }

    /// <summary>Where conv time actually goes: im2col vs the matmul, for both MNIST conv layers.</summary>
    public static void ConvBreakdown()
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.WriteLine("== Conv breakdown (MNIST conv layers, batch 64; fwd only, no-grad) ==");
        var rt = TensorRuntime.Instance;
        (int C, int H, int O)[] layers = { (1, 28, 8), (8, 14, 16) };
        foreach (var (C, H, O) in layers)
        {
            var rng = new Random(0);
            var data = new float[64 * C * H * H];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
            var x = Tensor.FromShaped(data, new[] { 64, C, H, H });
            var conv = new Conv2d(C, O, kernelSize: 3, padding: 1);
            double im2col, full;
            using (Tensor.NoGradScope())
            {
                im2col = Time(() => { TensorOps.Im2Col(x, 3, 3, 1, 1, 1, 1, 1, 1).Dispose(); }, rt);
                full = Time(() => { conv.Forward(x).Dispose(); }, rt);
            }
            // col2im = Im2Col backward (atomic scatter). Time a fwd+bwd over im2col alone.
            var xg = Tensor.FromShaped(data, new[] { 64, C, H, H }).RequireGrad();
            var seed = Tensor.Ones(new Shape(64, C * 9, H * H));
            double col2im = Time(() =>
            {
                xg.ZeroGrad();
                var col = TensorOps.Im2Col(xg, 3, 3, 1, 1, 1, 1, 1, 1);
                col.Backward(seed);
                col.DisposeGraph();
            }, rt) - im2col;
            // Full Conv2d fwd+bwd, gradient seeded directly on the output (NO scalar-reduce loss,
            // which would itself be a slow naive full reduction and mask the conv). A/B bias to
            // isolate the bias-gradient reduction (few output threads each looping batch·H·W).
            var xg2 = Tensor.FromShaped(data, new[] { 64, C, H, H }).RequireGrad();
            var convB = new Conv2d(C, O, kernelSize: 3, padding: 1);
            var convNoB = new Conv2d(C, O, kernelSize: 3, padding: 1, bias: false);
            var outSeed = Tensor.Ones(new Shape(64, O, H, H));
            double withBias = Time(() =>
            {
                var y = convB.Forward(xg2); y.Backward(outSeed); y.DisposeGraph(); xg2.ZeroGrad();
            }, rt, 60, 15);
            double noBias = Time(() =>
            {
                var y = convNoB.Forward(xg2); y.Backward(outSeed); y.DisposeGraph(); xg2.ZeroGrad();
            }, rt, 60, 15);
            Console.WriteLine($"  conv {C}->{O} {H}x{H}: im2col {im2col:0.000} | matmul+bias≈{full - im2col:0.000} | " +
                              $"col2im {col2im:0.000} | fwd+bwd: bias {withBias:0.000} ms vs no-bias {noBias:0.000} ms");
        }
        Console.WriteLine();
    }

    private static double Time(Action f, TensorRuntime rt, int iters = 100, int warm = 20)
    {
        var sw = new Stopwatch();
        for (int i = 0; i < iters + warm; i++) { if (i == warm) { rt.Sync(); sw.Start(); } f(); }
        rt.Sync(); sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    /// <summary>Serialized fwd/bwd/opt attribution for the MNIST CNN — finds where the step goes.</summary>
    public static void CnnPhases(int batch = 64)
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        Console.WriteLine($"== CNN phase attribution (batch {batch}, serialized) ==");
        Init.Seed(0);
        var rng = new Random(0);
        var data = new float[batch * 1 * 28 * 28];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        var x = Tensor.FromShaped(data, new[] { batch, 1, 28, 28 });
        var y = new int[batch];
        for (int i = 0; i < batch; i++) y[i] = rng.Next(10);
        var model = new Sequential(
            new Conv2d(1, 8, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Conv2d(8, 16, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Activation(t => TensorOps.Flatten(t, 1)),
            new Linear(16 * 7 * 7, 64), Activation.Relu(),
            new Linear(64, 10));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);
        var rt = TensorRuntime.Instance;
        var sw = new Stopwatch();
        double fwd = 0, bwd = 0, optMs = 0; const int steps = 80, warm = 20;
        for (int s = 0; s < steps; s++)
        {
            bool t = s >= warm;
            sw.Restart(); var loss = TensorOps.CrossEntropy(model.Forward(x), y); rt.Sync(); sw.Stop();
            if (t) fwd += sw.Elapsed.TotalMilliseconds;
            opt.ZeroGrad();
            sw.Restart(); loss.Backward(); rt.Sync(); sw.Stop();
            if (t) bwd += sw.Elapsed.TotalMilliseconds;
            sw.Restart(); opt.Step(); rt.Sync(); sw.Stop();
            if (t) optMs += sw.Elapsed.TotalMilliseconds;
            loss.DisposeGraph();
        }
        int n = steps - warm;
        Console.WriteLine($"  full model: fwd {fwd / n:0.000} ms | bwd {bwd / n:0.000} ms | opt {optMs / n:0.000} ms | total {(fwd + bwd + optMs) / n:0.000} ms");

        // Bisect: backward of the conv stack alone (scalar loss = Sum), and the classifier alone.
        var convStack = new Sequential(
            new Conv2d(1, 8, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Conv2d(8, 16, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2));
        double convBwd = Time(() =>
        {
            var loss = TensorOps.Sum(convStack.Forward(x));
            loss.Backward(); loss.DisposeGraph();
        }, rt, 60, 15);

        var flat = new float[batch * 16 * 7 * 7];
        for (int i = 0; i < flat.Length; i++) flat[i] = (float)rng.NextDouble();
        var feats = Tensor.FromShaped(flat, new[] { batch, 16 * 7 * 7 });
        var clf = new Sequential(new Linear(16 * 7 * 7, 64), Activation.Relu(), new Linear(64, 10));
        double clfBwd = Time(() =>
        {
            var loss = TensorOps.CrossEntropy(clf.Forward(feats), y);
            loss.Backward(); loss.DisposeGraph();
        }, rt, 60, 15);
        Console.WriteLine($"  isolated bwd: conv-stack {convBwd:0.000} ms | classifier {clfBwd:0.000} ms");
        Console.WriteLine();
    }

    /// <summary>The showcase MNIST CNN, timed on random data (fwd+bwd+Adam, CrossEntropy).</summary>
    private static (double ms, double launches, double allocs) MeasureCnn(int batch, int steps, int warmup)
    {
        Init.Seed(0);
        var rng = new Random(0);
        var data = new float[batch * 1 * 28 * 28];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble();
        var x = Tensor.FromShaped(data, new[] { batch, 1, 28, 28 });
        var y = new int[batch];
        for (int i = 0; i < batch; i++) y[i] = rng.Next(10);

        var model = new Sequential(
            new Conv2d(1, 8, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Conv2d(8, 16, kernelSize: 3, padding: 1), Activation.Relu(), new MaxPool2d(2),
            new Activation(t => TensorOps.Flatten(t, 1)),
            new Linear(16 * 7 * 7, 64), Activation.Relu(),
            new Linear(64, 10));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);

        var rt = TensorRuntime.Instance;
        var sw = new Stopwatch();
        long launches0 = 0, allocs0 = 0;
        for (int step = 0; step < steps; step++)
        {
            if (step == warmup) { rt.ResetCounters(); launches0 = rt.Launches; allocs0 = rt.Allocs; sw.Start(); }
            var logits = model.Forward(x);
            var loss = TensorOps.CrossEntropy(logits, y);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (step == steps - 1) _ = loss.Item(); // honest final host sync
            loss.DisposeGraph();                      // recycle activations into the pool
        }
        sw.Stop();
        int timed = steps - warmup;
        return (sw.Elapsed.TotalMilliseconds / timed,
                (double)(rt.Launches - launches0) / timed,
                (double)(rt.Allocs - allocs0) / timed);
    }

    /// <summary>Forward GEMM C=A·B throughput (back-to-back, single final sync), recycling the
    /// output buffer each iter so the caching allocator is in steady state.</summary>
    private static (double ms, double gflops) MeasureGemm(int M, int N, int K, int iters, int warmup)
    {
        var rng = new Random(0);
        var aData = new float[M * K];
        var bData = new float[K * N];
        for (int i = 0; i < aData.Length; i++) aData[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bData.Length; i++) bData[i] = (float)(rng.NextDouble() - 0.5);
        var a = Tensor.FromArray(aData, M, K);
        var b = Tensor.FromArray(bData, K, N);

        var rt = TensorRuntime.Instance;
        var sw = new Stopwatch();
        using (Tensor.NoGradScope())
        {
            for (int it = 0; it < iters + warmup; it++)
            {
                if (it == warmup) { rt.Sync(); sw.Start(); }
                var c = TensorOps.MatMul(a, b);
                if (it == iters + warmup - 1) rt.Sync();
                c.Dispose(); // single-stream ordering makes pool reuse safe
            }
        }
        sw.Stop();
        double ms = sw.Elapsed.TotalMilliseconds / iters;
        double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
        a.Dispose(); b.Dispose();
        return (ms, gflops);
    }

    public readonly record struct Result(
        int Batch, int InDim, int Width, int Depth, int Timed,
        double TotalMs, double MsPerStep,
        double FwdMs, double BwdMs, double OptMs,
        double LaunchesPerStep, double AllocsPerStep, double HostUploadsPerStep);

    private static Result Measure(int batch, int inDim, int width, int depth, int steps, int warmup,
        bool attribute, bool freeGraph = false)
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
            if (freeGraph) loss.DisposeGraph();      // recycle this step's activations into the pool
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
