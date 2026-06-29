using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Mid-training checkpoint must capture EVERYTHING needed to resume identically: model parameters,
/// persistent buffers (BatchNorm running mean/var), optimizer state (Adam moments / SGD &amp; RMSprop
/// buffers + step), and LR-scheduler state (the epoch counter). Strategy: train a model, save, keep
/// training the original AND a freshly-reloaded copy for two more deterministic steps, then assert
/// they stayed equal. Any state that wasn't serialized makes the copy diverge and fails the test.
/// Run across several optimizer + scheduler combos so the round-trip isn't secretly Adam-specific.
/// </summary>
public class CheckpointResumeTests
{
    // One of every checkpointable state group:
    //   Linear weight/bias          -> parameters
    //   LayerNorm gamma/beta        -> parameters (inside a norm)
    //   BatchNorm1d running_mean/var-> BUFFERS (the bug this guards against)
    private static Sequential MakeModel() => new(
        new Linear(8, 16),
        new BatchNorm1d(16),
        Activation.Relu(),
        new LayerNorm(16),
        new Linear(16, 4));

    // Fixed, RNG-free batch: the ONLY way the kept model and the reloaded copy can diverge is
    // incompletely-restored state (no dropout, no shuffling, same inputs every step).
    private static (Tensor x, Tensor y) Batch()
    {
        var x = new float[16 * 8];
        for (int i = 0; i < x.Length; i++) x[i] = MathF.Sin(i * 0.123f) * 0.5f;
        var y = new float[16 * 4];
        for (int i = 0; i < y.Length; i++) y[i] = MathF.Cos(i * 0.077f) * 0.3f;
        return (Tensor.FromShaped(x, new[] { 16, 8 }), Tensor.FromShaped(y, new[] { 16, 4 }));
    }

    private static void TrainStep(Module m, Optimizer opt, LrScheduler sched, Tensor x, Tensor y)
    {
        opt.ZeroGrad();
        var loss = TensorOps.MseLoss(m.Forward(x), y);
        loss.Backward();
        opt.Step();
        sched.Step();
    }

    [Fact]
    public void Checkpoint_ResumesTraining_Identically_Adam_Cosine()
        => RunResumeTest(ps => new Adam(ps, lr: 0.1f), o => new CosineAnnealingLR(o, tMax: 20));

    // SGD momentum buffer + a step-decay schedule: a different optimizer-state shape and a
    // non-cosine scheduler, so the round-trip isn't secretly Adam/Cosine-specific.
    [Fact]
    public void Checkpoint_ResumesTraining_Identically_SgdMomentum_StepLR()
        => RunResumeTest(ps => new Sgd(ps, lr: 0.02f, momentum: 0.9f), o => new StepLR(o, stepSize: 2, gamma: 0.5f));

    // RMSprop square-avg + momentum buffers + an exponential schedule.
    [Fact]
    public void Checkpoint_ResumesTraining_Identically_RmsPropMomentum_ExponentialLR()
        => RunResumeTest(ps => new RmsProp(ps, lr: 0.01f, momentum: 0.9f), o => new ExponentialLR(o, gamma: 0.9f));

    private static void RunResumeTest(
        Func<IReadOnlyList<Tensor>, Optimizer> makeOpt,
        Func<Optimizer, LrScheduler> makeSched)
    {
        Init.Seed(7);
        var model = MakeModel();
        // Largeish LR + momentum so any missing optimizer/scheduler state diverges >> the tolerance.
        var opt = makeOpt(model.Parameters().ToList());
        var sched = makeSched(opt);
        var (x, y) = Batch();

        // Train enough that optimizer buffers, BN running stats, and the scheduler epoch are all
        // well away from their init defaults before checkpointing.
        for (int i = 0; i < 6; i++) TrainStep(model, opt, sched, x, y);

        var path = Path.Combine(Path.GetTempPath(), $"tensotron_ckpt_{Guid.NewGuid():N}.tns");
        try
        {
            Serialization.SaveCheckpoint(model, opt, sched, path);

            // Reload into a FRESH model+opt+sched built with a DIFFERENT init seed but the ORIGINAL
            // base LR (the documented resume contract) — the checkpoint must overwrite all of it.
            Init.Seed(999);
            var model2 = MakeModel();
            var opt2 = makeOpt(model2.Parameters().ToList());
            var sched2 = makeSched(opt2);
            Serialization.LoadCheckpoint(model2, opt2, sched2, path);

            // Continue BOTH for two more identical, deterministic steps.
            for (int i = 0; i < 2; i++)
            {
                TrainStep(model, opt, sched, x, y);
                TrainStep(model2, opt2, sched2, x, y);
            }

            // (1) Scheduler restored -> learning rates tracked.
            Assert.True(MathF.Abs(opt.LearningRate - opt2.LearningRate) < 1e-6f,
                $"LR drift: {opt.LearningRate} vs {opt2.LearningRate}");

            // (2) Parameters equal -> params + optimizer moments/buffers restored. NOTE: training-mode
            // BatchNorm normalizes with BATCH stats, so this alone does NOT exercise running stats.
            var pa = model.NamedParameters().ToDictionary(p => p.name, p => p.param);
            var pb = model2.NamedParameters().ToDictionary(p => p.name, p => p.param);
            Assert.Equal(pa.Keys.OrderBy(k => k), pb.Keys.OrderBy(k => k));
            foreach (var k in pa.Keys) AssertClose(pa[k].ToArray(), pb[k].ToArray(), $"param {k}");

            // (3) Buffers equal -> BatchNorm running_mean/running_var restored (the actual bug).
            var ba = model.NamedBuffers().ToDictionary(b => b.name, b => b.buffer);
            var bb = model2.NamedBuffers().ToDictionary(b => b.name, b => b.buffer);
            Assert.NotEmpty(ba);   // sanity: the model really has buffers to check
            Assert.Equal(ba.Keys.OrderBy(k => k), bb.Keys.OrderBy(k => k));
            foreach (var k in ba.Keys) AssertClose(ba[k].ToArray(), bb[k].ToArray(), $"buffer {k}");

            // (4) Eval output equal -> running stats actually used at inference match end-to-end.
            model.Eval(); model2.Eval();
            AssertClose(model.Forward(x).ToArray(), model2.Forward(x).ToArray(), "eval output");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    // atol + rtol, comfortably above GPU atomic-reduction noise (~1e-5) and far below the divergence
    // a missing-state bug produces (order of the param/stat magnitude).
    private static void AssertClose(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            float tol = 1e-3f + 1e-3f * MathF.Abs(a[i]);
            Assert.True(diff <= tol, $"{what}[{i}]: {a[i]} vs {b[i]} (diff {diff} > tol {tol})");
        }
    }
}
