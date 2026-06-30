using Tensotron;

namespace Tensotron.Tests;

public class TraceReplayTests
{
    // A captured forward+backward must, on replay, reproduce an independent eager computation for
    // NEW inputs uploaded into the same input buffer — proving replay genuinely recomputes (it is
    // not memoizing the captured step's values). No optimizer Step in the body, so params stay
    // fixed across replays and the comparison is pure recompute.
    [Fact]
    public void Replay_reproduces_eager_forward_backward_for_new_inputs()
    {
        Init.Seed(0);
        const int batch = 16, inDim = 4, width = 8, outDim = 2;
        var l1 = new Linear(inDim, width);
        var l2 = new Linear(width, outDim);
        var model = new Sequential(l1, Activation.Tanh(), l2);
        var ps = model.Parameters().ToList();
        var opt = new Adam(ps, lr: 1e-3f);   // used only for ZeroGrad; never Step → params fixed

        var rng = new Random(7);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var target = Tensor.FromArray(Rand(batch * outDim), batch, outDim);
        var input = Tensor.FromArray(Rand(batch * inDim), batch, inDim);   // stable input buffer

        // Expected eager results (loss + l1 weight grad) for three new inputs, params held fixed.
        var inputs = new[] { Rand(batch * inDim), Rand(batch * inDim), Rand(batch * inDim) };
        var expLoss = new float[inputs.Length];
        var expGrad = new float[inputs.Length][];
        for (int t = 0; t < inputs.Length; t++)
        {
            input.Upload(inputs[t]);
            opt.ZeroGrad();
            var loss = TensorOps.MseLoss(model.Forward(input), target);
            loss.Backward();
            expLoss[t] = loss.Item();
            expGrad[t] = l1.Weight.Grad!.ToArray();
        }

        // Capture the same forward+backward step once.
        var graph = TensorRuntime.Instance.Capture(() =>
        {
            opt.ZeroGrad();
            var loss = TensorOps.MseLoss(model.Forward(input), target);
            loss.Backward();
            return loss;
        });
        var gradOut = l1.Weight.Grad!;   // the (pinned) buffer the trace writes each replay

        // Replay each new input and compare to the eager reference.
        for (int t = 0; t < inputs.Length; t++)
        {
            input.Upload(inputs[t]);
            graph.Replay();
            Assert.Equal(expLoss[t], graph.Output.Item(), 3);
            var g = gradOut.ToArray();
            for (int i = 0; i < g.Length; i++)
                Assert.Equal(expGrad[t][i], g[i], 3);
        }
    }

    // The capture-time "untapped op" guard must fire (not silently corrupt) if a captured body uses
    // an op that does not record a replay thunk. MaxPool2d allocates a data-dependent argmax index
    // buffer and is intentionally not trace-supported.
    [Fact]
    public void Capture_throws_on_unsupported_op()
    {
        Init.Seed(0);
        var x = Tensor.FromArray(new float[1 * 1 * 4 * 4], 1, 1, 4, 4).RequireGrad();
        var pool = new MaxPool2d(2);
        Assert.ThrowsAny<InvalidOperationException>(() =>
            TensorRuntime.Instance.Capture(() => pool.Forward(x).Sum()));
    }

    // A conv stack (im2col + cuBLAS/kernel matmul + bias + ReLU + AvgPool + Linear) exercises the
    // newly trace-tapped ops (Im2Col/Col2Im, AvgPool fwd/grad, batched/2D matmul). Replaying the
    // captured forward+backward for new inputs must match an independent eager computation.
    [Fact]
    public void Replay_reproduces_conv_net_forward_backward()
    {
        Init.Seed(0);
        const int batch = 4, cin = 1, hw = 8;
        var conv = new Conv2d(cin, 4, kernelSize: 3, padding: 1);
        var pool = new AvgPool2d(2);                       // avg, not max — no data-dependent argmax
        var head = new Linear(4 * 4 * 4, 3);
        var ps = conv.Parameters().Concat(head.Parameters()).ToList();
        var opt = new Adam(ps, lr: 1e-3f);                 // ZeroGrad only; never Step → params fixed

        var rng = new Random(11);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var target = Tensor.FromArray(Rand(batch * 3), batch, 3);
        var input = Tensor.FromShaped(Rand(batch * cin * hw * hw), new[] { batch, cin, hw, hw });

        Tensor Body()
        {
            opt.ZeroGrad();
            var h = pool.Forward(conv.Forward(input).Relu());
            var logits = head.Forward(TensorOps.Flatten(h, 1));
            var loss = TensorOps.MseLoss(logits, target);
            loss.Backward();
            return loss;
        }

        var inputs = new[] { Rand(batch * cin * hw * hw), Rand(batch * cin * hw * hw) };
        var expLoss = new float[inputs.Length];
        var expGrad = new float[inputs.Length][];
        for (int t = 0; t < inputs.Length; t++)
        {
            input.Upload(inputs[t]);
            var loss = Body();
            expLoss[t] = loss.Item();
            expGrad[t] = conv.Weight.Grad!.ToArray();
            loss.DisposeGraph();
        }

        using var graph = TensorRuntime.Instance.Capture(Body);
        var gradOut = conv.Weight.Grad!;
        for (int t = 0; t < inputs.Length; t++)
        {
            input.Upload(inputs[t]);
            graph.Replay();
            Assert.Equal(expLoss[t], graph.Output.Item(), 3);
            var g = gradOut.ToArray();
            for (int i = 0; i < g.Length; i++)
                Assert.Equal(expGrad[t][i], g[i], 3);
        }
    }

    // Capturing a step that INCLUDES opt.Step() from a fresh Adam (t=0) and replaying it must track
    // eager training exactly — the Adam bias correction has to advance on each replay (it lives in a
    // device buffer advanced by AdvanceAdam). With the old host-frozen invBc1/invBc2 the replayed
    // params would diverge from eager after the first step; this asserts they don't.
    [Fact]
    public void Replay_with_optimizer_advances_bias_correction()
    {
        const int batch = 8, inDim = 4, width = 8, outDim = 2, steps = 12;
        var rng = new Random(3);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var xData = Rand(batch * inDim);
        var tData = Rand(batch * outDim);

        float[] TrainEager()
        {
            Init.Seed(5);
            var m = new Sequential(new Linear(inDim, width), Activation.Tanh(), new Linear(width, outDim));
            var x = Tensor.FromArray(xData, batch, inDim);
            var t = Tensor.FromArray(tData, batch, outDim);
            var ps = m.Parameters().ToList();
            var opt = new Adam(ps, lr: 1e-2f);
            for (int i = 0; i < steps; i++)
            {
                opt.ZeroGrad();
                var loss = TensorOps.MseLoss(m.Forward(x), t);
                loss.Backward();
                opt.Step();
                loss.DisposeGraph();
            }
            return ps[0].ToArray();
        }

        float[] TrainReplay()
        {
            Init.Seed(5);                              // identical init to the eager reference
            var m = new Sequential(new Linear(inDim, width), Activation.Tanh(), new Linear(width, outDim));
            var x = Tensor.FromArray(xData, batch, inDim);
            var t = Tensor.FromArray(tData, batch, outDim);
            var ps = m.Parameters().ToList();
            var opt = new Adam(ps, lr: 1e-2f);
            Tensor Body()
            {
                opt.ZeroGrad();
                var loss = TensorOps.MseLoss(m.Forward(x), t);
                loss.Backward();
                opt.Step();
                return loss;
            }
            // Warm up one eager step so the optimizer's persistent state (m, v, adv) is allocated
            // BEFORE capture — otherwise its one-time zero-init would be recorded and re-run every
            // replay. This is the documented capture contract (the same warmup PyTorch's CUDA graphs
            // require). Capture is then a steady-state step; bias correction still starts at t=1, so
            // replaying to t=N exercises the on-device advance.
            { var l = Body(); l.DisposeGraph(); }                 // step 1 (warmup)
            using var g = TensorRuntime.Instance.Capture(Body);   // step 2
            for (int i = 0; i < steps - 2; i++) g.Replay();       // steps 3..N (bias correction must advance)
            TensorRuntime.Instance.Sync();
            return ps[0].ToArray();
        }

        var refW = TrainEager();
        var gotW = TrainReplay();
        for (int i = 0; i < refW.Length; i++)
            Assert.Equal(refW[i], gotW[i], 3);
    }

    // Disposing a captured graph releases its pinned trace buffers; a subsequent capture must still
    // work (the pin set is not corrupted) and replay correctly. Guards the reclamation path.
    [Fact]
    public void Disposed_graph_releases_pins_and_recapture_works()
    {
        Init.Seed(0);
        var model = new Sequential(new Linear(4, 8), Activation.Tanh(), new Linear(8, 1));
        var input = Tensor.FromArray(new float[16 * 4], 16, 4);
        var target = Tensor.FromArray(new float[16 * 1], 16, 1);
        Tensor Body() { var l = TensorOps.MseLoss(model.Forward(input), target); l.Backward(); return l; }

        var g1 = TensorRuntime.Instance.Capture(Body);
        g1.Replay();
        g1.Dispose();
        g1.Dispose();                                       // idempotent

        using var g2 = TensorRuntime.Instance.Capture(Body); // pin set not corrupted by g1's release
        g2.Replay();
        Assert.False(float.IsNaN(g2.Output.Item()));
    }
}
