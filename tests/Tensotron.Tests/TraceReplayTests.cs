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

    // The point of capture on CUDA: the recorded launches fold into ONE native CUDA graph, so replay
    // is a single cuGraphLaunch (UsesNativeGraph), not N host dispatches. Asserts the graph engages
    // AND still reproduces eager results for new inputs uploaded into the same input buffer.
    [Fact]
    public void Native_cuda_graph_is_used_and_reproduces_eager()
    {
        Init.Seed(0);
        const int batch = 16, inDim = 4, width = 8, outDim = 2;
        var model = new Sequential(new Linear(inDim, width), Activation.Tanh(), new Linear(width, outDim));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);   // ZeroGrad only; params fixed
        var rng = new Random(7);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var target = Tensor.FromArray(Rand(batch * outDim), batch, outDim);
        var input = Tensor.FromArray(Rand(batch * inDim), batch, inDim);
        Tensor Body() { opt.ZeroGrad(); var l = TensorOps.MseLoss(model.Forward(input), target); l.Backward(); return l; }

        var inputs = new[] { Rand(batch * inDim), Rand(batch * inDim) };
        var expLoss = new float[inputs.Length];
        for (int t = 0; t < inputs.Length; t++) { input.Upload(inputs[t]); var l = Body(); expLoss[t] = l.Item(); l.DisposeGraph(); }

        using var graph = TensorRuntime.Instance.Capture(Body);
        if (TensorRuntime.Instance.UsesCuBlas)   // CUDA backend -> a native graph must have been built
            Assert.True(graph.UsesNativeGraph, "expected the captured step to fold into a native CUDA graph");
        for (int t = 0; t < inputs.Length; t++)
        {
            input.Upload(inputs[t]);
            graph.Replay();
            Assert.Equal(expLoss[t], graph.Output.Item(), 3);
        }
    }

    // A step whose matmul is large enough (M,N,K >= 64) to take the cuBLAS SGEMM path, captured into
    // the native graph. Proves cuBLAS launches are graph-capturable (else the build would fail and
    // replay silently fall back), and that the graph reproduces eager.
    [Fact]
    public void Native_cuda_graph_captures_cublas_gemm()
    {
        Init.Seed(0);
        const int batch = 128, inDim = 128, outDim = 128;   // M,N,K >= 64 -> cuBLAS path
        var model = new Sequential(new Linear(inDim, outDim));
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-4f); // ZeroGrad only; params fixed
        var rng = new Random(9);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var target = Tensor.FromArray(Rand(batch * outDim), batch, outDim);
        var input = Tensor.FromArray(Rand(batch * inDim), batch, inDim);
        Tensor Body() { opt.ZeroGrad(); var l = TensorOps.MseLoss(model.Forward(input), target); l.Backward(); return l; }

        var inputs = new[] { Rand(batch * inDim), Rand(batch * inDim) };
        var expLoss = new float[inputs.Length];
        for (int t = 0; t < inputs.Length; t++) { input.Upload(inputs[t]); var l = Body(); expLoss[t] = l.Item(); l.DisposeGraph(); }

        using var graph = TensorRuntime.Instance.Capture(Body);
        if (TensorRuntime.Instance.UsesCuBlas)
            Assert.True(graph.UsesNativeGraph, "cuBLAS GEMM should be capturable into the native CUDA graph");
        for (int t = 0; t < inputs.Length; t++)
        {
            input.Upload(inputs[t]);
            graph.Replay();
            Assert.Equal(expLoss[t], graph.Output.Item(), 2);
        }
    }

    // The capturable optimizer's whole point: the learning rate is read from a device scalar on every
    // replay, so a captured full step (fwd→loss→bwd→opt.Step) honours an LR uploaded BETWEEN replays
    // instead of freezing the capture-time rate. This is checked DETERMINISTICALLY via the lr=0
    // invariant: with lr=0 the update term is exactly zero ((1−0·wd)·p − 0·… = p), so the params cannot
    // move — independent of the (noisy, atomic-reduced) gradient the replayed backward produces. A
    // frozen capture-time LR (the bug) WOULD move them, so an exact "unchanged" assertion is an
    // unambiguous, noise-immune discriminator. The mirror assertions show a non-zero live LR actually
    // moves params, so the test can't pass by the kernel simply doing nothing.
    [Fact]
    public void Captured_step_honours_live_learning_rate_per_replay()
    {
        const int batch = 8, inDim = 4, width = 8, outDim = 2;
        var rng = new Random(13);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var x = Tensor.FromArray(Rand(batch * inDim), batch, inDim);
        var t = Tensor.FromArray(Rand(batch * outDim), batch, outDim);

        Init.Seed(5);
        var m = new Sequential(new Linear(inDim, width), Activation.Tanh(), new Linear(width, outDim));
        var ps = m.Parameters().ToList();
        var opt = new Adam(ps, lr: 1e-2f, capturable: true);
        Tensor Body()
        {
            opt.ZeroGrad();
            var loss = TensorOps.MseLoss(m.Forward(x), t);
            loss.Backward();
            opt.Step();
            return loss;
        }

        // Warmup eagerly so the optimizer's state (m, v, adv, and the LR scalar) is allocated before
        // capture (the documented steady-state contract). Capture then EXECUTES the body once (a real
        // non-zero-LR step); snapshot params AFTER capture as the reference point.
        opt.LearningRate = 1e-2f;
        { var l = Body(); l.DisposeGraph(); }
        using var g = TensorRuntime.Instance.Capture(Body);
        if (TensorRuntime.Instance.UsesCuBlas)
            Assert.True(g.UsesNativeGraph, "expected the captured capturable step to fold into a native CUDA graph");
        var p0 = ps[0].ToArray();

        // Replay with LR uploaded as 0 → update term is exactly zero → params bit-unchanged.
        opt.LearningRate = 0f;
        g.Replay();
        TensorRuntime.Instance.Sync();
        var afterZero = ps[0].ToArray();
        for (int i = 0; i < p0.Length; i++)
            Assert.Equal(p0[i], afterZero[i]);   // exact: a frozen non-zero LR would have moved these

        // Replay with a real LR uploaded → params must actually move (the live scalar drives work).
        opt.LearningRate = 5e-2f;
        g.Replay();
        TensorRuntime.Instance.Sync();
        var afterStep = ps[0].ToArray();
        bool moved = false;
        for (int i = 0; i < p0.Length; i++) if (MathF.Abs(afterStep[i] - p0[i]) > 1e-6f) { moved = true; break; }
        Assert.True(moved, "a non-zero live LR must move params on replay");

        // LR back to 0 from the new point → frozen again, proving each replay re-reads the scalar.
        opt.LearningRate = 0f;
        g.Replay();
        TensorRuntime.Instance.Sync();
        var afterZero2 = ps[0].ToArray();
        for (int i = 0; i < afterStep.Length; i++)
            Assert.Equal(afterStep[i], afterZero2[i]);
    }

    // A coefficient routed through Tensor.ScalarInput must stay LIVE across captured-graph replays:
    // uploading a new value before Replay changes both the forward result and the backward grads,
    // exactly like feeding a new minibatch into a stable input buffer. This is the capture-frozen-scalar
    // fix for annealed loss coefficients (entropy / clip-ε weight) and grad-clip thresholds — a plain
    // float operand would bake the capture-time value into the graph and freeze it, so every replay
    // would return the SAME loss. The body has no optimizer Step, so params stay fixed and the only
    // thing that varies between replays is the uploaded scalar.
    [Fact]
    public void Captured_step_honours_live_scalar_input_per_replay()
    {
        Init.Seed(0);
        const int batch = 8, inDim = 4, width = 8, outDim = 2;
        var l1 = new Linear(inDim, width);
        var l2 = new Linear(width, outDim);
        var model = new Sequential(l1, Activation.Tanh(), l2);
        var opt = new Adam(model.Parameters().ToList(), lr: 1e-3f);   // ZeroGrad only; never Step → params fixed

        var rng = new Random(21);
        float[] Rand(int n) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() - 0.5); return a; }
        var x = Tensor.FromArray(Rand(batch * inDim), batch, inDim);          // fixed input
        var target = Tensor.FromArray(Rand(batch * outDim), batch, outDim);
        var coef = Tensor.ScalarInput(1f, "coef");                            // persistent, uploaded per replay

        Tensor Body()
        {
            opt.ZeroGrad();
            var loss = TensorOps.Mul(TensorOps.MseLoss(model.Forward(x), target), coef);
            loss.Backward();
            return loss;
        }

        // Eager reference: loss + l1 weight grad for several distinct coefficient values (params fixed).
        var coefs = new[] { 2f, 0.5f, 3f };
        var expLoss = new float[coefs.Length];
        var expGrad = new float[coefs.Length][];
        for (int k = 0; k < coefs.Length; k++)
        {
            coef.Upload(new[] { coefs[k] });
            var l = Body();
            expLoss[k] = l.Item();
            expGrad[k] = l1.Weight.Grad!.ToArray();
            l.DisposeGraph();
        }
        // Distinct coefficients must give distinct losses — else a frozen scalar could pass trivially.
        Assert.True(MathF.Abs(expLoss[0] - expLoss[1]) > 1e-4f);

        using var g = TensorRuntime.Instance.Capture(Body);
        if (TensorRuntime.Instance.UsesCuBlas)
            Assert.True(g.UsesNativeGraph, "expected the captured step to fold into a native CUDA graph");
        var gradOut = l1.Weight.Grad!;
        for (int k = 0; k < coefs.Length; k++)
        {
            coef.Upload(new[] { coefs[k] });
            g.Replay();
            Assert.Equal(expLoss[k], g.Output.Item(), 3);
            var gg = gradOut.ToArray();
            for (int i = 0; i < gg.Length; i++)
                Assert.Equal(expGrad[k][i], gg[i], 3);
        }
    }
}
