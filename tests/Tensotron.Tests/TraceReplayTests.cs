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
}
