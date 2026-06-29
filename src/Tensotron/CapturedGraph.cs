namespace Tensotron;

/// <summary>
/// A fixed-shape step (forward + backward + optimizer) recorded once as an ordered list of device
/// kernel launches, replayable with no host-side autograd-graph rebuild. Produced by
/// <see cref="TensorRuntime.Capture"/>.
///
/// For a small model the host cost of rebuilding the C# graph every step dominates the wall clock
/// (the GPU is otherwise idle ~95% of the step); <see cref="Replay"/> skips all of it and just
/// re-fires the recorded launches. Write the next minibatch into the SAME input tensors with
/// <see cref="Tensor.Upload"/> before each replay, then read <see cref="Output"/> after.
///
/// Limitations (see <see cref="TensorRuntime.Capture"/> remarks): trace buffers are pinned for the
/// graph's lifetime (no reclamation), step-dependent optimizer scalars (Adam bias correction,
/// scheduled LR) are frozen at capture time, and data-dependent index ops (maxpool argmax, gather)
/// are not supported.
/// </summary>
public sealed class CapturedGraph
{
    private readonly TensorRuntime _rt;
    private readonly List<Action> _thunks;

    /// <summary>The tensor returned by the captured body (e.g. the loss). Its buffer is pinned and
    /// overwritten by each <see cref="Replay"/>, so reading it (Item/ToArray) yields the latest result.</summary>
    public Tensor Output { get; }

    /// <summary>Number of device launches the captured step performs (the per-replay launch count).</summary>
    public int LaunchCount => _thunks.Count;

    internal CapturedGraph(TensorRuntime rt, List<Action> thunks, Tensor output)
    {
        _rt = rt;
        _thunks = thunks;
        Output = output;
    }

    /// <summary>Re-execute the recorded launches in order. Does not synchronize — read
    /// <see cref="Output"/> (which pulls to host) or call <see cref="TensorRuntime.Sync"/> when the
    /// result is needed.</summary>
    public void Replay() => _rt.Replay(_thunks);
}
