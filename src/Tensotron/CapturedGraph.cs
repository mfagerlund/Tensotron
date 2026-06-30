namespace Tensotron;

/// <summary>
/// A fixed-shape step (forward + backward + optimizer) recorded once as an ordered list of device
/// kernel launches, replayable with no host-side autograd-graph rebuild. Produced by
/// <see cref="TensorRuntime.Capture"/>.
///
/// For a small model the host cost of rebuilding the C# graph every step dominates the wall clock
/// (the GPU is otherwise idle ~95% of the step); <see cref="Replay"/> skips all of it. On the CUDA
/// backend the recorded launches are folded into one native CUDA graph and replayed with a single
/// <c>cuGraphLaunch</c> (see <see cref="UsesNativeGraph"/>); otherwise replay re-fires the recorded
/// launches one by one. Either way: write the next minibatch into the SAME input tensors with
/// <see cref="Tensor.Upload"/> before each replay, then read <see cref="Output"/> after.
///
/// Limitations (see <see cref="TensorRuntime.Capture"/> remarks): trace buffers are pinned for the
/// graph's lifetime and released when the graph is disposed; Adam bias correction advances on the
/// device so it stays correct across replays, but other step-dependent scalars baked into a kernel
/// at capture (e.g. a scheduled LR) are frozen, and data-dependent index ops (maxpool argmax,
/// gather) are not supported.
/// </summary>
public sealed class CapturedGraph : IDisposable
{
    private readonly TensorRuntime _rt;
    private readonly List<Action> _thunks;
    private readonly Action? _nativeReplay;   // non-null => replay via one cuGraphLaunch
    private Action? _onDispose;   // un-pins trace buffers (+ destroys the native graph); null once disposed

    /// <summary>The tensor returned by the captured body (e.g. the loss). Its buffer is pinned and
    /// overwritten by each <see cref="Replay"/>, so reading it (Item/ToArray) yields the latest result.</summary>
    public Tensor Output { get; }

    /// <summary>Number of device launches the captured step performs (the per-replay launch count).</summary>
    public int LaunchCount => _thunks.Count;

    /// <summary>True when replay issues a single native CUDA-graph launch (the whole step folded into
    /// one driver call) rather than re-firing the recorded launches host-side. False on non-CUDA
    /// backends or if graph capture/instantiation failed (replay then falls back to the thunks).</summary>
    public bool UsesNativeGraph => _nativeReplay != null;

    internal CapturedGraph(TensorRuntime rt, List<Action> thunks, Tensor output,
        Action? nativeReplay, Action onDispose)
    {
        _rt = rt;
        _thunks = thunks;
        _nativeReplay = nativeReplay;
        Output = output;
        _onDispose = onDispose;
    }

    /// <summary>Re-execute the captured step — one <c>cuGraphLaunch</c> when a native graph was built
    /// (<see cref="UsesNativeGraph"/>), else re-fire the recorded launches in order. Does not
    /// synchronize — read <see cref="Output"/> (which pulls to host) or call
    /// <see cref="TensorRuntime.Sync"/> when the result is needed.</summary>
    public void Replay()
    {
        if (_nativeReplay != null) _nativeReplay();
        else _rt.Replay(_thunks);
    }

    /// <summary>Release the graph's pinned trace buffers back to the runtime so the allocator can
    /// recycle them. After disposal the graph must not be replayed. Idempotent.</summary>
    public void Dispose()
    {
        _onDispose?.Invoke();
        _onDispose = null;
    }
}
