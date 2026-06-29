namespace Tensotron;

/// <summary>
/// Learning-rate schedulers (torch.optim.lr_scheduler). torch semantics: construction
/// applies the epoch-0 rate; each <see cref="Step"/> advances one epoch and reapplies the
/// closed-form rate. <see cref="CurrentLr"/> mirrors torch's get_last_lr().
/// </summary>
public abstract class LrScheduler
{
    protected readonly Optimizer Optimizer;
    protected readonly float BaseLr;
    protected int LastEpoch;
    private bool _initialized;

    protected LrScheduler(Optimizer optimizer)
    {
        Optimizer = optimizer;
        BaseLr = optimizer.LearningRate;
        LastEpoch = 0;
        // The epoch-0 rate is applied lazily: Compute() is virtual and the derived
        // ctor's fields aren't set until *after* this base ctor returns.
    }

    private void EnsureInitialized()
    {
        if (_initialized) return;
        _initialized = true;
        Optimizer.LearningRate = Compute(0);
    }

    public float CurrentLr
    {
        get { EnsureInitialized(); return Optimizer.LearningRate; }
    }

    public void Step()
    {
        EnsureInitialized();
        LastEpoch++;
        Optimizer.LearningRate = Compute(LastEpoch);
    }

    /// <summary>Scheduler state for checkpointing: the epoch counter + the lazy-init flag. The
    /// closed-form rate is re-derived from the constructor args (tMax, etaMin, …), so restoring the
    /// epoch fully restores the LR position. Stored as 1-element tensors to share the serializer.</summary>
    public IEnumerable<(string name, Tensor tensor)> StateDict()
    {
        yield return ("last_epoch", Tensor.FromArray(new[] { (float)LastEpoch }, 1));
        yield return ("initialized", Tensor.FromArray(new[] { _initialized ? 1f : 0f }, 1));
    }

    /// <summary>Restore state from <see cref="StateDict"/> and re-apply the LR for that epoch, so a
    /// resumed run continues the same schedule instead of restarting it.</summary>
    public void LoadStateDict(IReadOnlyDictionary<string, Tensor> state)
    {
        if (state.TryGetValue("last_epoch", out var le)) LastEpoch = (int)MathF.Round(le.Item());
        _initialized = state.TryGetValue("initialized", out var ini) && ini.Item() != 0f;
        if (_initialized) Optimizer.LearningRate = Compute(LastEpoch);
    }

    protected abstract float Compute(int epoch);
}

/// <summary>Decay by gamma every stepSize epochs (torch.optim.lr_scheduler.StepLR).</summary>
public sealed class StepLR : LrScheduler
{
    private readonly int _stepSize;
    private readonly float _gamma;
    public StepLR(Optimizer opt, int stepSize, float gamma = 0.1f) : base(opt)
    {
        if (stepSize <= 0) throw new ArgumentOutOfRangeException(nameof(stepSize), $"StepLR requires stepSize > 0 (got {stepSize}).");
        _stepSize = stepSize; _gamma = gamma;
    }
    protected override float Compute(int epoch) => BaseLr * MathF.Pow(_gamma, epoch / _stepSize);
}

/// <summary>Decay by gamma every epoch (torch.optim.lr_scheduler.ExponentialLR).</summary>
public sealed class ExponentialLR : LrScheduler
{
    private readonly float _gamma;
    public ExponentialLR(Optimizer opt, float gamma) : base(opt) => _gamma = gamma;
    protected override float Compute(int epoch) => BaseLr * MathF.Pow(_gamma, epoch);
}

/// <summary>Cosine annealing (torch.optim.lr_scheduler.CosineAnnealingLR).</summary>
public sealed class CosineAnnealingLR : LrScheduler
{
    private readonly int _tMax;
    private readonly float _etaMin;
    public CosineAnnealingLR(Optimizer opt, int tMax, float etaMin = 0f) : base(opt)
    {
        if (tMax <= 0) throw new ArgumentOutOfRangeException(nameof(tMax), $"CosineAnnealingLR requires tMax > 0 (got {tMax}).");
        _tMax = tMax; _etaMin = etaMin;
    }
    protected override float Compute(int epoch)
        => _etaMin + (BaseLr - _etaMin) * (1f + MathF.Cos(MathF.PI * epoch / _tMax)) / 2f;
}

/// <summary>
/// Linear ramp of a multiplicative factor from startFactor to endFactor over totalIters
/// epochs, then constant (torch.optim.lr_scheduler.LinearLR). Handy as a warmup.
/// </summary>
public sealed class LinearLR : LrScheduler
{
    private readonly float _start, _end;
    private readonly int _total;
    public LinearLR(Optimizer opt, float startFactor = 1f / 3f, float endFactor = 1f, int totalIters = 5) : base(opt)
    {
        if (totalIters <= 0) throw new ArgumentOutOfRangeException(nameof(totalIters), $"LinearLR requires totalIters > 0 (got {totalIters}).");
        _start = startFactor; _end = endFactor; _total = totalIters;
    }
    protected override float Compute(int epoch)
    {
        float t = MathF.Min(epoch, _total);
        float factor = _start + (_end - _start) * (t / _total);
        return BaseLr * factor;
    }
}
