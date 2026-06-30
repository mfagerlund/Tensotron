using static Tensotron.TensorOps;

namespace Tensotron;

/// <summary>
/// Gradient-descent optimizers, PyTorch-faithful. Each Step() runs under NoGrad and
/// writes updated parameters back in place (so the leaf identity survives across steps).
/// State (momentum buffers, Adam moments) is kept per-parameter, keyed by identity.
/// </summary>
public abstract class Optimizer
{
    protected readonly IReadOnlyList<Tensor> Parameters;

    private float _learningRate;

    /// <summary>The learning rate. For a <c>capturable</c> optimizer the setter also mirrors the new
    /// value into a persistent device scalar (via <see cref="OnLearningRateChanged"/>), so changing it
    /// between captured-graph replays — or stepping an <see cref="LrScheduler"/> — is honoured by the
    /// next replay rather than frozen at capture time. See <see cref="Adam"/>.</summary>
    public float LearningRate
    {
        get => _learningRate;
        set { _learningRate = value; OnLearningRateChanged(value); }
    }

    /// <summary>Fired whenever <see cref="LearningRate"/> is set. Default no-op; a capturable optimizer
    /// overrides it to upload the rate into its device scalar. (Called from the base constructor before
    /// derived fields are initialized, so overrides must tolerate a not-yet-allocated buffer.)</summary>
    protected virtual void OnLearningRateChanged(float lr) { }

    protected Optimizer(IReadOnlyList<Tensor> parameters, float lr)
    {
        Parameters = parameters;
        LearningRate = lr;
    }

    /// <summary>Clear the accumulated gradient on every parameter.</summary>
    public void ZeroGrad()
    {
        foreach (var p in Parameters) p.Grad = null;
    }

    public abstract void Step();

    /// <summary>
    /// Per-parameter optimizer state (momentum/Adam moments + step counts) as named tensors, keyed
    /// by the parameter's index in <see cref="Parameters"/> (e.g. <c>"3.m"</c>, <c>"3.step"</c>).
    /// Step counts are stored as 1-element tensors so the whole state rides the same tensor
    /// serializer. Empty for stateless optimizers / params not yet stepped.
    /// </summary>
    public virtual IEnumerable<(string name, Tensor tensor)> StateDict() => Array.Empty<(string, Tensor)>();

    /// <summary>Restore state produced by <see cref="StateDict"/> (same parameter ordering). Resumes
    /// training exactly — without it, a reloaded optimizer restarts its moments and diverges.</summary>
    public virtual void LoadStateDict(IReadOnlyDictionary<string, Tensor> state) { }

    /// <summary>Wrap an int (e.g. a step count) as a 1-element tensor for serialization.</summary>
    protected static Tensor Scalar1(float v) => Tensor.FromArray(new[] { v }, 1);

    // Run an update closure with autograd disabled and write the result back in place.
    protected static void Update(Tensor param, Tensor newValue)
    {
        param.CopyInPlace(newValue);
    }

    protected void StepNoGrad(Action body)
    {
        using (Tensor.NoGradScope()) body();
    }
}

/// <summary>SGD with optional momentum, weight decay, dampening and Nesterov (torch.optim.SGD).</summary>
public sealed class Sgd : Optimizer
{
    private readonly float _momentum, _weightDecay, _dampening;
    private readonly bool _nesterov;
    private readonly bool _capturable;
    private readonly Tensor? _lrDevice;
    private readonly Dictionary<Tensor, Tensor> _buf = new();

    /// <param name="capturable">See <see cref="Adam(IReadOnlyList{Tensor}, float, float, float, float, float, bool)"/>:
    /// reads the learning rate from a device scalar so the step can live in a captured graph.</param>
    public Sgd(IReadOnlyList<Tensor> parameters, float lr,
        float momentum = 0f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false,
        bool capturable = false)
        : base(parameters, lr)
    {
        _momentum = momentum;
        _weightDecay = weightDecay;
        _dampening = dampening;
        _nesterov = nesterov;
        _capturable = capturable;
        if (capturable) _lrDevice = Tensor.FromArray(new[] { lr }, 1);
    }

    protected override void OnLearningRateChanged(float lr) => _lrDevice?.Upload(new[] { lr });

    public override void Step() => StepNoGrad(() =>
    {
        var rt = TensorRuntime.Instance;
        foreach (var p in Parameters)
        {
            if (p.Grad == null) continue;

            // Momentum buffer (when used) is allocated once and updated in place by the kernel;
            // hasBuf=0 on its first step seeds buf=g (matching torch). momentum=0 ⇒ buf unused,
            // so we pass the grad buffer as a harmless placeholder.
            Tensor buf;
            float hasBuf;
            if (_momentum != 0f)
            {
                if (_buf.TryGetValue(p, out buf!)) hasBuf = 1f;
                else { buf = Tensor.Zeros(p.Shape); _buf[p] = buf; hasBuf = 0f; }
            }
            else { buf = p.Grad!; hasBuf = 0f; }

            if (_capturable)
                rt.LaunchSgdCapturable(
                    p.Buffer, p.Grad!.Buffer, buf.Buffer,
                    _lrDevice!.Buffer, _momentum, _weightDecay, _dampening, _nesterov ? 1f : 0f, hasBuf);
            else
                rt.LaunchSgd(
                    p.Buffer, p.Grad!.Buffer, buf.Buffer,
                    LearningRate, _momentum, _weightDecay, _dampening, _nesterov ? 1f : 0f, hasBuf);
        }
    });

    public override IEnumerable<(string, Tensor)> StateDict()
    {
        for (int i = 0; i < Parameters.Count; i++)
            if (_buf.TryGetValue(Parameters[i], out var buf))
                yield return ($"{i}.momentum_buffer", buf);
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, Tensor> state)
    {
        for (int i = 0; i < Parameters.Count; i++)
            if (state.TryGetValue($"{i}.momentum_buffer", out var buf))
                _buf[Parameters[i]] = buf;
    }
}

/// <summary>Adam (torch.optim.Adam). weight_decay is the coupled L2 variant.</summary>
public sealed class Adam : Optimizer
{
    private readonly float _b1, _b2, _eps, _weightDecay;
    private readonly bool _capturable;
    private readonly Tensor? _lrDevice;   // capturable: persistent [lr], the kernel reads it per replay
    private readonly Dictionary<Tensor, (Tensor m, Tensor v, Tensor adv)> _state = new();

    /// <param name="capturable">When true, the fused step reads the learning rate from a device scalar
    /// (torch's <c>capturable=True</c>): the whole <see cref="Step"/> can be folded into a captured
    /// graph and the LR still varies across replays. Off by default — it adds a tiny per-step H2D copy
    /// that the ungraphed path doesn't need.</param>
    public Adam(IReadOnlyList<Tensor> parameters, float lr = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 0f,
        bool capturable = false)
        : base(parameters, lr)
    {
        _b1 = beta1; _b2 = beta2; _eps = eps; _weightDecay = weightDecay;
        _capturable = capturable;
        if (capturable) _lrDevice = Tensor.FromArray(new[] { lr }, 1);
    }

    protected override void OnLearningRateChanged(float lr) => _lrDevice?.Upload(new[] { lr });

    public override void Step() => StepNoGrad(() =>
    {
        var rt = TensorRuntime.Instance;
        foreach (var p in Parameters)
        {
            if (p.Grad == null) continue;

            // State (m, v) and the step/bias-correction triple adv=[t, invBc1, invBc2] are allocated
            // once and updated IN PLACE on the device — no per-step intermediate tensors or host
            // scalar uploads, so a captured step replays with the bias correction advancing on-device.
            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape), Tensor.Zeros(p.Shape), NewAdvState());
            _state[p] = s;

            rt.LaunchAdvanceAdam(s.adv.Buffer, _b1, _b2);
            if (_capturable)
                rt.LaunchAdamCapturable(
                    p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
                    _b1, 1f - _b1, _b2, 1f - _b2, _lrDevice!.Buffer, _eps, s.adv.Buffer,
                    coupledWd: _weightDecay, decoupledWd: 0f);
            else
                rt.LaunchAdam(
                    p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
                    _b1, 1f - _b1, _b2, 1f - _b2, LearningRate, _eps, s.adv.Buffer,
                    coupledWd: _weightDecay, decoupledFactor: 1f);
        }
    });

    // The device-resident Adam step state: [t=0, invBc1, invBc2]; LaunchAdvanceAdam fills the
    // bias corrections on the first step (t→1). Held for the optimizer's life, like m/v.
    internal static Tensor NewAdvState() => Tensor.FromArray(new[] { 0f, 0f, 0f }, 3);

    public override IEnumerable<(string, Tensor)> StateDict() => AdamState(_state, Parameters);
    public override void LoadStateDict(IReadOnlyDictionary<string, Tensor> state) => LoadAdamState(_state, Parameters, state, _b1, _b2);

    // Shared (m, v, step) serialization for Adam/AdamW. The step count lives in adv[0] on the device;
    // it is pulled to host only here (checkpointing), not in the hot step.
    internal static IEnumerable<(string, Tensor)> AdamState(
        Dictionary<Tensor, (Tensor m, Tensor v, Tensor adv)> st, IReadOnlyList<Tensor> ps)
    {
        for (int i = 0; i < ps.Count; i++)
            if (st.TryGetValue(ps[i], out var s))
            {
                yield return ($"{i}.m", s.m);
                yield return ($"{i}.v", s.v);
                yield return ($"{i}.step", Scalar1(s.adv.ToArray()[0]));
            }
    }

    internal static void LoadAdamState(
        Dictionary<Tensor, (Tensor m, Tensor v, Tensor adv)> st, IReadOnlyList<Tensor> ps,
        IReadOnlyDictionary<string, Tensor> state, float b1, float b2)
    {
        for (int i = 0; i < ps.Count; i++)
            if (state.TryGetValue($"{i}.m", out var m) &&
                state.TryGetValue($"{i}.v", out var v) &&
                state.TryGetValue($"{i}.step", out var step))
            {
                int t = (int)MathF.Round(step.Item());
                // Rebuild the device step state so the next step resumes at t+1 with correct bias
                // correction (matching torch resume). invBc are recomputed from t via the same Pow.
                float invBc1 = t == 0 ? 0f : 1f / (1f - MathF.Pow(b1, t));
                float invBc2 = t == 0 ? 0f : 1f / (1f - MathF.Pow(b2, t));
                st[ps[i]] = (m, v, Tensor.FromArray(new[] { (float)t, invBc1, invBc2 }, 3));
            }
    }
}

/// <summary>AdamW (torch.optim.AdamW): decoupled weight decay applied directly to the params.</summary>
public sealed class AdamW : Optimizer
{
    private readonly float _b1, _b2, _eps, _weightDecay;
    private readonly bool _capturable;
    private readonly Tensor? _lrDevice;
    private readonly Dictionary<Tensor, (Tensor m, Tensor v, Tensor adv)> _state = new();

    /// <param name="capturable">See <see cref="Adam(IReadOnlyList{Tensor}, float, float, float, float, float, bool)"/>.
    /// In the capturable path the decoupled-decay factor 1 − lr·wd is recomputed on-device from the
    /// live LR each step, so it tracks an annealed LR across replays.</param>
    public AdamW(IReadOnlyList<Tensor> parameters, float lr = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 1e-2f,
        bool capturable = false)
        : base(parameters, lr)
    {
        _b1 = beta1; _b2 = beta2; _eps = eps; _weightDecay = weightDecay;
        _capturable = capturable;
        if (capturable) _lrDevice = Tensor.FromArray(new[] { lr }, 1);
    }

    protected override void OnLearningRateChanged(float lr) => _lrDevice?.Upload(new[] { lr });

    public override void Step() => StepNoGrad(() =>
    {
        var rt = TensorRuntime.Instance;
        foreach (var p in Parameters)
        {
            if (p.Grad == null) continue;

            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape), Tensor.Zeros(p.Shape), Adam.NewAdvState());
            _state[p] = s;

            // Decoupled decay p <- p·(1 − lr·wd) is folded into the fused kernel's factor.
            rt.LaunchAdvanceAdam(s.adv.Buffer, _b1, _b2);
            if (_capturable)
                // decoupledWd carries wd; the kernel forms 1 − lr·wd from the live device LR.
                rt.LaunchAdamCapturable(
                    p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
                    _b1, 1f - _b1, _b2, 1f - _b2, _lrDevice!.Buffer, _eps, s.adv.Buffer,
                    coupledWd: 0f, decoupledWd: _weightDecay);
            else
            {
                float decoupled = _weightDecay != 0f ? 1f - LearningRate * _weightDecay : 1f;
                rt.LaunchAdam(
                    p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
                    _b1, 1f - _b1, _b2, 1f - _b2, LearningRate, _eps, s.adv.Buffer,
                    coupledWd: 0f, decoupledFactor: decoupled);
            }
        }
    });

    public override IEnumerable<(string, Tensor)> StateDict() => Adam.AdamState(_state, Parameters);
    public override void LoadStateDict(IReadOnlyDictionary<string, Tensor> state) => Adam.LoadAdamState(_state, Parameters, state, _b1, _b2);
}

/// <summary>RMSprop (torch.optim.RMSprop) with optional momentum and centering.</summary>
public sealed class RmsProp : Optimizer
{
    private readonly float _alpha, _eps, _weightDecay, _momentum;
    private readonly bool _centered;
    private readonly Dictionary<Tensor, (Tensor v, Tensor? avg, Tensor? buf)> _state = new();

    public RmsProp(IReadOnlyList<Tensor> parameters, float lr = 1e-2f,
        float alpha = 0.99f, float eps = 1e-8f, float weightDecay = 0f,
        float momentum = 0f, bool centered = false)
        : base(parameters, lr)
    {
        _alpha = alpha; _eps = eps; _weightDecay = weightDecay;
        _momentum = momentum; _centered = centered;
    }

    public override void Step() => StepNoGrad(() =>
    {
        foreach (var p in Parameters)
        {
            if (p.Grad == null) continue;
            var g = p.Grad;
            if (_weightDecay != 0f) g = Add(g, Mul(p, Scalar(_weightDecay)));

            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape),
                     _centered ? Tensor.Zeros(p.Shape) : null,
                     _momentum != 0f ? Tensor.Zeros(p.Shape) : null);

            var v = Add(Mul(s.v, Scalar(_alpha)), Mul(Square(g), Scalar(1f - _alpha)));
            Tensor denomBase = v;
            Tensor? avg = s.avg;
            if (_centered)
            {
                avg = Add(Mul(s.avg!, Scalar(_alpha)), Mul(g, Scalar(1f - _alpha)));
                denomBase = Sub(v, Square(avg));
            }
            var denom = Add(Sqrt(denomBase), Scalar(_eps));

            Tensor? buf = s.buf;
            if (_momentum != 0f)
            {
                buf = Add(Mul(s.buf!, Scalar(_momentum)), Div(g, denom));
                Update(p, Sub(p, Mul(buf, Scalar(LearningRate))));
            }
            else
            {
                Update(p, Sub(p, Mul(Div(g, denom), Scalar(LearningRate))));
            }
            _state[p] = (v, avg, buf);
        }
    });

    public override IEnumerable<(string, Tensor)> StateDict()
    {
        for (int i = 0; i < Parameters.Count; i++)
            if (_state.TryGetValue(Parameters[i], out var s))
            {
                yield return ($"{i}.square_avg", s.v);
                if (s.avg is not null) yield return ($"{i}.grad_avg", s.avg);
                if (s.buf is not null) yield return ($"{i}.momentum_buffer", s.buf);
            }
    }

    public override void LoadStateDict(IReadOnlyDictionary<string, Tensor> state)
    {
        for (int i = 0; i < Parameters.Count; i++)
            if (state.TryGetValue($"{i}.square_avg", out var v))
                _state[Parameters[i]] = (
                    v,
                    state.TryGetValue($"{i}.grad_avg", out var avg) ? avg : null,
                    state.TryGetValue($"{i}.momentum_buffer", out var buf) ? buf : null);
    }
}

/// <summary>Gradient utilities (torch.nn.utils).</summary>
public static class GradUtils
{
    /// <summary>
    /// Clip the global L2 norm of all parameter gradients to <paramref name="maxNorm"/>,
    /// in place. Returns the pre-clip total norm. Matches torch.nn.utils.clip_grad_norm_.
    /// </summary>
    /// <remarks>
    /// Fully device-resident: Σ‖g‖² accumulates into a single device scalar, the norm and
    /// scale coefficient are computed on-device, and every grad is unconditionally scaled by
    /// coef (a no-op multiply by 1.0 when within norm) — so there is no host
    /// <c>if (coef &lt; 1)</c> branch and no per-parameter <c>.Item()</c> pull. The ONLY sync
    /// is the final pull of the pre-clip norm for the return value; pass
    /// <paramref name="returnTotalNorm"/> = false (as the PPO update does) to skip it and run
    /// the whole clip at zero syncs. Clip semantics are identical either way.
    /// </remarks>
    public static float ClipGradNorm(IReadOnlyList<Tensor> parameters, float maxNorm,
                                     float eps = 1e-6f, bool returnTotalNorm = true)
    {
        using (Tensor.NoGradScope())
        {
            // Σ‖g‖² as a single device scalar — add per-param sum-of-squares on-device.
            Tensor? sumSq = null;
            foreach (var p in parameters)
            {
                if (p.Grad == null) continue;
                var s = Sum(Square(p.Grad));
                sumSq = sumSq == null ? s : Add(sumSq, s);
            }
            if (sumSq == null) return 0f;   // no grads: norm 0, nothing to scale

            var total = Sqrt(sumSq);        // device scalar; pre-clip norm
            // coef = min(1, maxNorm / (total + eps)); ratio is ≥ 0 so clamp-low never bites.
            var coef = Clamp(Div(Scalar(maxNorm), Add(total, Scalar(eps))), 0f, 1f);

            // Unconditionally scale every grad by coef — multiplying by 1.0 within norm is a
            // cheap no-op, and avoids the host branch + sync the conditional version needed.
            foreach (var p in parameters)
                if (p.Grad != null) p.Grad.CopyInPlace(Mul(p.Grad, coef));

            // `total` is a distinct buffer computed before the in-place scaling was queued, so
            // (in-order stream) this reads the genuine pre-clip norm. Skipped when not wanted.
            return returnTotalNorm ? total.Item() : float.NaN;
        }
    }
}
