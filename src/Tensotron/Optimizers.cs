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
    public float LearningRate { get; set; }

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
    private readonly Dictionary<Tensor, Tensor> _buf = new();

    public Sgd(IReadOnlyList<Tensor> parameters, float lr,
        float momentum = 0f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false)
        : base(parameters, lr)
    {
        _momentum = momentum;
        _weightDecay = weightDecay;
        _dampening = dampening;
        _nesterov = nesterov;
    }

    public override void Step() => StepNoGrad(() =>
    {
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

            TensorRuntime.Instance.LaunchSgd(
                p.Buffer, p.Grad!.Buffer, buf.Buffer,
                LearningRate, _momentum, _weightDecay, _dampening, _nesterov ? 1f : 0f, hasBuf);
        }
    });
}

/// <summary>Adam (torch.optim.Adam). weight_decay is the coupled L2 variant.</summary>
public sealed class Adam : Optimizer
{
    private readonly float _b1, _b2, _eps, _weightDecay;
    private readonly Dictionary<Tensor, (Tensor m, Tensor v, int t)> _state = new();

    public Adam(IReadOnlyList<Tensor> parameters, float lr = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 0f)
        : base(parameters, lr)
    {
        _b1 = beta1; _b2 = beta2; _eps = eps; _weightDecay = weightDecay;
    }

    public override void Step() => StepNoGrad(() =>
    {
        foreach (var p in Parameters)
        {
            if (p.Grad == null) continue;

            // State (m, v) is allocated once and updated IN PLACE by the fused kernel — no
            // per-step intermediate tensors or scalar uploads (the old path was ~15 ops/param).
            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape), Tensor.Zeros(p.Shape), 0);
            int t = s.t + 1;
            _state[p] = (s.m, s.v, t);

            float invBc1 = 1f / (1f - MathF.Pow(_b1, t));
            float invBc2 = 1f / (1f - MathF.Pow(_b2, t));
            TensorRuntime.Instance.LaunchAdam(
                p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
                _b1, 1f - _b1, _b2, 1f - _b2, LearningRate, _eps, invBc1, invBc2,
                coupledWd: _weightDecay, decoupledFactor: 1f);
        }
    });
}

/// <summary>AdamW (torch.optim.AdamW): decoupled weight decay applied directly to the params.</summary>
public sealed class AdamW : Optimizer
{
    private readonly float _b1, _b2, _eps, _weightDecay;
    private readonly Dictionary<Tensor, (Tensor m, Tensor v, int t)> _state = new();

    public AdamW(IReadOnlyList<Tensor> parameters, float lr = 1e-3f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 1e-2f)
        : base(parameters, lr)
    {
        _b1 = beta1; _b2 = beta2; _eps = eps; _weightDecay = weightDecay;
    }

    public override void Step() => StepNoGrad(() =>
    {
        foreach (var p in Parameters)
        {
            if (p.Grad == null) continue;

            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape), Tensor.Zeros(p.Shape), 0);
            int t = s.t + 1;
            _state[p] = (s.m, s.v, t);

            float invBc1 = 1f / (1f - MathF.Pow(_b1, t));
            float invBc2 = 1f / (1f - MathF.Pow(_b2, t));
            // Decoupled decay p <- p·(1 − lr·wd) is folded into the fused kernel's factor.
            float decoupled = _weightDecay != 0f ? 1f - LearningRate * _weightDecay : 1f;
            TensorRuntime.Instance.LaunchAdam(
                p.Buffer, p.Grad!.Buffer, s.m.Buffer, s.v.Buffer,
                _b1, 1f - _b1, _b2, 1f - _b2, LearningRate, _eps, invBc1, invBc2,
                coupledWd: 0f, decoupledFactor: decoupled);
        }
    });
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
