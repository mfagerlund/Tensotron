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
            using var scope = Tensor.NewBufferScope(); // recycle this step's scratch buffers
            var g = p.Grad;
            if (_weightDecay != 0f) g = Add(g, Mul(p, Scalar(_weightDecay)));

            if (_momentum != 0f)
            {
                Tensor buf;
                if (!_buf.TryGetValue(p, out buf!))
                    buf = g.Clone(); // first step: buffer initialized to the gradient
                else
                    buf = Add(Mul(buf, Scalar(_momentum)), Mul(g, Scalar(1f - _dampening)));
                _buf[p] = scope.Keep(buf); // momentum buffer persists across steps
                g = _nesterov ? Add(g, Mul(buf, Scalar(_momentum))) : buf;
            }

            Update(p, Sub(p, Mul(g, Scalar(LearningRate))));
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
            using var scope = Tensor.NewBufferScope(); // recycle this step's scratch buffers
            var g = p.Grad;
            if (_weightDecay != 0f) g = Add(g, Mul(p, Scalar(_weightDecay)));

            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape), Tensor.Zeros(p.Shape), 0);
            int t = s.t + 1;
            var m = Add(Mul(s.m, Scalar(_b1)), Mul(g, Scalar(1f - _b1)));
            var v = Add(Mul(s.v, Scalar(_b2)), Mul(Square(g), Scalar(1f - _b2)));
            _state[p] = (scope.Keep(m), scope.Keep(v), t); // moment estimates persist across steps

            float bc1 = 1f - MathF.Pow(_b1, t);
            float bc2 = 1f - MathF.Pow(_b2, t);
            var mhat = Mul(m, Scalar(1f / bc1));
            var vhat = Mul(v, Scalar(1f / bc2));
            var step = Div(mhat, Add(Sqrt(vhat), Scalar(_eps)));
            Update(p, Sub(p, Mul(step, Scalar(LearningRate))));
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
            using var scope = Tensor.NewBufferScope(); // recycle this step's scratch buffers
            var g = p.Grad;

            // decoupled decay: p <- p * (1 - lr*wd) before the Adam step.
            var pd = _weightDecay != 0f ? Mul(p, Scalar(1f - LearningRate * _weightDecay)) : p;

            if (!_state.TryGetValue(p, out var s))
                s = (Tensor.Zeros(p.Shape), Tensor.Zeros(p.Shape), 0);
            int t = s.t + 1;
            var m = Add(Mul(s.m, Scalar(_b1)), Mul(g, Scalar(1f - _b1)));
            var v = Add(Mul(s.v, Scalar(_b2)), Mul(Square(g), Scalar(1f - _b2)));
            _state[p] = (scope.Keep(m), scope.Keep(v), t); // moment estimates persist across steps

            float bc1 = 1f - MathF.Pow(_b1, t);
            float bc2 = 1f - MathF.Pow(_b2, t);
            var mhat = Mul(m, Scalar(1f / bc1));
            var vhat = Mul(v, Scalar(1f / bc2));
            var step = Div(mhat, Add(Sqrt(vhat), Scalar(_eps)));
            Update(p, Sub(pd, Mul(step, Scalar(LearningRate))));
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
            using var scope = Tensor.NewBufferScope(); // recycle this step's scratch buffers
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
            // running stats persist across steps; the rest is scratch the scope reclaims
            _state[p] = (scope.Keep(v),
                         avg is null ? null : (Tensor?)scope.Keep(avg),
                         buf is null ? null : (Tensor?)scope.Keep(buf));
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
    public static float ClipGradNorm(IReadOnlyList<Tensor> parameters, float maxNorm, float eps = 1e-6f)
    {
        using (Tensor.NoGradScope())
        {
            double sumSq = 0;
            foreach (var p in parameters)
                if (p.Grad != null) sumSq += Sum(Square(p.Grad)).Item();
            float total = MathF.Sqrt((float)sumSq);

            float coef = maxNorm / (total + eps);
            if (coef < 1f)
                foreach (var p in parameters)
                    if (p.Grad != null) p.Grad.CopyInPlace(Mul(p.Grad, Scalar(coef)));
            return total;
        }
    }
}
