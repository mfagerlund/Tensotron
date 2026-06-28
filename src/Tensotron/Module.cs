namespace Tensotron;

/// <summary>
/// Base class for composable network layers (torch.nn.Module). Subclasses expose their
/// own learnable parameters via <see cref="OwnParameters"/> and any sub-modules via
/// <see cref="Children"/>; parameter collection and train/eval propagation are handled
/// here so state_dict / optimizer wiring is uniform.
/// </summary>
public abstract class Module
{
    public bool Training { get; private set; } = true;

    protected virtual IEnumerable<Module> Children => Array.Empty<Module>();
    protected virtual IEnumerable<(string name, Tensor param)> OwnParameters() => Array.Empty<(string, Tensor)>();
    protected virtual IEnumerable<(string name, Module child)> NamedChildren =>
        Children.Select((c, i) => (i.ToString(), c));

    public void Train(bool mode = true)
    {
        Training = mode;
        foreach (var c in Children) c.Train(mode);
    }

    public void Eval() => Train(false);

    /// <summary>All learnable parameters with dotted names (torch state_dict keys).</summary>
    public IEnumerable<(string name, Tensor param)> NamedParameters(string prefix = "")
    {
        foreach (var (n, p) in OwnParameters()) yield return (prefix + n, p);
        foreach (var (cn, c) in NamedChildren)
            foreach (var kv in c.NamedParameters($"{prefix}{cn}."))
                yield return kv;
    }

    public IEnumerable<Tensor> Parameters() => NamedParameters().Select(x => x.param);

    public void ZeroGrad()
    {
        foreach (var p in Parameters()) p.Grad = null;
    }

    public abstract Tensor Forward(Tensor x);

    public Tensor Call(Tensor x) => Forward(x);
}

/// <summary>Runs sub-modules in sequence (torch.nn.Sequential).</summary>
public sealed class Sequential : Module
{
    private readonly Module[] _modules;
    public Sequential(params Module[] modules) => _modules = modules;
    protected override IEnumerable<Module> Children => _modules;
    public override Tensor Forward(Tensor x)
    {
        foreach (var m in _modules) x = m.Forward(x);
        return x;
    }
}

/// <summary>Affine layer y = x·Wᵀ + b (torch.nn.Linear), with torch-default init.</summary>
public sealed class Linear : Module
{
    public Tensor Weight { get; }   // (out, in)
    public Tensor? Bias { get; }    // (out,)

    public Linear(int inFeatures, int outFeatures, bool bias = true)
    {
        Weight = Tensor.Zeros(new Shape(outFeatures, inFeatures)).RequireGrad();
        Init.KaimingUniform_(Weight, a: MathF.Sqrt(5f)); // torch Linear default

        if (bias)
        {
            Bias = Tensor.Zeros(new Shape(outFeatures)).RequireGrad();
            float bound = 1f / MathF.Sqrt(inFeatures);
            Init.Uniform_(Bias, -bound, bound);
        }
    }

    public override Tensor Forward(Tensor x)
    {
        var y = TensorOps.MatMul(x, Weight.T());   // (..., in) @ (in, out)
        return Bias is null ? y : TensorOps.Add(y, Bias);
    }

    protected override IEnumerable<(string, Tensor)> OwnParameters()
    {
        yield return ("weight", Weight);
        if (Bias is not null) yield return ("bias", Bias);
    }
}

/// <summary>Wraps any unary op as a parameter-free module (e.g. for Sequential).</summary>
public sealed class Activation : Module
{
    private readonly Func<Tensor, Tensor> _fn;
    public Activation(Func<Tensor, Tensor> fn) => _fn = fn;
    public override Tensor Forward(Tensor x) => _fn(x);

    public static Activation Relu() => new(TensorOps.Relu);
    public static Activation Tanh() => new(TensorOps.Tanh);
    public static Activation Sigmoid() => new(TensorOps.Sigmoid);
    public static Activation Gelu() => new(TensorOps.Gelu);
}

/// <summary>Inverted dropout (torch.nn.Dropout): active only in training mode.</summary>
public sealed class Dropout : Module
{
    private readonly float _p;
    private Random _rng;

    public Dropout(float p = 0.5f, int seed = 12345)
    {
        if (p < 0f || p >= 1f) throw new InvalidOperationException($"Dropout p must be in [0,1), got {p}.");
        _p = p;
        _rng = new Random(seed);
    }

    public override Tensor Forward(Tensor x)
    {
        if (!Training || _p == 0f) return x;
        float keep = 1f - _p;
        var mask = new float[x.Shape.Size];
        for (int i = 0; i < mask.Length; i++) mask[i] = _rng.NextDouble() < keep ? 1f / keep : 0f;
        var maskT = Tensor.FromShaped(mask, x.Shape.Dims); // no grad: gradient flows through the multiply to x
        return TensorOps.Mul(x, maskT);
    }
}
