using ILGPU.Algorithms;

namespace Tensotron;

/// <summary>
/// Struct-generic unary op. Forward maps x->y; Backward computes the local
/// gradient given x, y (=Forward(x)) and the upstream grad gy. One forward and
/// one backward kernel are generic over this struct so every op inlines — adding
/// a unary op is just a struct + a fixture case (see README: the test collapse).
/// </summary>
public interface IUnaryOp
{
    float Forward(float x);
    float Backward(float x, float y, float gy);
}

public readonly struct NegOp : IUnaryOp
{
    public float Forward(float x) => -x;
    public float Backward(float x, float y, float gy) => -gy;
}

public readonly struct AbsOp : IUnaryOp
{
    public float Forward(float x) => XMath.Abs(x);
    public float Backward(float x, float y, float gy) => gy * (x > 0f ? 1f : (x < 0f ? -1f : 0f));
}

public readonly struct SignOp : IUnaryOp
{
    public float Forward(float x) => x > 0f ? 1f : (x < 0f ? -1f : 0f);
    public float Backward(float x, float y, float gy) => 0f;
}

public readonly struct ReciprocalOp : IUnaryOp
{
    public float Forward(float x) => 1f / x;
    public float Backward(float x, float y, float gy) => -gy * y * y;
}

public readonly struct SquareOp : IUnaryOp
{
    public float Forward(float x) => x * x;
    public float Backward(float x, float y, float gy) => gy * 2f * x;
}

public readonly struct SqrtOp : IUnaryOp
{
    public float Forward(float x) => XMath.Sqrt(x);
    public float Backward(float x, float y, float gy) => gy * 0.5f / y;
}

public readonly struct RsqrtOp : IUnaryOp
{
    public float Forward(float x) => 1f / XMath.Sqrt(x);
    public float Backward(float x, float y, float gy) => -0.5f * gy * y / x;
}

public readonly struct ExpOp : IUnaryOp
{
    public float Forward(float x) => XMath.Exp(x);
    public float Backward(float x, float y, float gy) => gy * y;
}

public readonly struct LogOp : IUnaryOp
{
    public float Forward(float x) => XMath.Log(x);
    public float Backward(float x, float y, float gy) => gy / x;
}

public readonly struct Log1pOp : IUnaryOp
{
    public float Forward(float x) => XMath.Log(1f + x);
    public float Backward(float x, float y, float gy) => gy / (1f + x);
}

public readonly struct SinOp : IUnaryOp
{
    public float Forward(float x) => XMath.Sin(x);
    public float Backward(float x, float y, float gy) => gy * XMath.Cos(x);
}

public readonly struct CosOp : IUnaryOp
{
    public float Forward(float x) => XMath.Cos(x);
    public float Backward(float x, float y, float gy) => -gy * XMath.Sin(x);
}

public readonly struct TanhOp : IUnaryOp
{
    public float Forward(float x) => XMath.Tanh(x);
    public float Backward(float x, float y, float gy) => gy * (1f - y * y);
}

public readonly struct SigmoidOp : IUnaryOp
{
    public float Forward(float x) => 1f / (1f + XMath.Exp(-x));
    public float Backward(float x, float y, float gy) => gy * y * (1f - y);
}

public readonly struct ReluOp : IUnaryOp
{
    public float Forward(float x) => x > 0f ? x : 0f;
    public float Backward(float x, float y, float gy) => x > 0f ? gy : 0f;
}

/// <summary>LeakyReLU (torch.nn.functional.leaky_relu). Negative branch scales by slope; the
/// boundary x==0 takes the negative-slope gradient, matching torch (which branches on x > 0).</summary>
public readonly struct LeakyReluOp : IUnaryOp
{
    private readonly float _slope;
    public LeakyReluOp(float slope) => _slope = slope;
    public float Forward(float x) => x > 0f ? x : _slope * x;
    public float Backward(float x, float y, float gy) => gy * (x > 0f ? 1f : _slope);
}

/// <summary>ELU (torch.nn.functional.elu). Negative branch is alpha·(eˣ−1) with derivative
/// alpha·eˣ = y + alpha; at x==0 this is alpha (1.0 for the default), matching torch.</summary>
public readonly struct EluOp : IUnaryOp
{
    private readonly float _alpha;
    public EluOp(float alpha) => _alpha = alpha;
    public float Forward(float x) => x > 0f ? x : _alpha * (XMath.Exp(x) - 1f);
    public float Backward(float x, float y, float gy) => x > 0f ? gy : gy * (y + _alpha);
}

/// <summary>Clamp (torch.clamp). Gradient passes through across the whole CLOSED interval
/// [min, max] and is zero strictly outside — composing from Maximum/Minimum instead would
/// tie-split the gradient to 0.5 at the bounds.</summary>
public readonly struct ClampOp : IUnaryOp
{
    private readonly float _min, _max;
    public ClampOp(float min, float max) { _min = min; _max = max; }
    public float Forward(float x) => x < _min ? _min : (x > _max ? _max : x);
    public float Backward(float x, float y, float gy) => (x >= _min && x <= _max) ? gy : 0f;
}

/// <summary>GELU, tanh approximation (matches torch gelu(approximate='tanh')).</summary>
public readonly struct GeluOp : IUnaryOp
{
    private const float K = 0.7978845608028654f; // sqrt(2/pi)
    private const float A = 0.044715f;

    public float Forward(float x)
    {
        float u = K * (x + A * x * x * x);
        return 0.5f * x * (1f + XMath.Tanh(u));
    }

    public float Backward(float x, float y, float gy)
    {
        float u = K * (x + A * x * x * x);
        float tu = XMath.Tanh(u);
        float up = K * (1f + 3f * A * x * x);
        float gp = 0.5f * (1f + tu) + 0.5f * x * (1f - tu * tu) * up;
        return gy * gp;
    }
}

public readonly struct SoftplusOp : IUnaryOp
{
    public float Forward(float x) => x > 20f ? x : XMath.Log(1f + XMath.Exp(x));
    public float Backward(float x, float y, float gy) => gy / (1f + XMath.Exp(-x));
}
