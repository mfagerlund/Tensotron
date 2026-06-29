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

/// <summary>
/// GELU, exact (erf) form — matches torch's DEFAULT <c>nn.GELU()</c> / <c>F.gelu(approximate='none')</c>.
/// This is what <see cref="TensorOps.Gelu"/> uses by default; the tanh approximation
/// (<see cref="GeluOp"/>) is opt-in via <c>approximateTanh: true</c>.
/// </summary>
public readonly struct GeluErfOp : IUnaryOp
{
    private const float InvSqrt2 = 0.7071067811865476f;    // 1/sqrt(2)
    private const float InvSqrt2Pi = 0.3989422804014327f;  // 1/sqrt(2*pi)

    public float Forward(float x) => 0.5f * x * (1f + Erf(x * InvSqrt2));

    public float Backward(float x, float y, float gy)
    {
        float cdf = 0.5f * (1f + Erf(x * InvSqrt2));
        float pdf = InvSqrt2Pi * XMath.Exp(-0.5f * x * x);   // exact N(0,1) density
        return gy * (cdf + x * pdf);
    }

    // erf via Abramowitz & Stegun 7.1.26: |abs err| <= 1.5e-7 — comfortably inside the 2e-4 parity
    // gate, and identical on the GPU and managed-CPU paths (both execute this same struct).
    private static float Erf(float x)
    {
        float s = x < 0f ? -1f : 1f;
        float ax = x < 0f ? -x : x;
        float t = 1f / (1f + 0.3275911f * ax);
        float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        return s * (1f - poly * XMath.Exp(-ax * ax));
    }
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
