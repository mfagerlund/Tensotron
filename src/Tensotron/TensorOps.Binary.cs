namespace Tensotron;

public static partial class TensorOps
{
    // Rank-0 so scalar ops never change the rank of the other operand (matches torch).
    internal static Tensor Scalar(float v) => Tensor.FromShaped(new[] { v }, Array.Empty<int>());

    // ---- div / pow ----

    public static Tensor Div(Tensor a, Tensor b) => Binary<DivOp>("Div", a, b, DivBackward);

    private static void DivBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(Div(g, b), a.Shape));
        // ∂b = -g * a / b² = -g * res / b
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Neg(Mul(g, Div(res, b))), b.Shape));
    }

    public static Tensor Pow(Tensor a, Tensor b) => Binary<PowOp>("Pow", a, b, PowBackward);

    private static void PowBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        // ∂a = g * b * a^(b-1) = g * b * res / a
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(Mul(Mul(g, b), Div(res, a)), a.Shape));
        // ∂b = g * res * log(a)
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Mul(Mul(g, res), Log(a)), b.Shape));
    }

    // ---- maximum / minimum (grad to the selected side; ties split 0.5, matching torch) ----

    public static Tensor Maximum(Tensor a, Tensor b) => Binary<MaximumOp>("Maximum", a, b, MaxBackward);
    public static Tensor Minimum(Tensor a, Tensor b) => Binary<MinimumOp>("Minimum", a, b, MinBackward);

    private static void MaxBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(Mul(g, SelectMask(a, b, greater: true)), a.Shape));
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Mul(g, SelectMask(b, a, greater: true)), b.Shape));
    }

    private static void MinBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(Mul(g, SelectMask(a, b, greater: false)), a.Shape));
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Mul(g, SelectMask(b, a, greater: false)), b.Shape));
    }

    // mask for "x is the chosen one vs y": 1 where x wins, 0.5 on tie, 0 otherwise.
    private static Tensor SelectMask(Tensor x, Tensor y, bool greater)
    {
        var win = greater ? Gt(x, y) : Lt(x, y);
        return Add(win, Mul(Eq(x, y), Scalar(0.5f)));
    }

    // ---- comparisons (no grad) ----

    public static Tensor Gt(Tensor a, Tensor b) => BinaryNoGrad<GtOp>(a, b);
    public static Tensor Ge(Tensor a, Tensor b) => BinaryNoGrad<GeOp>(a, b);
    public static Tensor Lt(Tensor a, Tensor b) => BinaryNoGrad<LtOp>(a, b);
    public static Tensor Le(Tensor a, Tensor b) => BinaryNoGrad<LeOp>(a, b);
    public static Tensor Eq(Tensor a, Tensor b) => BinaryNoGrad<EqOp>(a, b);
    public static Tensor Ne(Tensor a, Tensor b) => BinaryNoGrad<NeOp>(a, b);

    // ---- parameterized activations, composed from the above ----

    // Clamp passes the gradient through the whole closed interval [min,max], matching torch; a
    // Minimum(Maximum(...)) composite would tie-split the gradient to 0.5 at the bounds, so Clamp
    // is a dedicated op.
    public static Tensor Clamp(Tensor x, float min, float max) => UnaryP("Clamp", new ClampOp(min, max), x);

    // LeakyRelu and Elu follow torch's exact branch at the kink: the negative side is slope*x (so
    // slope > 1 is not max(x, slope*x)) and the gradient at x==0 takes a single branch, not a 0.5
    // tie-split. Each is a dedicated unary op because a Maximum/Minimum composite can't express that.
    public static Tensor LeakyRelu(Tensor x, float slope = 0.01f) => UnaryP("LeakyRelu", new LeakyReluOp(slope), x);

    public static Tensor Elu(Tensor x, float alpha = 1f) => UnaryP("Elu", new EluOp(alpha), x);
}

public sealed partial class Tensor
{
    public static Tensor operator /(Tensor a, Tensor b) => TensorOps.Div(a, b);

    public static Tensor operator +(Tensor a, float s) => TensorOps.Add(a, TensorOps.Scalar(s));
    public static Tensor operator +(float s, Tensor a) => TensorOps.Add(a, TensorOps.Scalar(s));
    public static Tensor operator -(Tensor a, float s) => TensorOps.Sub(a, TensorOps.Scalar(s));
    public static Tensor operator -(float s, Tensor a) => TensorOps.Sub(TensorOps.Scalar(s), a);
    public static Tensor operator *(Tensor a, float s) => TensorOps.Mul(a, TensorOps.Scalar(s));
    public static Tensor operator *(float s, Tensor a) => TensorOps.Mul(a, TensorOps.Scalar(s));
    public static Tensor operator /(Tensor a, float s) => TensorOps.Div(a, TensorOps.Scalar(s));
    public static Tensor operator /(float s, Tensor a) => TensorOps.Div(TensorOps.Scalar(s), a);

    // Comparison operators return 0/1 masks (no grad), matching torch's `a > b`.
    // == / != are intentionally NOT overloaded: returning a Tensor would break C#
    // reference/null checks. Use .Eq()/.Ne() for elementwise equality.
    public static Tensor operator >(Tensor a, Tensor b) => TensorOps.Gt(a, b);
    public static Tensor operator <(Tensor a, Tensor b) => TensorOps.Lt(a, b);
    public static Tensor operator >=(Tensor a, Tensor b) => TensorOps.Ge(a, b);
    public static Tensor operator <=(Tensor a, Tensor b) => TensorOps.Le(a, b);

    public static Tensor operator >(Tensor a, float s) => TensorOps.Gt(a, TensorOps.Scalar(s));
    public static Tensor operator <(Tensor a, float s) => TensorOps.Lt(a, TensorOps.Scalar(s));
    public static Tensor operator >=(Tensor a, float s) => TensorOps.Ge(a, TensorOps.Scalar(s));
    public static Tensor operator <=(Tensor a, float s) => TensorOps.Le(a, TensorOps.Scalar(s));

    public static Tensor operator >(float s, Tensor a) => TensorOps.Gt(TensorOps.Scalar(s), a);
    public static Tensor operator <(float s, Tensor a) => TensorOps.Lt(TensorOps.Scalar(s), a);
    public static Tensor operator >=(float s, Tensor a) => TensorOps.Ge(TensorOps.Scalar(s), a);
    public static Tensor operator <=(float s, Tensor a) => TensorOps.Le(TensorOps.Scalar(s), a);

    public Tensor Eq(Tensor other) => TensorOps.Eq(this, other);
    public Tensor Ne(Tensor other) => TensorOps.Ne(this, other);

    public Tensor Pow(Tensor exponent) => TensorOps.Pow(this, exponent);
    public Tensor Pow(float exponent) => TensorOps.Pow(this, TensorOps.Scalar(exponent));
    public Tensor Maximum(Tensor other) => TensorOps.Maximum(this, other);
    public Tensor Minimum(Tensor other) => TensorOps.Minimum(this, other);
    public Tensor Clamp(float min, float max) => TensorOps.Clamp(this, min, max);
    public Tensor Div(Tensor other) => TensorOps.Div(this, other);
}
