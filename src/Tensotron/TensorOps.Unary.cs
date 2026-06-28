namespace Tensotron;

public static partial class TensorOps
{
    private static Tensor Unary<TOp>(string name, Tensor x) where TOp : struct, IUnaryOp
    {
        var outBuf = Runtime.Allocate(x.Shape.Size);
        Runtime.LaunchUnaryFwd<TOp>(x.Buffer, outBuf);
        var result = new Tensor(x.Shape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            var y = result;
            result.Node = new GradNode(name, new[] { x }, g =>
            {
                var gxBuf = Runtime.Allocate(x.Shape.Size);
                Runtime.LaunchUnaryBwd<TOp>(x.Buffer, y.Buffer, g.Buffer, gxBuf);
                x.AddGrad(new Tensor(x.Shape, gxBuf));
            });
        }
        return result;
    }

    // Parameterized variant: the op instance (carrying e.g. slope/alpha) is passed into the
    // kernel by value, so a single compiled kernel per op type serves all parameter values.
    private static Tensor UnaryP<TOp>(string name, TOp op, Tensor x) where TOp : unmanaged, IUnaryOp
    {
        var outBuf = Runtime.Allocate(x.Shape.Size);
        Runtime.LaunchUnaryFwdP(op, x.Buffer, outBuf);
        var result = new Tensor(x.Shape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            var y = result;
            result.Node = new GradNode(name, new[] { x }, g =>
            {
                var gxBuf = Runtime.Allocate(x.Shape.Size);
                Runtime.LaunchUnaryBwdP(op, x.Buffer, y.Buffer, g.Buffer, gxBuf);
                x.AddGrad(new Tensor(x.Shape, gxBuf));
            });
        }
        return result;
    }

    public static Tensor Neg(Tensor x) => Unary<NegOp>("Neg", x);
    public static Tensor Abs(Tensor x) => Unary<AbsOp>("Abs", x);
    public static Tensor Sign(Tensor x) => Unary<SignOp>("Sign", x);
    public static Tensor Reciprocal(Tensor x) => Unary<ReciprocalOp>("Reciprocal", x);
    public static Tensor Square(Tensor x) => Unary<SquareOp>("Square", x);
    public static Tensor Sqrt(Tensor x) => Unary<SqrtOp>("Sqrt", x);
    public static Tensor Rsqrt(Tensor x) => Unary<RsqrtOp>("Rsqrt", x);
    public static Tensor Exp(Tensor x) => Unary<ExpOp>("Exp", x);
    public static Tensor Log(Tensor x) => Unary<LogOp>("Log", x);
    public static Tensor Log1p(Tensor x) => Unary<Log1pOp>("Log1p", x);
    public static Tensor Sin(Tensor x) => Unary<SinOp>("Sin", x);
    public static Tensor Cos(Tensor x) => Unary<CosOp>("Cos", x);
    public static Tensor Tanh(Tensor x) => Unary<TanhOp>("Tanh", x);
    public static Tensor Sigmoid(Tensor x) => Unary<SigmoidOp>("Sigmoid", x);
    public static Tensor Relu(Tensor x) => Unary<ReluOp>("Relu", x);
    public static Tensor Gelu(Tensor x) => Unary<GeluOp>("Gelu", x);
    public static Tensor Softplus(Tensor x) => Unary<SoftplusOp>("Softplus", x);
}

public sealed partial class Tensor
{
    public Tensor Neg() => TensorOps.Neg(this);
    public static Tensor operator -(Tensor x) => TensorOps.Neg(x);
    public Tensor Abs() => TensorOps.Abs(this);
    public Tensor Sign() => TensorOps.Sign(this);
    public Tensor Reciprocal() => TensorOps.Reciprocal(this);
    public Tensor Square() => TensorOps.Square(this);
    public Tensor Sqrt() => TensorOps.Sqrt(this);
    public Tensor Rsqrt() => TensorOps.Rsqrt(this);
    public Tensor Exp() => TensorOps.Exp(this);
    public Tensor Log() => TensorOps.Log(this);
    public Tensor Log1p() => TensorOps.Log1p(this);
    public Tensor Sin() => TensorOps.Sin(this);
    public Tensor Cos() => TensorOps.Cos(this);
    public Tensor Tanh() => TensorOps.Tanh(this);
    public Tensor Sigmoid() => TensorOps.Sigmoid(this);
    public Tensor Relu() => TensorOps.Relu(this);
    public Tensor Gelu() => TensorOps.Gelu(this);
    public Tensor Softplus() => TensorOps.Softplus(this);
}
