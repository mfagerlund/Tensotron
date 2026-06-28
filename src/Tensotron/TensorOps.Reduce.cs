namespace Tensotron;

public static partial class TensorOps
{
    // Build (reduceMask, keepShape, finalShape, reducedCount) from a dims spec.
    private static (int[] mask, Shape keepShape, Shape finalShape, int count) ReduceMeta(
        Shape shape, int[]? dims, bool keepdim)
    {
        int r = shape.Rank;
        var mask = new int[r];
        if (dims == null)
            for (int i = 0; i < r; i++) mask[i] = 1;
        else
            foreach (var d in dims)
            {
                int a = d < 0 ? d + r : d;
                if (a < 0 || a >= r)
                    throw new ArgumentOutOfRangeException(nameof(dims),
                        $"Reduce dim {d} is out of range for rank-{r} tensor {shape}.");
                mask[a] = 1;
            }

        var keepDims = (int[])shape.Dims.Clone();
        int count = 1;
        for (int i = 0; i < r; i++)
            if (mask[i] == 1) { count *= shape.Dims[i]; keepDims[i] = 1; }

        var keepShape = new Shape(keepDims);
        Shape finalShape;
        if (keepdim) finalShape = keepShape;
        else
        {
            var fd = new List<int>();
            for (int i = 0; i < r; i++) if (mask[i] == 0) fd.Add(shape.Dims[i]);
            finalShape = fd.Count == 0 ? new Shape() : new Shape(fd.ToArray());
        }
        return (mask, keepShape, finalShape, count);
    }

    /// <summary>Reduce (max/min/prod) returning the keepdim-shaped result, no grad.</summary>
    private static Tensor ReduceKeep<TR>(Tensor x, int[] mask, Shape keepShape) where TR : struct, IReduceOp
    {
        var buf = Runtime.Allocate(keepShape.Size);
        Runtime.LaunchReduce<TR>(x.Buffer, buf, x.Shape.Dims, x.Shape.Strides, keepShape.Dims, mask);
        return new Tensor(keepShape, buf);
    }

    public static Tensor Mean(Tensor x, int[]? dims = null, bool keepdim = false)
    {
        var (_, _, _, count) = ReduceMeta(x.Shape, dims, keepdim);
        return Mul(Sum(x, dims, keepdim), Scalar(1f / count));
    }

    public static Tensor Max(Tensor x, int[]? dims = null, bool keepdim = false)
        => MaxMin<MaxReduce>("Max", x, dims, keepdim);

    public static Tensor Min(Tensor x, int[]? dims = null, bool keepdim = false)
        => MaxMin<MinReduce>("Min", x, dims, keepdim);

    private static Tensor MaxMin<TR>(string name, Tensor x, int[]? dims, bool keepdim) where TR : struct, IReduceOp
    {
        var (mask, keepShape, finalShape, _) = ReduceMeta(x.Shape, dims, keepdim);
        var keep = ReduceKeep<TR>(x, mask, keepShape);
        var result = finalShape.Equals(keepShape) ? keep : new Tensor(finalShape, keep.Buffer);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            bool isMax = typeof(TR) == typeof(MaxReduce);
            result.Node = new GradNode(name, new[] { x }, g =>
            {
                // Route the gradient to the single first-winning element per group
                // (torch.max(dim).values semantics — not split across ties).
                var gk = g.Shape.Equals(keepShape) ? g : g.Reshape(keepShape.Dims);
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchReduceArgGrad(x.Buffer, gk.Buffer, gx.Buffer,
                    x.Shape.Dims, x.Shape.Strides, keepShape.Dims, mask, isMax);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    public static Tensor Prod(Tensor x, int[]? dims = null, bool keepdim = false)
    {
        var (mask, keepShape, finalShape, _) = ReduceMeta(x.Shape, dims, keepdim);
        var keep = ReduceKeep<ProdReduce>(x, mask, keepShape);
        var result = finalShape.Equals(keepShape) ? keep : new Tensor(finalShape, keep.Buffer);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("Prod", new[] { x }, g =>
            {
                // ∂x_i = g * (product of the OTHER elements in the group). Computed via
                // an explicit skip-self product so zeros don't blow up (g*prod/x_i).
                var gk = g.Shape.Equals(keepShape) ? g : g.Reshape(keepShape.Dims);
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchProdGrad(x.Buffer, gk.Buffer, gx.Buffer,
                    x.Shape.Dims, x.Shape.Strides, keepShape.Strides, mask);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    public static Tensor Var(Tensor x, int[]? dims = null, bool keepdim = false, bool unbiased = true)
    {
        var (_, _, _, count) = ReduceMeta(x.Shape, dims, keepdim);
        var mean = Mean(x, dims, keepdim: true);
        var centered = Sub(x, mean);
        var sq = Square(centered);
        float denom = unbiased ? count - 1 : count;
        return Mul(Sum(sq, dims, keepdim), Scalar(1f / denom));
    }

    public static Tensor Std(Tensor x, int[]? dims = null, bool keepdim = false, bool unbiased = true)
        => Sqrt(Var(x, dims, keepdim, unbiased));

    public static Tensor LogSumExp(Tensor x, int dim, bool keepdim = false)
    {
        var mK = Max(x, new[] { dim }, keepdim: true).Detach();
        var sK = Sum(Exp(Sub(x, mK)), new[] { dim }, keepdim: true);
        var lseK = Add(Log(sK), mK);
        if (keepdim) return lseK;
        var (_, _, finalShape, _) = ReduceMeta(x.Shape, new[] { dim }, keepdim: false);
        return lseK.Reshape(finalShape.Dims);
    }

    public static Tensor Softmax(Tensor x, int dim)
    {
        var mK = Max(x, new[] { dim }, keepdim: true).Detach();
        var e = Exp(Sub(x, mK));
        var s = Sum(e, new[] { dim }, keepdim: true);
        return Div(e, s);
    }

    public static Tensor LogSoftmax(Tensor x, int dim)
        => Sub(x, LogSumExp(x, dim, keepdim: true));

    // ---- argmax / argmin (no grad) ----

    public static Tensor Argmax(Tensor x, int dim, bool keepdim = false) => Arg(x, dim, keepdim, isMax: true);
    public static Tensor Argmin(Tensor x, int dim, bool keepdim = false) => Arg(x, dim, keepdim, isMax: false);

    private static Tensor Arg(Tensor x, int dim, bool keepdim, bool isMax)
    {
        var (mask, keepShape, finalShape, _) = ReduceMeta(x.Shape, new[] { dim }, keepdim);
        var buf = Runtime.Allocate(keepShape.Size);
        Runtime.LaunchReduceArg(x.Buffer, buf, x.Shape.Dims, x.Shape.Strides, keepShape.Dims, mask, isMax);
        return new Tensor(finalShape, buf);
    }
}

public sealed partial class Tensor
{
    public Tensor Mean(int[]? dims = null, bool keepdim = false) => TensorOps.Mean(this, dims, keepdim);
    public Tensor Max(int[]? dims = null, bool keepdim = false) => TensorOps.Max(this, dims, keepdim);
    public Tensor Min(int[]? dims = null, bool keepdim = false) => TensorOps.Min(this, dims, keepdim);
    public Tensor Prod(int[]? dims = null, bool keepdim = false) => TensorOps.Prod(this, dims, keepdim);
    public Tensor Var(int[]? dims = null, bool keepdim = false, bool unbiased = true) => TensorOps.Var(this, dims, keepdim, unbiased);
    public Tensor Std(int[]? dims = null, bool keepdim = false, bool unbiased = true) => TensorOps.Std(this, dims, keepdim, unbiased);
    public Tensor Softmax(int dim) => TensorOps.Softmax(this, dim);
    public Tensor LogSoftmax(int dim) => TensorOps.LogSoftmax(this, dim);
    public Tensor LogSumExp(int dim, bool keepdim = false) => TensorOps.LogSumExp(this, dim, keepdim);
    public Tensor Argmax(int dim, bool keepdim = false) => TensorOps.Argmax(this, dim, keepdim);
    public Tensor Argmin(int dim, bool keepdim = false) => TensorOps.Argmin(this, dim, keepdim);
}
