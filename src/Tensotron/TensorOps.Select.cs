namespace Tensotron;

public static partial class TensorOps
{
    // Selection is a true branch (the ternary Select kernel), not an arithmetic blend, so a
    // NaN/Inf in the unselected operand is discarded rather than poisoning the result via 0*NaN
    // — matching torch.where / masked_fill. `cond`/`mask` are 0/1 float tensors with no gradient
    // (gradient flows only to the selected data tensors).

    /// <summary>
    /// Elementwise select: <c>cond ? a : b</c>. <paramref name="cond"/> is a 0/1 mask
    /// (e.g. from a comparison). Broadcasts like torch.where.
    /// </summary>
    public static Tensor Where(Tensor cond, Tensor a, Tensor b)
    {
        // The condition is non-differentiable (torch.where: gradient flows only to a/b).
        // Detach so a RequiresGrad cond never leaks gradient into the mask.
        var c = cond.Detach();
        var outDims = BroadcastDims(c.Shape, a.Shape, b.Shape);
        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchSelect(c.Buffer, a.Buffer, b.Buffer, outBuf, outDims,
            BroadcastStridesTo(c.Shape, outDims),
            BroadcastStridesTo(a.Shape, outDims),
            BroadcastStridesTo(b.Shape, outDims));
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(a) || Tensor.NeedsGrad(b)))
            result.Node = new GradNode("Where", new[] { a, b }, g =>
            {
                // Route the whole gradient to the selected side; g is finite and c is 0/1, so the
                // mask multiply here can't reintroduce a NaN the forward select removed.
                if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(Mul(g, c), a.Shape));
                if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Mul(g, Sub(Scalar(1f), c)), b.Shape));
            });
        return result;
    }

    /// <summary>Replace elements where <paramref name="mask"/> is 1 with <paramref name="value"/>.</summary>
    public static Tensor MaskedFill(Tensor x, Tensor mask, float value)
        // mask ? value : x — a true select, so a NaN/Inf at a position being replaced is dropped
        // (torch.masked_fill); gradient flows to x only (the fill value is a constant).
        => Where(mask, Scalar(value), x);
}

public sealed partial class Tensor
{
    public Tensor MaskedFill(Tensor mask, float value) => TensorOps.MaskedFill(this, mask, value);
}
