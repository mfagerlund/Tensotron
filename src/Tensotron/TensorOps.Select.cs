namespace Tensotron;

public static partial class TensorOps
{
    // Selection ops compose from elementwise primitives, so their gradients fall out
    // of the existing backward closures. `cond`/`mask` are 0/1 float tensors with no
    // gradient (matching torch: gradient flows only to the selected data tensors).

    /// <summary>
    /// Elementwise select: <c>cond ? a : b</c>. <paramref name="cond"/> is a 0/1 mask
    /// (e.g. from a comparison). Broadcasts like torch.where.
    /// </summary>
    public static Tensor Where(Tensor cond, Tensor a, Tensor b)
    {
        // The condition is non-differentiable (torch.where: gradient flows only to a/b).
        // Detach so a RequiresGrad cond never leaks gradient into the mask.
        var c = cond.Detach();
        return Add(Mul(c, a), Mul(Sub(Scalar(1f), c), b));
    }

    /// <summary>Replace elements where <paramref name="mask"/> is 1 with <paramref name="value"/>.</summary>
    public static Tensor MaskedFill(Tensor x, Tensor mask, float value)
    {
        // The mask is non-differentiable (torch.masked_fill); detach it.
        var m = mask.Detach();
        return Add(Mul(x, Sub(Scalar(1f), m)), Mul(m, Scalar(value)));
    }
}

public sealed partial class Tensor
{
    public Tensor MaskedFill(Tensor mask, float value) => TensorOps.MaskedFill(this, mask, value);
}
