namespace Tensotron;

public static partial class TensorOps
{
    /// <summary>
    /// Layer normalization over the last <c>normalizedShape.Length</c> axes (torch
    /// F.layer_norm). Uses the biased variance, matching torch. Optional affine gamma/beta
    /// broadcast over the leading (batch) axes. Composed from primitive ops, so backward
    /// to x, gamma and beta is automatic.
    /// </summary>
    public static Tensor LayerNorm(Tensor x, int[] normalizedShape, Tensor? gamma = null, Tensor? beta = null, float eps = 1e-5f)
    {
        int d = normalizedShape.Length;
        int r = x.Rank;
        if (d > r) throw new InvalidOperationException($"LayerNorm normalized_shape rank {d} > input rank {r}.");
        var dims = new int[d];
        for (int i = 0; i < d; i++)
        {
            dims[i] = r - d + i;
            if (x.Shape.Dims[dims[i]] != normalizedShape[i])
                throw new InvalidOperationException($"LayerNorm axis {dims[i]} ({x.Shape.Dims[dims[i]]}) != normalized_shape {normalizedShape[i]}.");
        }

        var mean = Mean(x, dims, keepdim: true);
        var variance = Var(x, dims, keepdim: true, unbiased: false);
        var norm = Div(Sub(x, mean), Sqrt(Add(variance, Scalar(eps))));

        if (gamma is not null) norm = Mul(norm, gamma);
        if (beta is not null) norm = Add(norm, beta);
        return norm;
    }

    // Reshape a per-channel (C,) affine vector to broadcast over a rank-r tensor whose
    // channel axis is 1, i.e. (1, C, 1, ..., 1).
    private static Tensor ChannelView(Tensor perChannel, int rank)
    {
        var dims = new int[rank];
        for (int i = 0; i < rank; i++) dims[i] = 1;
        dims[1] = perChannel.Shape.Size;
        return perChannel.Reshape(dims);
    }

    /// <summary>
    /// Group normalization (torch.nn.functional.group_norm). Splits the C channels into
    /// <paramref name="numGroups"/> groups and normalizes over each group's channels and
    /// all spatial positions (biased var). Optional per-channel affine. Input is (N, C, *).
    /// </summary>
    public static Tensor GroupNorm(Tensor x, int numGroups, Tensor? weight = null, Tensor? bias = null, float eps = 1e-5f)
    {
        if (x.Rank < 2)
            throw new InvalidOperationException($"GroupNorm expects (N, C, *) with rank >= 2 (got rank {x.Rank}).");
        int n = x.Shape.Dims[0];
        int c = x.Shape.Dims[1];
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), $"GroupNorm requires numGroups > 0 (got {numGroups}).");
        if (c % numGroups != 0)
            throw new InvalidOperationException($"GroupNorm: channels {c} not divisible by groups {numGroups}.");
        int inner = x.Shape.Size / (n * numGroups); // (C/G) * spatial

        var view = x.Reshape(n, numGroups, inner);
        var mean = Mean(view, new[] { 2 }, keepdim: true);
        var variance = Var(view, new[] { 2 }, keepdim: true, unbiased: false);
        var norm = Div(Sub(view, mean), Sqrt(Add(variance, Scalar(eps))));
        norm = norm.Reshape(x.Shape.Dims);

        if (weight is not null) norm = Mul(norm, ChannelView(weight, x.Rank));
        if (bias is not null) norm = Add(norm, ChannelView(bias, x.Rank));
        return norm;
    }
}
