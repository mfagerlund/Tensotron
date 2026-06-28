namespace Tensotron;

public static partial class TensorOps
{
    // ---------------- permute / transpose ----------------

    /// <summary>
    /// Reorder axes. Output is materialized contiguous via a strided gather, so the
    /// rest of the library keeps assuming contiguous storage. Backward applies the
    /// inverse permutation to the gradient.
    /// </summary>
    private static Tensor PermuteRaw(Tensor x, int[] perm)
    {
        int r = x.Shape.Rank;
        var outDims = new int[r];
        var inStrides = new int[r]; // input stride for each OUTPUT axis
        for (int j = 0; j < r; j++)
        {
            outDims[j] = x.Shape.Dims[perm[j]];
            inStrides[j] = x.Shape.Strides[perm[j]];
        }

        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchStridedCopy(x.Buffer, outBuf, outDims, inStrides);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            var inv = new int[r];
            for (int j = 0; j < r; j++) inv[perm[j]] = j;
            result.Node = new GradNode("Permute", new[] { x }, g => x.AddGrad(PermuteRaw(g, inv)));
        }
        return result;
    }

    public static Tensor Permute(Tensor x, params int[] perm)
    {
        int r = x.Shape.Rank;
        if (perm.Length != r)
            throw new InvalidOperationException($"Permute expects {r} axes, got {perm.Length}.");
        var p = new int[r];
        var seen = new bool[r];
        for (int i = 0; i < r; i++)
        {
            int ax = perm[i] < 0 ? perm[i] + r : perm[i];
            if (ax < 0 || ax >= r)
                throw new InvalidOperationException($"Permute axis {perm[i]} out of range for rank {r}.");
            if (seen[ax])
                throw new InvalidOperationException($"Permute has duplicate axis {ax} in [{string.Join(",", perm)}].");
            seen[ax] = true;
            p[i] = ax;
        }
        return PermuteRaw(x, p);
    }

    public static Tensor Transpose(Tensor x, int dim0, int dim1)
    {
        int r = x.Shape.Rank;
        int d0 = dim0 < 0 ? dim0 + r : dim0;
        int d1 = dim1 < 0 ? dim1 + r : dim1;
        if (d0 < 0 || d0 >= r || d1 < 0 || d1 >= r)
            throw new InvalidOperationException($"Transpose dims ({dim0},{dim1}) out of range for rank {r}.");
        var perm = new int[r];
        for (int i = 0; i < r; i++) perm[i] = i;
        (perm[d0], perm[d1]) = (perm[d1], perm[d0]);
        return PermuteRaw(x, perm);
    }

    /// <summary>2D transpose (PyTorch's <c>.t()</c>).</summary>
    public static Tensor T(Tensor x)
    {
        if (x.Rank != 2)
            throw new InvalidOperationException($"T() expects a 2D tensor, got rank {x.Rank}.");
        return Transpose(x, 0, 1);
    }
}

public sealed partial class Tensor
{
    public Tensor Permute(params int[] perm) => TensorOps.Permute(this, perm);
    public Tensor Transpose(int dim0, int dim1) => TensorOps.Transpose(this, dim0, dim1);
    public Tensor T() => TensorOps.T(this);
}
