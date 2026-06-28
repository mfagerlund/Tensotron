namespace Tensotron;

public static partial class TensorOps
{
    // Indices are host int[] (they never carry gradient and usually originate on the
    // host as labels). Device upload happens inside the launch helpers. Mode 0 ops use
    // a 1D index addressed by the axis coordinate; mode 1 ops use a full-shaped index.

    // ---------------- index_select ----------------

    public static Tensor IndexSelect(Tensor x, int dim, int[] indices)
    {
        int r = x.Rank;
        int d = dim < 0 ? dim + r : dim;
        if (d < 0 || d >= r) throw new InvalidOperationException($"IndexSelect dim {dim} out of range for rank {r}.");
        int axisSize = x.Shape.Dims[d];
        foreach (var ix in indices)
            if (ix < 0 || ix >= axisSize)
                throw new InvalidOperationException($"IndexSelect index {ix} out of range [0,{axisSize}).");

        var outDims = (int[])x.Shape.Dims.Clone();
        outDims[d] = indices.Length;
        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchGatherAxis(x.Buffer, outBuf, indices, outDims, x.Shape.Strides, d, mode: 0);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("IndexSelect", new[] { x }, g =>
            {
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchScatterAddAxis(g.Buffer, gx.Buffer, indices, outDims, x.Shape.Strides, d, mode: 0);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    // ---------------- gather ----------------

    public static Tensor Gather(Tensor x, int dim, int[] index, int[] indexShape)
    {
        int r = x.Rank;
        int d = dim < 0 ? dim + r : dim;
        if (d < 0 || d >= r) throw new InvalidOperationException($"Gather dim {dim} out of range for rank {r}.");
        if (indexShape.Length != r)
            throw new InvalidOperationException($"Gather index rank {indexShape.Length} != input rank {r}.");
        for (int i = 0; i < r; i++)
            if (i != d && indexShape[i] > x.Shape.Dims[i])
                throw new InvalidOperationException($"Gather index dim {i} ({indexShape[i]}) exceeds input ({x.Shape.Dims[i]}).");
        int expectedLen = 1;
        foreach (var s in indexShape) expectedLen *= s;
        if (index.Length != expectedLen)
            throw new InvalidOperationException($"Gather index length {index.Length} != product(indexShape) {expectedLen}.");
        int axisSize = x.Shape.Dims[d];
        foreach (var ix in index)
            if (ix < 0 || ix >= axisSize)
                throw new InvalidOperationException($"Gather index {ix} out of range [0,{axisSize}).");

        var outShape = new Shape((int[])indexShape.Clone());
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchGatherAxis(x.Buffer, outBuf, index, outShape.Dims, x.Shape.Strides, d, mode: 1);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("Gather", new[] { x }, g =>
            {
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchScatterAddAxis(g.Buffer, gx.Buffer, index, outShape.Dims, x.Shape.Strides, d, mode: 1);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    /// <summary>Gather with the index supplied as a (float) tensor — values are rounded
    /// to int. Bridges argmax/argmin output straight into a gather.</summary>
    public static Tensor Gather(Tensor x, int dim, Tensor index)
        => Gather(x, dim, ToIndices(index), index.Shape.Dims);

    // ---------------- scatter_add ----------------

    public static Tensor ScatterAdd(Tensor self, int dim, int[] index, int[] indexShape, Tensor src)
    {
        int r = self.Rank;
        int d = dim < 0 ? dim + r : dim;
        if (d < 0 || d >= r) throw new InvalidOperationException($"ScatterAdd dim {dim} out of range for rank {r}.");
        if (indexShape.Length != r)
            throw new InvalidOperationException($"ScatterAdd index rank {indexShape.Length} != self rank {r}.");
        for (int i = 0; i < r; i++)
            if (i != d && indexShape[i] > self.Shape.Dims[i])
                throw new InvalidOperationException($"ScatterAdd index dim {i} ({indexShape[i]}) exceeds self ({self.Shape.Dims[i]}).");
        if (!src.Shape.Equals(new Shape((int[])indexShape.Clone())))
            throw new InvalidOperationException($"ScatterAdd src shape {src.Shape} != index shape [{string.Join(",", indexShape)}].");
        int expectedLen = 1;
        foreach (var s in indexShape) expectedLen *= s;
        if (index.Length != expectedLen)
            throw new InvalidOperationException($"ScatterAdd index length {index.Length} != product(indexShape) {expectedLen}.");
        int axisSize = self.Shape.Dims[d];
        foreach (var ix in index)
            if (ix < 0 || ix >= axisSize)
                throw new InvalidOperationException($"ScatterAdd index {ix} out of range [0,{axisSize}).");

        // result = self.clone(); then atomically add src into the scattered positions.
        var result = self.Clone();
        Runtime.LaunchScatterAddAxis(src.Buffer, result.Buffer, index, indexShape, self.Shape.Strides, d, mode: 1);

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(self) || Tensor.NeedsGrad(src)))
        {
            result.Node = new GradNode("ScatterAdd", new[] { self, src }, g =>
            {
                // d/d self = identity; d/d src[e] = g at the scattered position (a gather).
                if (Tensor.NeedsGrad(self)) self.AddGrad(ReduceGradToShape(g, self.Shape));
                if (Tensor.NeedsGrad(src))
                {
                    var gsrc = Runtime.Allocate(src.Shape.Size);
                    Runtime.LaunchGatherAxis(g.Buffer, gsrc, index, indexShape, self.Shape.Strides, d, mode: 1);
                    src.AddGrad(new Tensor(src.Shape, gsrc));
                }
            });
        }
        return result;
    }

    public static Tensor ScatterAdd(Tensor self, int dim, Tensor index, Tensor src)
        => ScatterAdd(self, dim, ToIndices(index), index.Shape.Dims, src);

    // ---------------- repeat (tile) ----------------

    public static Tensor Repeat(Tensor x, params int[] sizes)
    {
        int r = sizes.Length;
        if (r < x.Rank) throw new InvalidOperationException($"Repeat sizes rank {r} < input rank {x.Rank}.");
        foreach (var s in sizes)
            if (s < 1) throw new InvalidOperationException($"Repeat sizes must be >= 1 (got {s}).");

        int lead = r - x.Rank;
        var outDims = new int[r];
        var inDims = new int[r];
        var inStrides = new int[r];
        for (int j = 0; j < r; j++)
        {
            int ax = j - lead;
            int inDim = ax >= 0 ? x.Shape.Dims[ax] : 1;
            inDims[j] = inDim;
            inStrides[j] = ax >= 0 ? x.Shape.Strides[ax] : 0;
            outDims[j] = sizes[j] * inDim;
        }

        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchRepeat(x.Buffer, outBuf, outDims, inDims, inStrides);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("Repeat", new[] { x }, g =>
            {
                // Each input element is read by every tile; sum the tiles' gradients back.
                var gx = Tensor.Zeros(x.Shape);
                // gx is stored contiguous with x's strides, so RepeatGrad scatters into it
                // using the same (coord % inDim) mapping the forward read with.
                Runtime.LaunchRepeatGrad(g.Buffer, gx.Buffer, outDims, inDims, inStrides);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    // Convert a float index tensor to host int[]. torch indices are integral by contract;
    // float32 has no integer dtype, so an index tensor holds integral values (e.g. straight
    // from argmax/argmin, which are exact in float32 for indices < 2^24). Reject any non-integral
    // value rather than silently rounding 1.49 -> 1 / 1.51 -> 2.
    private static int[] ToIndices(Tensor index)
    {
        var data = index.ToArray();
        var ix = new int[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            float r = MathF.Round(v);
            if (MathF.Abs(v - r) > 1e-4f)
                throw new InvalidOperationException(
                    $"Index tensor must hold integral values (got {v} at flat position {i}). " +
                    "Float32 has no integer dtype, so indices are integral floats (e.g. from argmax/argmin).");
            ix[i] = (int)r;
        }
        return ix;
    }
}

public sealed partial class Tensor
{
    public Tensor IndexSelect(int dim, int[] indices) => TensorOps.IndexSelect(this, dim, indices);
    public Tensor Gather(int dim, int[] index, int[] indexShape) => TensorOps.Gather(this, dim, index, indexShape);
    public Tensor Gather(int dim, Tensor index) => TensorOps.Gather(this, dim, index);
    public Tensor ScatterAdd(int dim, int[] index, int[] indexShape, Tensor src) => TensorOps.ScatterAdd(this, dim, index, indexShape, src);
    public Tensor ScatterAdd(int dim, Tensor index, Tensor src) => TensorOps.ScatterAdd(this, dim, index, src);
    public Tensor Repeat(params int[] sizes) => TensorOps.Repeat(this, sizes);
}
