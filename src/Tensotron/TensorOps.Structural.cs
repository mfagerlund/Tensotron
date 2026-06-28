namespace Tensotron;

public static partial class TensorOps
{
    // ---------------- reshape family (zero-copy, tracked via Reshape) ----------------

    public static Tensor Squeeze(Tensor x, int? dim = null)
    {
        int r = x.Rank;
        var dims = new List<int>();
        if (dim is null)
        {
            for (int i = 0; i < r; i++) if (x.Shape.Dims[i] != 1) dims.Add(x.Shape.Dims[i]);
        }
        else
        {
            int d = dim.Value < 0 ? dim.Value + r : dim.Value;
            if (d < 0 || d >= r) throw new InvalidOperationException($"Squeeze dim {dim} out of range for rank {r}.");
            for (int i = 0; i < r; i++) if (i != d || x.Shape.Dims[i] != 1) dims.Add(x.Shape.Dims[i]);
        }
        return x.Reshape(dims.ToArray());
    }

    public static Tensor Unsqueeze(Tensor x, int dim)
    {
        int r = x.Rank;
        int d = dim < 0 ? dim + r + 1 : dim;
        if (d < 0 || d > r) throw new InvalidOperationException($"Unsqueeze dim {dim} out of range for rank {r}.");
        var dims = new int[r + 1];
        for (int i = 0, j = 0; i < r + 1; i++)
            dims[i] = i == d ? 1 : x.Shape.Dims[j++];
        return x.Reshape(dims);
    }

    public static Tensor Flatten(Tensor x, int start = 0, int end = -1)
    {
        int r = x.Rank;
        if (r == 0) return x.Reshape(1);
        int s = start < 0 ? start + r : start;
        int e = end < 0 ? end + r : end;
        if (s < 0 || s >= r || e < 0 || e >= r || s > e)
            throw new InvalidOperationException($"Flatten range ({start},{end}) invalid for rank {r}.");
        var dims = new List<int>();
        for (int i = 0; i < s; i++) dims.Add(x.Shape.Dims[i]);
        int prod = 1;
        for (int i = s; i <= e; i++) prod *= x.Shape.Dims[i];
        dims.Add(prod);
        for (int i = e + 1; i < r; i++) dims.Add(x.Shape.Dims[i]);
        return x.Reshape(dims.ToArray());
    }

    // ---------------- expand (broadcast view, materialized) ----------------

    public static Tensor Expand(Tensor x, params int[] sizes)
    {
        int r = sizes.Length;
        if (r < x.Rank) throw new InvalidOperationException($"Expand rank {r} < input rank {x.Rank}.");
        var outDims = new int[r];
        var inStrides = new int[r];
        for (int j = 0; j < r; j++)
        {
            int ax = j - (r - x.Rank);
            int inDim = ax >= 0 ? x.Shape.Dims[ax] : 1;
            int target = sizes[j];
            if (target == -1)
            {
                if (ax < 0) throw new InvalidOperationException("Expand cannot infer (-1) for a new leading axis.");
                target = inDim;
            }
            if (inDim == target)
                inStrides[j] = ax >= 0 ? x.Shape.Strides[ax] : 0;
            else if (inDim == 1)
                inStrides[j] = 0; // broadcast
            else
                throw new InvalidOperationException($"Expand: cannot expand dim {inDim} to {target}.");
            outDims[j] = target;
        }

        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchStridedCopy(x.Buffer, outBuf, outDims, inStrides);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
            result.Node = new GradNode("Expand", new[] { x },
                g => x.AddGrad(ReduceGradToShape(g, x.Shape)));
        return result;
    }

    // ---------------- narrow / slice ----------------

    // Forward-only gather of a contiguous range [start, start+length) along dim.
    private static Tensor NarrowRaw(Tensor x, int dim, int start, int length)
    {
        int r = x.Rank;
        var outDims = (int[])x.Shape.Dims.Clone();
        outDims[dim] = length;
        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        int baseOff = start * x.Shape.Strides[dim];
        Runtime.LaunchStridedCopy(x.Buffer, outBuf, outDims, x.Shape.Strides, baseOff);
        return new Tensor(outShape, outBuf);
    }

    public static Tensor Narrow(Tensor x, int dim, int start, int length)
    {
        int r = x.Rank;
        int d = dim < 0 ? dim + r : dim;
        if (d < 0 || d >= r) throw new InvalidOperationException($"Narrow dim {dim} out of range for rank {r}.");
        if (start < 0 || length < 0 || start + length > x.Shape.Dims[d])
            throw new InvalidOperationException($"Narrow [{start},{start + length}) out of bounds for dim {d} (size {x.Shape.Dims[d]}).");

        var result = NarrowRaw(x, d, start, length);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("Narrow", new[] { x }, g =>
            {
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchScatterAxisRange(g.Buffer, gx.Buffer, g.Shape.Dims, x.Shape.Strides, d, start);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    // ---------------- cat / stack ----------------

    public static Tensor Cat(IReadOnlyList<Tensor> tensors, int dim = 0)
    {
        if (tensors.Count == 0) throw new InvalidOperationException("Cat needs at least one tensor.");
        int r = tensors[0].Rank;
        int d = dim < 0 ? dim + r : dim;
        if (d < 0 || d >= r) throw new InvalidOperationException($"Cat dim {dim} out of range for rank {r}.");

        int catTotal = 0;
        foreach (var t in tensors)
        {
            if (t.Rank != r) throw new InvalidOperationException("Cat: all tensors must share rank.");
            for (int i = 0; i < r; i++)
                if (i != d && t.Shape.Dims[i] != tensors[0].Shape.Dims[i])
                    throw new InvalidOperationException($"Cat: dim {i} mismatch ({t.Shape} vs {tensors[0].Shape}).");
            catTotal += t.Shape.Dims[d];
        }

        var outDims = (int[])tensors[0].Shape.Dims.Clone();
        outDims[d] = catTotal;
        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);

        int offset = 0;
        var offsets = new int[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = offset;
            Runtime.LaunchScatterAxisRange(tensors[i].Buffer, outBuf,
                tensors[i].Shape.Dims, outShape.Strides, d, offset);
            offset += tensors[i].Shape.Dims[d];
        }
        var result = new Tensor(outShape, outBuf);

        bool anyGrad = false;
        foreach (var t in tensors) anyGrad |= Tensor.NeedsGrad(t);
        if (!Tensor.NoGrad && anyGrad)
        {
            var inputs = tensors.ToArray();
            result.Node = new GradNode("Cat", inputs, g =>
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var ti = inputs[i];
                    if (!Tensor.NeedsGrad(ti)) continue;
                    ti.AddGrad(NarrowRaw(g, d, offsets[i], ti.Shape.Dims[d]));
                }
            });
        }
        return result;
    }

    public static Tensor Cat(params Tensor[] tensors) => Cat(tensors, 0);

    public static Tensor Stack(IReadOnlyList<Tensor> tensors, int dim = 0)
    {
        var unsq = new Tensor[tensors.Count];
        for (int i = 0; i < tensors.Count; i++) unsq[i] = Unsqueeze(tensors[i], dim);
        return Cat(unsq, dim);
    }

    // ---------------- chunk / split (views via narrow) ----------------

    public static Tensor[] Chunk(Tensor x, int chunks, int dim = 0)
    {
        if (chunks <= 0) throw new InvalidOperationException($"Chunk requires chunks > 0 (got {chunks}).");
        int r = x.Rank;
        int d = dim < 0 ? dim + r : dim;
        int size = x.Shape.Dims[d];
        // ceil; last chunk may be smaller. Guard against a zero step when size == 0 (empty
        // dim → piece would be 0 → the loop would never advance). Max(1,...) is a no-op for
        // size > 0 and makes an empty dim return no chunks instead of hanging.
        int piece = Math.Max(1, (size + chunks - 1) / chunks);
        var outList = new List<Tensor>();
        for (int start = 0; start < size; start += piece)
            outList.Add(Narrow(x, d, start, Math.Min(piece, size - start)));
        return outList.ToArray();
    }

    public static Tensor[] Split(Tensor x, int splitSize, int dim = 0)
    {
        if (splitSize <= 0) throw new InvalidOperationException($"Split requires splitSize > 0 (got {splitSize}).");
        int r = x.Rank;
        int d = dim < 0 ? dim + r : dim;
        int size = x.Shape.Dims[d];
        var outList = new List<Tensor>();
        for (int start = 0; start < size; start += splitSize)
            outList.Add(Narrow(x, d, start, Math.Min(splitSize, size - start)));
        return outList.ToArray();
    }

    public static Tensor[] Split(Tensor x, int[] sizes, int dim = 0)
    {
        int r = x.Rank;
        int d = dim < 0 ? dim + r : dim;
        int total = 0;
        foreach (var s in sizes)
        {
            if (s < 0) throw new InvalidOperationException($"Split sizes must be non-negative (got {s}).");
            total += s;
        }
        if (total != x.Shape.Dims[d])
            throw new InvalidOperationException(
                $"Split sizes [{string.Join(",", sizes)}] sum to {total}, but dim {d} of {x.Shape} is {x.Shape.Dims[d]}.");
        var outList = new List<Tensor>();
        int start = 0;
        foreach (var s in sizes)
        {
            outList.Add(Narrow(x, d, start, s));
            start += s;
        }
        return outList.ToArray();
    }
}

public sealed partial class Tensor
{
    public Tensor Squeeze(int? dim = null) => TensorOps.Squeeze(this, dim);
    public Tensor Unsqueeze(int dim) => TensorOps.Unsqueeze(this, dim);
    public Tensor Flatten(int start = 0, int end = -1) => TensorOps.Flatten(this, start, end);
    public Tensor Expand(params int[] sizes) => TensorOps.Expand(this, sizes);
    public Tensor Narrow(int dim, int start, int length) => TensorOps.Narrow(this, dim, start, length);
}
