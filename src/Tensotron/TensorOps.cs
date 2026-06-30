namespace Tensotron;

/// <summary>
/// The op surface. PyTorch-named, broadcast-aware. Each op: compute forward via a
/// kernel, then (unless NoGrad) attach a named GradNode whose closure deposits
/// input gradients. This is the layer where converting PyTorch code is mechanical.
/// </summary>
public static partial class TensorOps
{
    private static TensorRuntime Runtime => TensorRuntime.Instance;

    // ---------------- broadcasting ----------------

    private static (int[] outDims, int[] aStride, int[] bStride) ComputeBroadcast(Shape a, Shape b)
    {
        int r = Math.Max(a.Rank, b.Rank);
        var outDims = new int[r];
        var aStride = new int[r];
        var bStride = new int[r];
        for (int j = 0; j < r; j++)
        {
            int axa = j - (r - a.Rank);
            int axb = j - (r - b.Rank);
            int da = axa >= 0 ? a.Dims[axa] : 1;
            int db = axb >= 0 ? b.Dims[axb] : 1;
            if (da != db && da != 1 && db != 1)
                throw new InvalidOperationException($"Cannot broadcast {a} with {b} (axis {j}: {da} vs {db}).");
            int od = Math.Max(da, db);
            outDims[j] = od;
            aStride[j] = (axa >= 0 && da == od) ? a.Strides[axa] : 0;
            bStride[j] = (axb >= 0 && db == od) ? b.Strides[axb] : 0;
        }
        return (outDims, aStride, bStride);
    }

    // Broadcast output dims across three operands (torch rules), for the ternary select.
    private static int[] BroadcastDims(Shape a, Shape b, Shape c)
    {
        int r = Math.Max(a.Rank, Math.Max(b.Rank, c.Rank));
        var outDims = new int[r];
        for (int j = 0; j < r; j++)
        {
            int da = DimFromRight(a, r, j), db = DimFromRight(b, r, j), dc = DimFromRight(c, r, j);
            int od = Math.Max(da, Math.Max(db, dc));
            if ((da != 1 && da != od) || (db != 1 && db != od) || (dc != 1 && dc != od))
                throw new InvalidOperationException($"Cannot broadcast {a}, {b}, {c} (axis {j}).");
            outDims[j] = od;
        }
        return outDims;
    }

    private static int DimFromRight(Shape s, int rank, int j)
    {
        int ax = j - (rank - s.Rank);
        return ax >= 0 ? s.Dims[ax] : 1;
    }

    // Per-operand broadcast strides aligned to outDims (stride 0 on a broadcast/absent axis).
    private static int[] BroadcastStridesTo(Shape s, int[] outDims)
    {
        int r = outDims.Length;
        var stride = new int[r];
        for (int j = 0; j < r; j++)
        {
            int ax = j - (r - s.Rank);
            int d = ax >= 0 ? s.Dims[ax] : 1;
            stride[j] = (ax >= 0 && d == outDims[j]) ? s.Strides[ax] : 0;
        }
        return stride;
    }

    /// <summary>
    /// Sum a gradient back to a (smaller, broadcast) input shape. Runs during
    /// backward (under NoGrad). Always returns a fresh tensor (no aliasing).
    /// </summary>
    private static Tensor ReduceGradToShape(Tensor g, Shape target)
    {
        if (g.Shape.Equals(target))
            return g.Clone();

        int r = g.Shape.Rank;
        var reduceMask = new int[r];
        var keepDims = (int[])g.Shape.Dims.Clone();
        for (int j = 0; j < r; j++)
        {
            int axt = j - (r - target.Rank);
            int dt = axt >= 0 ? target.Dims[axt] : 1;
            if (dt != g.Shape.Dims[j])
            {
                reduceMask[j] = 1;
                keepDims[j] = 1;
            }
        }

        var keepShape = new Shape(keepDims);
        var outBuf = Runtime.Allocate(keepShape.Size);
        Runtime.LaunchReduceSum(g.Buffer, outBuf, g.Shape.Dims, g.Shape.Strides, keepDims, reduceMask);
        var reduced = new Tensor(keepShape, outBuf);
        return reduced.Shape.Equals(target) ? reduced : reduced.Reshape(target.Dims);
    }

    private static Tensor BroadcastTo(Tensor src, Shape target)
    {
        // src has rank == target rank with 1s on broadcast axes; Add to zeros expands it.
        return Add(Tensor.Zeros(target), src);
    }

    // ---------------- elementwise ----------------

    internal static void AddInto(Tensor target, Tensor source)
        => Runtime.LaunchAddInto(target.Buffer, source.Buffer);

    public static Tensor Add(Tensor a, Tensor b) => Binary<AddOp>("Add", a, b, AddBackward);
    public static Tensor Sub(Tensor a, Tensor b) => Binary<SubOp>("Sub", a, b, SubBackward);
    public static Tensor Mul(Tensor a, Tensor b) => Binary<MulOp>("Mul", a, b, MulBackward);

    // backward receives (a, b, result, gradOfResult)
    internal static Tensor Binary<TOp>(string name, Tensor a, Tensor b, Action<Tensor, Tensor, Tensor, Tensor> backward)
        where TOp : struct, IBinaryOp
    {
        var (outDims, aStride, bStride) = ComputeBroadcast(a.Shape, b.Shape);
        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchBinary<TOp>(a.Buffer, b.Buffer, outBuf, outDims, aStride, bStride);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(a) || Tensor.NeedsGrad(b)))
            result.Node = new GradNode(name, new[] { a, b }, g => backward(a, b, result, g));

        return result;
    }

    /// <summary>Broadcast binary op with no gradient (comparisons).</summary>
    internal static Tensor BinaryNoGrad<TOp>(Tensor a, Tensor b) where TOp : struct, IBinaryOp
    {
        var (outDims, aStride, bStride) = ComputeBroadcast(a.Shape, b.Shape);
        var outShape = new Shape(outDims);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchBinary<TOp>(a.Buffer, b.Buffer, outBuf, outDims, aStride, bStride);
        return new Tensor(outShape, outBuf);
    }

    private static void AddBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(g, a.Shape));
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(g, b.Shape));
    }

    private static void SubBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(g, a.Shape));
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Neg(g), b.Shape));
    }

    private static void MulBackward(Tensor a, Tensor b, Tensor res, Tensor g)
    {
        if (Tensor.NeedsGrad(a)) a.AddGrad(ReduceGradToShape(Mul(g, b), a.Shape));
        if (Tensor.NeedsGrad(b)) b.AddGrad(ReduceGradToShape(Mul(g, a), b.Shape));
    }

    // ---------------- reduction ----------------

    public static Tensor Sum(Tensor x, int[]? dims = null, bool keepdim = false)
    {
        int r = x.Shape.Rank;
        var reduceMask = new int[r];
        if (dims == null)
            for (int i = 0; i < r; i++) reduceMask[i] = 1;
        else
            foreach (var d in dims)
            {
                int a = d < 0 ? d + r : d;
                if (a < 0 || a >= r)
                    throw new ArgumentOutOfRangeException(nameof(dims),
                        $"Reduce dim {d} is out of range for rank-{r} tensor {x.Shape}.");
                reduceMask[a] = 1;
            }

        var keepDims = (int[])x.Shape.Dims.Clone();
        for (int i = 0; i < r; i++) if (reduceMask[i] == 1) keepDims[i] = 1;
        var keepShape = new Shape(keepDims);

        var outBuf = Runtime.Allocate(keepShape.Size);
        Runtime.LaunchReduceSum(x.Buffer, outBuf,
            x.Shape.Dims, x.Shape.Strides, keepDims, reduceMask);

        Tensor result = new Tensor(keepShape, outBuf);

        if (!keepdim)
        {
            var finalDims = new List<int>();
            for (int i = 0; i < r; i++) if (reduceMask[i] == 0) finalDims.Add(x.Shape.Dims[i]);
            var finalShape = finalDims.Count == 0 ? new Shape() : new Shape(finalDims.ToArray());
            result = new Tensor(finalShape, outBuf);
        }

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("Sum", new[] { x }, g =>
            {
                var gk = g.Shape.Equals(keepShape) ? g : g.Reshape(keepShape.Dims);
                x.AddGrad(BroadcastTo(gk, x.Shape));
            });
        }

        return result;
    }

    // ---------------- matmul ----------------

    public static Tensor MatMul(Tensor a, Tensor b)
    {
        // PyTorch promotion rules. 1D operands are promoted to 2D, then squeezed back.
        if (a.Rank == 2 && b.Rank == 2)
            return MatMul2D(a, b);
        if (a.Rank == 1 && b.Rank == 2)
            return MatMul2D(a.Reshape(1, a.Shape.Dims[0]), b).Reshape(b.Shape.Dims[1]);
        if (a.Rank == 2 && b.Rank == 1)
            return MatMul2D(a, b.Reshape(b.Shape.Dims[0], 1)).Reshape(a.Shape.Dims[0]);
        if (a.Rank == 1 && b.Rank == 1)
            return MatMul2D(a.Reshape(1, a.Shape.Dims[0]), b.Reshape(b.Shape.Dims[0], 1)).Reshape();

        // N-D: at least one operand is batched. Promote any 1D operand the PyTorch way
        // (prepend for a, append for b) then squeeze the inserted axis off the result.
        if (a.Rank == 1) // a:(K) , b:(...,K,N) -> (...,1,N) -> squeeze -2
        {
            var res = MatMulND(a.Reshape(1, a.Shape.Dims[0]), b);
            var d = res.Shape.Dims;
            var keep = new int[d.Length - 1];
            for (int i = 0, j = 0; i < d.Length; i++) if (i != d.Length - 2) keep[j++] = d[i];
            return res.Reshape(keep);
        }
        if (b.Rank == 1) // a:(...,M,K) , b:(K) -> (...,M,1) -> squeeze -1
        {
            var res = MatMulND(a, b.Reshape(b.Shape.Dims[0], 1));
            var d = res.Shape.Dims;
            return res.Reshape(d[..^1]);
        }
        return MatMulND(a, b);
    }

    private static Tensor MatMul2D(Tensor a, Tensor b)
    {
        int M = a.Shape.Dims[0], K = a.Shape.Dims[1];
        int K2 = b.Shape.Dims[0], N = b.Shape.Dims[1];
        if (K != K2)
            throw new InvalidOperationException($"MatMul inner dims differ: {a.Shape} @ {b.Shape}.");

        var outBuf = Runtime.Allocate(M * N);
        // forward: a row-major (M,K) -> aMs=K, aKs=1 ; b row-major (K,N) -> bKs=N, bNs=1
        Runtime.LaunchMatMul(a.Buffer, b.Buffer, outBuf, M, N, K, K, 1, N, 1);
        var result = new Tensor(new Shape(M, N), outBuf);

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(a) || Tensor.NeedsGrad(b)))
        {
            result.Node = new GradNode("MatMul", new[] { a, b }, g =>
            {
                int Mm = M, Nn = N, Kk = K;
                if (Tensor.NeedsGrad(a))
                {
                    // dA (M,K) = g (M,N) @ Bᵀ (N,K).  c[m,kk]=sum_n g[m,n]*B[kk,n]
                    //   a=g: aMs=N,aKs=1 ; b=B (Bᵀ via strides): bNs=N,bKs=1
                    var dABuf = Runtime.Allocate(Mm * Kk);
                    Runtime.LaunchMatMul(g.Buffer, b.Buffer, dABuf, Mm, Kk, Nn, Nn, 1, 1, Nn);
                    a.AddGrad(new Tensor(new Shape(Mm, Kk), dABuf));
                }
                if (Tensor.NeedsGrad(b))
                {
                    // dB (K,N) = Aᵀ (K,M) @ g (M,N).  c[kk,n]=sum_m A[m,kk]*g[m,n]
                    //   a=A (Aᵀ via strides): aMs=1,aKs=K ; b=g: bKs=N,bNs=1
                    var dBBuf = Runtime.Allocate(Kk * Nn);
                    Runtime.LaunchMatMul(a.Buffer, g.Buffer, dBBuf, Kk, Nn, Mm, 1, Kk, Nn, 1);
                    b.AddGrad(new Tensor(new Shape(Kk, Nn), dBBuf));
                }
            });
        }

        return result;
    }
}

/// <summary>PyTorch-style operator sugar and instance methods.</summary>
public sealed partial class Tensor
{
    public static Tensor operator +(Tensor a, Tensor b) => TensorOps.Add(a, b);
    public static Tensor operator -(Tensor a, Tensor b) => TensorOps.Sub(a, b);
    public static Tensor operator *(Tensor a, Tensor b) => TensorOps.Mul(a, b);

    public Tensor MatMul(Tensor other) => TensorOps.MatMul(this, other);
    public Tensor Sum(int[]? dims = null, bool keepdim = false) => TensorOps.Sum(this, dims, keepdim);
    public int Rank => Shape.Rank;
}
