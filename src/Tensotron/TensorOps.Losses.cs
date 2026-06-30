namespace Tensotron;

/// <summary>How a per-element loss is collapsed to its final value (torch's `reduction`).</summary>
public enum Reduction { Mean, Sum, None }

public static partial class TensorOps
{
    private static Tensor ApplyReduction(Tensor perElement, Reduction reduction) => reduction switch
    {
        Reduction.Mean => Mean(perElement),
        Reduction.Sum => Sum(perElement),
        Reduction.None => perElement,
        _ => throw new ArgumentOutOfRangeException(nameof(reduction)),
    };

    // ---------------- regression losses ----------------

    /// <summary>
    /// Mean/sum/none of (input − target)². Matches torch.nn.functional.mse_loss.
    /// </summary>
    /// <remarks>
    /// Fused loss tail: one elementwise (input−target)² pass plus one reduction sit behind a
    /// single grad node, rather than composing Sub→Square→Sum→Mul (up to four nodes — and the
    /// dominant per-op cost is host-side graph-node construction). The forward is bit-identical
    /// to <c>ApplyReduction(Square(Sub(input, target)), reduction)</c>; losses.json pins it to
    /// torch and <c>MseLoss_FusedMatchesComposed</c> pins it to the composed expression across
    /// every reduction and a broadcast case.
    /// </remarks>
    public static Tensor MseLoss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
    {
        var (outDims, aStride, bStride) = ComputeBroadcast(input.Shape, target.Shape);
        var outShape = new Shape(outDims);
        int count = outShape.Size;

        Tensor result;
        using (Tensor.NoGradScope())
        {
            var perBuf = Runtime.Allocate(outShape.Size);
            Runtime.LaunchBinary<SqDiffOp>(input.Buffer, target.Buffer, perBuf, outDims, aStride, bStride);
            var per = new Tensor(outShape, perBuf);
            result = reduction switch
            {
                Reduction.None => per,
                Reduction.Sum => Sum(per),
                Reduction.Mean => Mul(Sum(per), Scalar(1f / count)),
                _ => throw new ArgumentOutOfRangeException(nameof(reduction)),
            };
        }

        if (!Tensor.NoGrad && (Tensor.NeedsGrad(input) || Tensor.NeedsGrad(target)))
        {
            // d/dinput (input−target)² = 2·(input−target)·g. For None, g is per-element; for
            // Sum/Mean g is a scalar that broadcasts, and Mean folds 1/count into the scale.
            float gScale = reduction == Reduction.Mean ? 2f / count : 2f;
            result.Node = new GradNode("MseLoss", new[] { input, target }, g =>
            {
                var diff = Sub(input, target);
                var ga = Mul(Mul(g, Scalar(gScale)), diff);
                if (Tensor.NeedsGrad(input)) input.AddGrad(ReduceGradToShape(ga, input.Shape));
                if (Tensor.NeedsGrad(target)) target.AddGrad(ReduceGradToShape(Neg(ga), target.Shape));
            });
        }

        return result;
    }

    public static Tensor L1Loss(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
        => ApplyReduction(Abs(Sub(input, target)), reduction);

    /// <summary>Huber loss: 0.5·d² for |d| ≤ delta, else delta·(|d| − 0.5·delta).</summary>
    public static Tensor HuberLoss(Tensor input, Tensor target, float delta = 1.0f, Reduction reduction = Reduction.Mean)
    {
        var diff = Sub(input, target);
        var abs = Abs(diff);
        var quad = Mul(Scalar(0.5f), Square(diff));
        var lin = Mul(Scalar(delta), Sub(abs, Scalar(0.5f * delta)));
        var per = Where(abs <= delta, quad, lin);
        return ApplyReduction(per, reduction);
    }

    // ---------------- classification losses ----------------

    /// <summary>
    /// Numerically stable BCE on logits: max(x,0) − x·z + log(1 + e^−|x|).
    /// Matches torch.nn.functional.binary_cross_entropy_with_logits.
    /// </summary>
    public static Tensor BceWithLogits(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
    {
        var per = Add(
            Sub(Maximum(input, Scalar(0f)), Mul(input, target)),
            Log1p(Exp(Neg(Abs(input)))));
        return ApplyReduction(per, reduction);
    }

    /// <summary>Negative log-likelihood. <paramref name="logProbs"/> is (N, C) log-probabilities;
    /// <paramref name="target"/> holds the N true class indices.</summary>
    public static Tensor NllLoss(Tensor logProbs, int[] target, Reduction reduction = Reduction.Mean)
    {
        int n = logProbs.Shape.Dims[0];
        if (target.Length != n)
            throw new InvalidOperationException($"NllLoss target length {target.Length} != batch {n}.");
        var picked = Gather(logProbs, 1, target, new[] { n, 1 }); // (N,1)
        var per = Neg(picked.Reshape(n));                          // (N,)
        return ApplyReduction(per, reduction);
    }

    public static Tensor CrossEntropy(Tensor logits, int[] target, Reduction reduction = Reduction.Mean)
        => NllLoss(LogSoftmax(logits, 1), target, reduction);

    /// <summary>KL divergence. <paramref name="input"/> is log-probabilities, <paramref name="target"/>
    /// is probabilities: pointwise = target·(log(target) − input). Matches torch.nn.functional.kl_div.</summary>
    public static Tensor KlDiv(Tensor input, Tensor target, Reduction reduction = Reduction.Mean)
    {
        // pointwise = target·(log(target) − input). At target==0 torch treats the term as 0
        // (lim_{t→0} t·log t = 0). Computing log(0) = −∞ would give 0·−∞ = NaN, and Where
        // can't rescue it (0·NaN is still NaN), so clamp the log argument: Maximum(target, ε)
        // leaves every real positive target untouched and yields 0·finite = 0 at target==0.
        var per = Mul(target, Sub(Log(Maximum(target, Scalar(float.Epsilon))), input));
        return ApplyReduction(per, reduction);
    }
}

public sealed partial class Tensor
{
    public Tensor MseLoss(Tensor target, Reduction reduction = Reduction.Mean) => TensorOps.MseLoss(this, target, reduction);
    public Tensor L1Loss(Tensor target, Reduction reduction = Reduction.Mean) => TensorOps.L1Loss(this, target, reduction);
    public Tensor HuberLoss(Tensor target, float delta = 1.0f, Reduction reduction = Reduction.Mean) => TensorOps.HuberLoss(this, target, delta, reduction);
    public Tensor BceWithLogits(Tensor target, Reduction reduction = Reduction.Mean) => TensorOps.BceWithLogits(this, target, reduction);
    public Tensor NllLoss(int[] target, Reduction reduction = Reduction.Mean) => TensorOps.NllLoss(this, target, reduction);
    public Tensor CrossEntropy(int[] target, Reduction reduction = Reduction.Mean) => TensorOps.CrossEntropy(this, target, reduction);
    public Tensor KlDiv(Tensor target, Reduction reduction = Reduction.Mean) => TensorOps.KlDiv(this, target, reduction);
}
