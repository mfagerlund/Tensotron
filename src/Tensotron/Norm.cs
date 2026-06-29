namespace Tensotron;

/// <summary>Layer normalization layer (torch.nn.LayerNorm) with learnable affine.</summary>
public sealed class LayerNorm : Module
{
    private readonly int[] _normalizedShape;
    private readonly float _eps;
    public Tensor? Weight { get; }   // gamma, shape = normalizedShape
    public Tensor? Bias { get; }     // beta,  shape = normalizedShape

    public LayerNorm(int[] normalizedShape, bool affine = true, float eps = 1e-5f)
    {
        _normalizedShape = (int[])normalizedShape.Clone();
        _eps = eps;
        if (affine)
        {
            Weight = Tensor.Ones(new Shape(_normalizedShape)).RequireGrad();   // gamma init 1
            Bias = Tensor.Zeros(new Shape(_normalizedShape)).RequireGrad();    // beta  init 0
        }
    }

    public LayerNorm(int normalizedSize, bool affine = true, float eps = 1e-5f)
        : this(new[] { normalizedSize }, affine, eps) { }

    public override Tensor Forward(Tensor x)
        => TensorOps.LayerNorm(x, _normalizedShape, Weight, Bias, _eps);

    protected override IEnumerable<(string, Tensor)> OwnParameters()
    {
        if (Weight is not null) yield return ("weight", Weight);
        if (Bias is not null) yield return ("bias", Bias);
    }
}

/// <summary>
/// Batch normalization for (N, C) inputs (torch.nn.BatchNorm1d). In training it
/// normalizes with batch statistics (biased var) and updates running stats with the
/// unbiased var; in eval it normalizes with the running stats. Affine gamma/beta.
/// </summary>
public sealed class BatchNorm1d : Module
{
    private readonly int _numFeatures;
    private readonly float _eps, _momentum;
    public Tensor Weight { get; }        // gamma (C)
    public Tensor Bias { get; }          // beta  (C)
    public Tensor RunningMean { get; }   // (C), buffer (no grad)
    public Tensor RunningVar { get; }    // (C), buffer (no grad)

    public BatchNorm1d(int numFeatures, float eps = 1e-5f, float momentum = 0.1f)
    {
        _numFeatures = numFeatures;
        _eps = eps;
        _momentum = momentum;
        Weight = Tensor.Ones(new Shape(numFeatures)).RequireGrad();
        Bias = Tensor.Zeros(new Shape(numFeatures)).RequireGrad();
        RunningMean = Tensor.Zeros(new Shape(numFeatures));
        RunningVar = Tensor.Ones(new Shape(numFeatures));
    }

    public override Tensor Forward(Tensor x)
    {
        if (x.Rank != 2 || x.Shape.Dims[1] != _numFeatures)
            throw new InvalidOperationException($"BatchNorm1d expects (N,{_numFeatures}), got {x.Shape}.");

        Tensor normalized;
        if (Training)
        {
            var mean = TensorOps.Mean(x, new[] { 0 }, keepdim: true);                       // (1,C)
            var varBiased = TensorOps.Var(x, new[] { 0 }, keepdim: true, unbiased: false);  // (1,C)
            normalized = TensorOps.Div(TensorOps.Sub(x, mean),
                TensorOps.Sqrt(TensorOps.Add(varBiased, TensorOps.Scalar(_eps))));

            UpdateRunningStats(x, mean);
        }
        else
        {
            var mean = RunningMean.Reshape(1, _numFeatures);
            var variance = RunningVar.Reshape(1, _numFeatures);
            normalized = TensorOps.Div(TensorOps.Sub(x, mean),
                TensorOps.Sqrt(TensorOps.Add(variance, TensorOps.Scalar(_eps))));
        }

        // affine: gamma/beta are (C,), broadcast over the batch axis.
        return TensorOps.Add(TensorOps.Mul(normalized, Weight), Bias);
    }

    // running = (1-momentum)*running + momentum*batch_stat, detached. torch uses the
    // UNBIASED batch variance for the running estimate (biased for normalization).
    private void UpdateRunningStats(Tensor x, Tensor meanKeep)
    {
        using (Tensor.NoGradScope())
        {
            // Unbiased variance divides by (count-1); torch rejects training a BatchNorm on a
            // single sample (would be division by zero → NaN running stats).
            int count = x.Shape.Dims[0];
            if (count <= 1)
                throw new InvalidOperationException(
                    $"BatchNorm1d training requires more than one value per channel (got N={count}).");
            var meanFlat = meanKeep.Reshape(_numFeatures);
            var varUnbiased = TensorOps.Var(x, new[] { 0 }, keepdim: false, unbiased: true);
            RunningMean.CopyInPlace(TensorOps.Add(
                TensorOps.Mul(RunningMean, TensorOps.Scalar(1f - _momentum)),
                TensorOps.Mul(meanFlat, TensorOps.Scalar(_momentum))));
            RunningVar.CopyInPlace(TensorOps.Add(
                TensorOps.Mul(RunningVar, TensorOps.Scalar(1f - _momentum)),
                TensorOps.Mul(varUnbiased, TensorOps.Scalar(_momentum))));
        }
    }

    protected override IEnumerable<(string, Tensor)> OwnParameters()
    {
        yield return ("weight", Weight);
        yield return ("bias", Bias);
    }

    protected override IEnumerable<(string, Tensor)> OwnBuffers()
    {
        yield return ("running_mean", RunningMean);
        yield return ("running_var", RunningVar);
    }
}

/// <summary>Group normalization layer (torch.nn.GroupNorm) with per-channel affine.</summary>
public sealed class GroupNorm : Module
{
    private readonly int _numGroups;
    private readonly float _eps;
    public Tensor? Weight { get; }   // (C,)
    public Tensor? Bias { get; }     // (C,)

    public GroupNorm(int numGroups, int numChannels, bool affine = true, float eps = 1e-5f)
    {
        _numGroups = numGroups;
        _eps = eps;
        if (affine)
        {
            Weight = Tensor.Ones(new Shape(numChannels)).RequireGrad();
            Bias = Tensor.Zeros(new Shape(numChannels)).RequireGrad();
        }
    }

    public override Tensor Forward(Tensor x)
        => TensorOps.GroupNorm(x, _numGroups, Weight, Bias, _eps);

    protected override IEnumerable<(string, Tensor)> OwnParameters()
    {
        if (Weight is not null) yield return ("weight", Weight);
        if (Bias is not null) yield return ("bias", Bias);
    }
}

/// <summary>
/// Batch normalization for (N, C, H, W) inputs (torch.nn.BatchNorm2d). Normalizes over
/// the N, H, W axes per channel; running stats use the unbiased batch variance.
/// </summary>
public sealed class BatchNorm2d : Module
{
    private readonly int _numFeatures;
    private readonly float _eps, _momentum;
    public Tensor Weight { get; }        // gamma (C,)
    public Tensor Bias { get; }          // beta  (C,)
    public Tensor RunningMean { get; }   // (C,), buffer
    public Tensor RunningVar { get; }    // (C,), buffer

    public BatchNorm2d(int numFeatures, float eps = 1e-5f, float momentum = 0.1f)
    {
        _numFeatures = numFeatures;
        _eps = eps;
        _momentum = momentum;
        Weight = Tensor.Ones(new Shape(numFeatures)).RequireGrad();
        Bias = Tensor.Zeros(new Shape(numFeatures)).RequireGrad();
        RunningMean = Tensor.Zeros(new Shape(numFeatures));
        RunningVar = Tensor.Ones(new Shape(numFeatures));
    }

    private static readonly int[] StatAxes = { 0, 2, 3 };

    public override Tensor Forward(Tensor x)
    {
        if (x.Rank != 4 || x.Shape.Dims[1] != _numFeatures)
            throw new InvalidOperationException($"BatchNorm2d expects (N,{_numFeatures},H,W), got {x.Shape}.");

        Tensor normalized;
        if (Training)
        {
            var mean = TensorOps.Mean(x, StatAxes, keepdim: true);                       // (1,C,1,1)
            var varBiased = TensorOps.Var(x, StatAxes, keepdim: true, unbiased: false);
            normalized = TensorOps.Div(TensorOps.Sub(x, mean),
                TensorOps.Sqrt(TensorOps.Add(varBiased, TensorOps.Scalar(_eps))));
            UpdateRunningStats(x, mean);
        }
        else
        {
            var mean = RunningMean.Reshape(1, _numFeatures, 1, 1);
            var variance = RunningVar.Reshape(1, _numFeatures, 1, 1);
            normalized = TensorOps.Div(TensorOps.Sub(x, mean),
                TensorOps.Sqrt(TensorOps.Add(variance, TensorOps.Scalar(_eps))));
        }

        var w = Weight.Reshape(1, _numFeatures, 1, 1);
        var b = Bias.Reshape(1, _numFeatures, 1, 1);
        return TensorOps.Add(TensorOps.Mul(normalized, w), b);
    }

    private void UpdateRunningStats(Tensor x, Tensor meanKeep)
    {
        using (Tensor.NoGradScope())
        {
            // As BatchNorm1d: unbiased var over N·H·W needs more than one element per channel.
            int count = x.Shape.Dims[0] * x.Shape.Dims[2] * x.Shape.Dims[3];
            if (count <= 1)
                throw new InvalidOperationException(
                    $"BatchNorm2d training requires more than one value per channel (got N·H·W={count}).");
            var meanFlat = meanKeep.Reshape(_numFeatures);
            var varUnbiased = TensorOps.Var(x, StatAxes, keepdim: false, unbiased: true);
            RunningMean.CopyInPlace(TensorOps.Add(
                TensorOps.Mul(RunningMean, TensorOps.Scalar(1f - _momentum)),
                TensorOps.Mul(meanFlat, TensorOps.Scalar(_momentum))));
            RunningVar.CopyInPlace(TensorOps.Add(
                TensorOps.Mul(RunningVar, TensorOps.Scalar(1f - _momentum)),
                TensorOps.Mul(varUnbiased, TensorOps.Scalar(_momentum))));
        }
    }

    protected override IEnumerable<(string, Tensor)> OwnParameters()
    {
        yield return ("weight", Weight);
        yield return ("bias", Bias);
    }

    protected override IEnumerable<(string, Tensor)> OwnBuffers()
    {
        yield return ("running_mean", RunningMean);
        yield return ("running_var", RunningVar);
    }
}
