namespace Tensotron;

/// <summary>
/// Weight initialization (torch.nn.init). The scale formulas are torch-faithful and
/// are unit-tested against recorded torch values; the random draws use Tensotron's own
/// RNG (matching torch's *distribution*, not its bit-stream). Fillers mutate in place.
/// </summary>
public static class Init
{
    private static Random _rng = new(1234);

    /// <summary>Reseed the init RNG (for reproducible runs).</summary>
    public static void Seed(int seed) => _rng = new Random(seed);

    /// <summary>
    /// fan_in / fan_out for a weight tensor, torch-style: dims[1]/dims[0] scaled by the
    /// receptive field (product of dims[2:]). Requires rank ≥ 2.
    /// </summary>
    public static (int fanIn, int fanOut) FanInFanOut(Shape shape)
    {
        if (shape.Rank < 2)
            throw new InvalidOperationException("FanInFanOut requires a tensor with rank >= 2.");
        int receptive = 1;
        for (int i = 2; i < shape.Rank; i++) receptive *= shape.Dims[i];
        int fanIn = shape.Dims[1] * receptive;
        int fanOut = shape.Dims[0] * receptive;
        return (fanIn, fanOut);
    }

    /// <summary>Recommended gain for a nonlinearity (torch.nn.init.calculate_gain).</summary>
    public static float CalculateGain(string nonlinearity, float negativeSlope = 0.01f) => nonlinearity switch
    {
        "linear" or "conv1d" or "conv2d" or "conv3d" or "sigmoid" => 1f,
        "tanh" => 5f / 3f,
        "relu" => MathF.Sqrt(2f),
        "leaky_relu" => MathF.Sqrt(2f / (1f + negativeSlope * negativeSlope)),
        "selu" => 0.75f,
        _ => throw new InvalidOperationException($"Unsupported nonlinearity '{nonlinearity}'."),
    };

    private static int Fan(Shape shape, string mode)
    {
        var (fanIn, fanOut) = FanInFanOut(shape);
        return mode switch
        {
            "fan_in" => fanIn,
            "fan_out" => fanOut,
            _ => throw new InvalidOperationException($"mode must be fan_in/fan_out, got '{mode}'."),
        };
    }

    // ---- scale formulas (exposed for testing / introspection) ----

    public static float KaimingNormalStd(Shape shape, float a = 0f, string mode = "fan_in", string nonlinearity = "leaky_relu")
        => CalculateGain(nonlinearity, a) / MathF.Sqrt(Fan(shape, mode));

    public static float KaimingUniformBound(Shape shape, float a = 0f, string mode = "fan_in", string nonlinearity = "leaky_relu")
        => MathF.Sqrt(3f) * KaimingNormalStd(shape, a, mode, nonlinearity);

    public static float XavierNormalStd(Shape shape, float gain = 1f)
    {
        var (fanIn, fanOut) = FanInFanOut(shape);
        return gain * MathF.Sqrt(2f / (fanIn + fanOut));
    }

    public static float XavierUniformBound(Shape shape, float gain = 1f)
        => MathF.Sqrt(3f) * XavierNormalStd(shape, gain);

    // ---- in-place fillers ----

    public static Tensor Uniform_(Tensor t, float low, float high)
    {
        var data = new float[t.Shape.Size];
        for (int i = 0; i < data.Length; i++) data[i] = low + (high - low) * (float)_rng.NextDouble();
        t.CopyInPlace(Tensor.FromShaped(data, t.Shape.Dims));
        return t;
    }

    public static Tensor Normal_(Tensor t, float mean = 0f, float std = 1f)
    {
        var data = new float[t.Shape.Size];
        for (int i = 0; i < data.Length; i++) data[i] = mean + std * NextGaussian();
        t.CopyInPlace(Tensor.FromShaped(data, t.Shape.Dims));
        return t;
    }

    public static Tensor KaimingUniform_(Tensor t, float a = 0f, string mode = "fan_in", string nonlinearity = "leaky_relu")
    {
        float b = KaimingUniformBound(t.Shape, a, mode, nonlinearity);
        return Uniform_(t, -b, b);
    }

    public static Tensor KaimingNormal_(Tensor t, float a = 0f, string mode = "fan_in", string nonlinearity = "leaky_relu")
        => Normal_(t, 0f, KaimingNormalStd(t.Shape, a, mode, nonlinearity));

    public static Tensor XavierUniform_(Tensor t, float gain = 1f)
    {
        float b = XavierUniformBound(t.Shape, gain);
        return Uniform_(t, -b, b);
    }

    public static Tensor XavierNormal_(Tensor t, float gain = 1f)
        => Normal_(t, 0f, XavierNormalStd(t.Shape, gain));

    // Box–Muller standard-normal sample.
    private static float NextGaussian()
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
