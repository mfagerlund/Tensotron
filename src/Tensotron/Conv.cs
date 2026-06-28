namespace Tensotron;

/// <summary>2D convolution layer (torch.nn.Conv2d) with torch-default init.</summary>
public sealed class Conv2d : Module
{
    private readonly int _stride, _padding, _dilation;
    public Tensor Weight { get; }   // (O, C, kh, kw)
    public Tensor? Bias { get; }    // (O,)

    public Conv2d(int inChannels, int outChannels, int kernelSize,
        int stride = 1, int padding = 0, int dilation = 1, bool bias = true)
    {
        _stride = stride;
        _padding = padding;
        _dilation = dilation;

        Weight = Tensor.Zeros(new Shape(outChannels, inChannels, kernelSize, kernelSize)).RequireGrad();
        Init.KaimingUniform_(Weight, a: MathF.Sqrt(5f)); // torch Conv2d default

        if (bias)
        {
            Bias = Tensor.Zeros(new Shape(outChannels)).RequireGrad();
            int fanIn = inChannels * kernelSize * kernelSize;
            float bound = 1f / MathF.Sqrt(fanIn);
            Init.Uniform_(Bias, -bound, bound);
        }
    }

    public override Tensor Forward(Tensor x)
        => TensorOps.Conv2d(x, Weight, Bias, _stride, _padding, _dilation);

    protected override IEnumerable<(string, Tensor)> OwnParameters()
    {
        yield return ("weight", Weight);
        if (Bias is not null) yield return ("bias", Bias);
    }
}
