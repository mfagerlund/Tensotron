namespace Tensotron;

/// <summary>2D max pooling layer (torch.nn.MaxPool2d). Stateless. Square window.</summary>
public sealed class MaxPool2d : Module
{
    private readonly int _kernelSize, _stride, _padding, _dilation;

    public MaxPool2d(int kernelSize, int stride = -1, int padding = 0, int dilation = 1)
    {
        _kernelSize = kernelSize;
        _stride = stride <= 0 ? kernelSize : stride;
        _padding = padding;
        _dilation = dilation;
    }

    public override Tensor Forward(Tensor x)
        => TensorOps.MaxPool2d(x, _kernelSize, _stride, _padding, _dilation);
}

/// <summary>2D average pooling layer (torch.nn.AvgPool2d, count_include_pad=True). Stateless.</summary>
public sealed class AvgPool2d : Module
{
    private readonly int _kernelSize, _stride, _padding;

    public AvgPool2d(int kernelSize, int stride = -1, int padding = 0)
    {
        _kernelSize = kernelSize;
        _stride = stride <= 0 ? kernelSize : stride;
        _padding = padding;
    }

    public override Tensor Forward(Tensor x)
        => TensorOps.AvgPool2d(x, _kernelSize, _stride, _padding);
}
