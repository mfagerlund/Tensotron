namespace Tensotron;

public static partial class TensorOps
{
    private static int[] PoolCfg(int n, int c, int h, int w, int kh, int kw,
        int sh, int sw, int ph, int pw, int dh, int dw, int ho, int wo)
        => new[] { n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, ho, wo };

    /// <summary>
    /// 2D max pooling (torch.nn.functional.max_pool2d). Input (N,C,H,W); square window.
    /// Backward routes each output gradient to the cell it was max of.
    /// </summary>
    public static Tensor MaxPool2d(Tensor x, int kernelSize, int stride = -1, int padding = 0, int dilation = 1)
    {
        if (x.Rank != 4) throw new InvalidOperationException($"MaxPool2d expects (N,C,H,W), got {x.Shape}.");
        if (stride <= 0) stride = kernelSize; // torch: stride defaults to kernel_size
        ValidateWindowParams("MaxPool2d", kernelSize, stride, padding, dilation);
        int n = x.Shape.Dims[0], c = x.Shape.Dims[1], h = x.Shape.Dims[2], w = x.Shape.Dims[3];
        int ho = ConvOutSize(h, kernelSize, stride, padding, dilation);
        int wo = ConvOutSize(w, kernelSize, stride, padding, dilation);
        if (ho <= 0 || wo <= 0)
            throw new InvalidOperationException($"MaxPool2d: non-positive output size ({ho}x{wo}) for {x.Shape}.");
        var cfg = PoolCfg(n, c, h, w, kernelSize, kernelSize, stride, stride, padding, padding, dilation, dilation, ho, wo);

        var outShape = new Shape(n, c, ho, wo);
        var outBuf = Runtime.Allocate(outShape.Size);
        bool needsGrad = !Tensor.NoGrad && Tensor.NeedsGrad(x);
        // Keep the argmax on the device when we'll need it for backward — no host round-trip.
        var argmax = Runtime.LaunchMaxPool2d(x.Buffer, outBuf, cfg, keepArgmax: needsGrad);
        var result = new Tensor(outShape, outBuf);

        if (needsGrad)
        {
            var dArg = argmax!; // device-resident argmax, consumed directly by backward
            result.Node = new GradNode("MaxPool2d", new[] { x }, g =>
            {
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchMaxPool2dGrad(g.Buffer, gx.Buffer, dArg);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    /// <summary>
    /// 2D average pooling (torch.nn.functional.avg_pool2d, count_include_pad=True). Input
    /// (N,C,H,W); square window. Backward spreads each output gradient evenly over its window.
    /// </summary>
    public static Tensor AvgPool2d(Tensor x, int kernelSize, int stride = -1, int padding = 0)
    {
        if (x.Rank != 4) throw new InvalidOperationException($"AvgPool2d expects (N,C,H,W), got {x.Shape}.");
        if (stride <= 0) stride = kernelSize;
        ValidateWindowParams("AvgPool2d", kernelSize, stride, padding, 1);
        int n = x.Shape.Dims[0], c = x.Shape.Dims[1], h = x.Shape.Dims[2], w = x.Shape.Dims[3];
        int ho = ConvOutSize(h, kernelSize, stride, padding, 1);
        int wo = ConvOutSize(w, kernelSize, stride, padding, 1);
        if (ho <= 0 || wo <= 0)
            throw new InvalidOperationException($"AvgPool2d: non-positive output size ({ho}x{wo}) for {x.Shape}.");
        var cfg = PoolCfg(n, c, h, w, kernelSize, kernelSize, stride, stride, padding, padding, 1, 1, ho, wo);

        var outShape = new Shape(n, c, ho, wo);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchAvgPool2d(x.Buffer, outBuf, cfg);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("AvgPool2d", new[] { x }, g =>
            {
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchAvgPool2dGrad(g.Buffer, gx.Buffer, cfg);
                x.AddGrad(gx);
            });
        }
        return result;
    }
}

public sealed partial class Tensor
{
    public Tensor MaxPool2d(int kernelSize, int stride = -1, int padding = 0, int dilation = 1)
        => TensorOps.MaxPool2d(this, kernelSize, stride, padding, dilation);
    public Tensor AvgPool2d(int kernelSize, int stride = -1, int padding = 0)
        => TensorOps.AvgPool2d(this, kernelSize, stride, padding);
}
