namespace Tensotron;

public static partial class TensorOps
{
    private static int ConvOutSize(int input, int kernel, int stride, int pad, int dilation)
        => (input + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;

    // Stable argument validation for conv/pool window parameters (torch raises ValueError;
    // we throw before any division/allocation so callers see a clear message, not a backend fault).
    internal static void ValidateWindowParams(string op, int kernel, int stride, int pad, int dilation)
    {
        if (kernel <= 0) throw new ArgumentOutOfRangeException(nameof(kernel), $"{op}: kernel size must be > 0 (got {kernel}).");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), $"{op}: stride must be > 0 (got {stride}).");
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation), $"{op}: dilation must be > 0 (got {dilation}).");
        if (pad < 0) throw new ArgumentOutOfRangeException(nameof(pad), $"{op}: padding must be >= 0 (got {pad}).");
    }

    private static int[] ConvCfg(int n, int c, int h, int w, int kh, int kw,
        int sh, int sw, int ph, int pw, int dh, int dw, int ho, int wo)
        => new[] { n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, ho, wo, c * kh * kw, ho * wo };

    /// <summary>
    /// im2col: turn each conv receptive field of (N,C,H,W) into a column, giving
    /// (N, C·kh·kw, Hout·Wout). Tracked: backward scatter-adds columns back (col2im).
    /// </summary>
    public static Tensor Im2Col(Tensor x, int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw)
    {
        if (x.Rank != 4) throw new InvalidOperationException($"Im2Col expects (N,C,H,W), got {x.Shape}.");
        ValidateWindowParams("Im2Col", kh, sh, ph, dh);
        ValidateWindowParams("Im2Col", kw, sw, pw, dw);
        int n = x.Shape.Dims[0], c = x.Shape.Dims[1], h = x.Shape.Dims[2], w = x.Shape.Dims[3];
        int ho = ConvOutSize(h, kh, sh, ph, dh);
        int wo = ConvOutSize(w, kw, sw, pw, dw);
        if (ho <= 0 || wo <= 0)
            throw new InvalidOperationException($"Im2Col: non-positive output size ({ho}x{wo}) for {x.Shape}.");
        var cfg = ConvCfg(n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, ho, wo);

        var outShape = new Shape(n, c * kh * kw, ho * wo);
        var outBuf = Runtime.Allocate(outShape.Size);
        Runtime.LaunchIm2Col(x.Buffer, outBuf, cfg);
        var result = new Tensor(outShape, outBuf);

        if (!Tensor.NoGrad && Tensor.NeedsGrad(x))
        {
            result.Node = new GradNode("Im2Col", new[] { x }, g =>
            {
                var gx = Tensor.Zeros(x.Shape);
                Runtime.LaunchCol2Im(g.Buffer, gx.Buffer, cfg);
                x.AddGrad(gx);
            });
        }
        return result;
    }

    /// <summary>
    /// 2D convolution (torch.nn.functional.conv2d) via im2col + batched matmul. Input
    /// (N,C,H,W), weight (O,C,kh,kw), optional bias (O,). Backward is automatic through
    /// the matmul / im2col chain. Symmetric stride/padding/dilation.
    /// </summary>
    public static Tensor Conv2d(Tensor x, Tensor weight, Tensor? bias = null,
        int stride = 1, int padding = 0, int dilation = 1)
    {
        if (x.Rank != 4) throw new InvalidOperationException($"Conv2d expects (N,C,H,W), got {x.Shape}.");
        if (weight.Rank != 4) throw new InvalidOperationException($"Conv2d weight must be (O,C,kh,kw), got {weight.Shape}.");
        int o = weight.Shape.Dims[0], wc = weight.Shape.Dims[1], kh = weight.Shape.Dims[2], kw = weight.Shape.Dims[3];
        int n = x.Shape.Dims[0], c = x.Shape.Dims[1], h = x.Shape.Dims[2], w = x.Shape.Dims[3];
        if (wc != c) throw new InvalidOperationException($"Conv2d channel mismatch: input {c} vs weight {wc}.");

        int ho = ConvOutSize(h, kh, stride, padding, dilation);
        int wo = ConvOutSize(w, kw, stride, padding, dilation);

        var col = Im2Col(x, kh, kw, stride, stride, padding, padding, dilation, dilation); // (N, K, L)
        var wflat = weight.Reshape(1, o, c * kh * kw);   // (1, O, K) -> broadcast over batch
        var outNol = MatMul(wflat, col);                  // (N, O, L)
        var outNohw = outNol.Reshape(n, o, ho, wo);

        if (bias is not null)
            outNohw = Add(outNohw, bias.Reshape(1, o, 1, 1));
        return outNohw;
    }
}
