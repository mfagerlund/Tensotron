using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Regression tests for malformed-metadata and repeated-backward edge cases that the
/// happy-path parity fixtures don't exercise.
/// </summary>
public class ValidationTests
{
    // ---- repeated backward must not over-accumulate through interior nodes ----

    [Fact]
    public void RepeatedBackward_AccumulatesLinearlyIntoLeaf()
    {
        var x = Tensor.FromShaped(new[] { 1f, 2f, 3f }, new[] { 3 }).RequireGrad();
        var y = (x * x).Sum();   // dy/dx = 2x = [2,4,6]

        y.Backward();
        var g1 = x.Grad!.ToArray();
        Assert.Equal(new[] { 2f, 4f, 6f }, g1);

        // Second backward without ZeroGrad: leaf accumulates exactly one more 2x
        // (torch retain_graph semantics). Interior nodes must be recomputed fresh —
        // if they over-accumulated, the leaf would exceed 2x the first pass.
        y.Backward();
        var g2 = x.Grad!.ToArray();
        Assert.Equal(new[] { 4f, 8f, 12f }, g2);

        // After ZeroGrad a fresh backward reproduces the original single-pass gradient.
        x.ZeroGrad();
        y.Backward();
        Assert.Equal(new[] { 2f, 4f, 6f }, x.Grad!.ToArray());
    }

    // ---- gather / scatter_add index validation ----

    [Fact]
    public void Gather_RejectsIndexLengthMismatch()
    {
        var x = Tensor.FromShaped(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 });
        // indexShape [2,2] => product 4, but only 2 indices supplied.
        Assert.Throws<InvalidOperationException>(() => x.Gather(1, new[] { 0, 1 }, new[] { 2, 2 }));
    }

    [Fact]
    public void ScatterAdd_RejectsIndexLengthMismatch()
    {
        var self = Tensor.Zeros(new Shape(2, 3));
        var src = Tensor.Zeros(new Shape(2, 2));
        Assert.Throws<InvalidOperationException>(
            () => self.ScatterAdd(1, new[] { 0, 1 }, new[] { 2, 2 }, src));
    }

    [Fact]
    public void ScatterAdd_RejectsNonScatterDimExceedingSelf()
    {
        var self = Tensor.Zeros(new Shape(2, 3));
        var src = Tensor.Zeros(new Shape(3, 2));
        // indexShape [3,2]: dim 0 (3) exceeds self dim 0 (2) — would compute OOB offsets.
        var index = new[] { 0, 1, 0, 1, 0, 1 };
        Assert.Throws<InvalidOperationException>(
            () => self.ScatterAdd(1, index, new[] { 3, 2 }, src));
    }

    // ---- reshape spec validation ----

    [Fact]
    public void Reshape_RejectsMultipleInferredDims()
    {
        Assert.Throws<InvalidOperationException>(() => new Shape(2, 3).Reshape(new[] { -1, -1 }));
    }

    [Fact]
    public void Reshape_RejectsNonIntegralInferredDim()
    {
        Assert.Throws<InvalidOperationException>(() => new Shape(2, 3).Reshape(new[] { 4, -1 }));
    }

    // ---- chunk / split validation ----

    [Fact]
    public void Chunk_RejectsNonPositiveCount()
    {
        var x = Tensor.Zeros(new Shape(6));
        Assert.Throws<InvalidOperationException>(() => TensorOps.Chunk(x, 0));
    }

    [Fact]
    public void Split_RejectsNonPositiveSize()
    {
        var x = Tensor.Zeros(new Shape(6));
        Assert.Throws<InvalidOperationException>(() => TensorOps.Split(x, 0));
    }

    [Fact]
    public void Split_BySizes_RejectsSumMismatch()
    {
        var x = Tensor.Zeros(new Shape(6));
        // sizes sum to 5, not 6
        Assert.Throws<InvalidOperationException>(() => TensorOps.Split(x, new[] { 2, 3 }));
    }

    // ---- shape / reduce / window param validation ----

    [Fact]
    public void Shape_RejectsNegativeDim()
    {
        Assert.Throws<ArgumentException>(() => new Shape(2, -3));
    }

    [Fact]
    public void Reduce_RejectsOutOfRangeDim()
    {
        var x = Tensor.Zeros(new Shape(2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorOps.Sum(x, new[] { 5 }));
    }

    [Fact]
    public void MaxPool2d_RejectsNonPositiveKernel()
    {
        var x = Tensor.Zeros(new Shape(1, 1, 4, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorOps.MaxPool2d(x, 0));
    }

    [Fact]
    public void Conv2d_RejectsNonPositiveDilation()
    {
        var x = Tensor.Zeros(new Shape(1, 1, 5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorOps.Im2Col(x, 3, 3, 1, 1, 0, 0, 0, 0));
    }

    // ---- buffer disposal / ownership ----

    [Fact]
    public void Dispose_FreesOwnedBuffer_AndIsIdempotent()
    {
        var x = Tensor.FromShaped(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        Assert.False(x.IsDisposed);
        x.Dispose();
        Assert.True(x.IsDisposed);
        x.Dispose(); // idempotent — must not throw or double-free
        Assert.True(x.IsDisposed);
    }

    [Fact]
    public void Dispose_View_DoesNotFreeSharedParentBuffer()
    {
        var parent = Tensor.FromShaped(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var view = parent.Reshape(4);   // zero-copy view: shares parent's buffer
        view.Dispose();                 // must NOT free the shared buffer
        Assert.True(view.IsDisposed);
        // parent buffer still valid: reading back works
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, parent.ToArray());
        parent.Dispose();               // owner frees here
    }

    // ---- LeakyReLU / ELU PyTorch-faithful forward + boundary gradient ----

    [Fact]
    public void LeakyRelu_SlopeGreaterThanOne_MatchesTorch()
    {
        // torch.nn.functional.leaky_relu(-1, negative_slope=2) == -2 (old Maximum form gave -1).
        var x = Tensor.FromShaped(new[] { -1f, 2f }, new[] { 2 });
        var y = TensorOps.LeakyRelu(x, slope: 2f).ToArray();
        Assert.Equal(-2f, y[0], 5);
        Assert.Equal(2f, y[1], 5);
    }

    [Fact]
    public void LeakyRelu_GradientAtZero_IsNegativeSlopeBranch()
    {
        // torch: derivative at x==0 takes the negative-slope branch (0.01), not a 0.5 tie-split.
        var x = Tensor.FromShaped(new[] { 0f }, new[] { 1 }).RequireGrad();
        TensorOps.LeakyRelu(x).Sum().Backward();
        Assert.Equal(0.01f, x.Grad!.ToArray()[0], 5);
    }

    [Fact]
    public void Elu_GradientAtZero_IsOne()
    {
        // torch: ELU derivative at x==0 (alpha=1) is 1.0, not a 0.5 tie-split.
        var x = Tensor.FromShaped(new[] { 0f }, new[] { 1 }).RequireGrad();
        TensorOps.Elu(x).Sum().Backward();
        Assert.Equal(1f, x.Grad!.ToArray()[0], 5);
    }

    // ---- backward on non-scalar root requires explicit gradient ----

    [Fact]
    public void Backward_NonScalarRoot_WithoutGrad_Throws()
    {
        var x = Tensor.FromShaped(new[] { 1f, 2f }, new[] { 2 }).RequireGrad();
        var y = x * x;   // non-scalar
        Assert.Throws<InvalidOperationException>(() => y.Backward());
    }

    // ---- chunk on an empty dimension must not hang ----

    [Fact]
    public void Chunk_EmptyDim_DoesNotHang()
    {
        var x = Tensor.Zeros(new Shape(0, 3));
        var parts = TensorOps.Chunk(x, 4); // size 0 → no chunks, must return promptly
        Assert.Empty(parts);
    }

    // ---- DataLoader rejects rank-0 tensors with a stable error ----

    [Fact]
    public void DataLoader_RejectsRankZeroTensor()
    {
        var scalar = Tensor.FromShaped(new[] { 1f }, System.Array.Empty<int>());
        Assert.Throws<InvalidOperationException>(() => new DataLoader(new[] { scalar }, batchSize: 1));
    }

    // ---- float index tensors must be integral, not silently rounded ----

    [Fact]
    public void Gather_NonIntegralFloatIndex_Throws()
    {
        var x = Tensor.FromShaped(new[] { 10f, 11f, 12f, 13f }, new[] { 1, 4 });
        var badIdx = Tensor.FromShaped(new[] { 1.49f }, new[] { 1, 1 });
        Assert.Throws<InvalidOperationException>(() => TensorOps.Gather(x, dim: 1, badIdx));
    }

    [Fact]
    public void Gather_IntegralFloatIndex_Works()
    {
        // argmax/argmin produce exact integral floats; those must flow through unharmed.
        var x = Tensor.FromShaped(new[] { 10f, 11f, 12f, 13f }, new[] { 1, 4 });
        var idx = Tensor.FromShaped(new[] { 2f }, new[] { 1, 1 });
        var picked = TensorOps.Gather(x, dim: 1, idx).ToArray();
        Assert.Equal(12f, picked[0]);
    }

    // ---- GroupNorm gives a useful error on low-rank input, not IndexOutOfRange ----

    [Fact]
    public void GroupNorm_LowRankInput_ThrowsInvalidOperation()
    {
        var x = Tensor.FromShaped(new[] { 1f, 2f, 3f }, new[] { 3 }); // rank 1
        Assert.Throws<InvalidOperationException>(() => TensorOps.GroupNorm(x, numGroups: 1));
    }

    // ---- NoGradScope suspends autograd and always restores, even on exceptions / nesting ----

    [Fact]
    public void NoGradScope_SuspendsAndRestores()
    {
        Assert.False(Tensor.NoGrad);
        using (Tensor.NoGradScope())
        {
            Assert.True(Tensor.NoGrad);
            using (Tensor.NoGradScope()) Assert.True(Tensor.NoGrad);
            Assert.True(Tensor.NoGrad); // inner dispose restores to prior (true), not false
        }
        Assert.False(Tensor.NoGrad);
    }

    [Fact]
    public void NoGradScope_RestoresOnException()
    {
        Assert.False(Tensor.NoGrad);
        Action act = () => { using (Tensor.NoGradScope()) throw new InvalidOperationException("boom"); };
        Assert.Throws<InvalidOperationException>(act);
        Assert.False(Tensor.NoGrad);
    }

    [Fact]
    public void NoGradScope_DisablesGraphRecording()
    {
        var x = Tensor.FromShaped(new[] { 1f, 2f, 3f }, new[] { 3 }).RequireGrad();
        Tensor y;
        using (Tensor.NoGradScope())
            y = (x * x).Sum();
        Assert.Null(y.Node); // no backward graph was built inside the scope
    }
}
