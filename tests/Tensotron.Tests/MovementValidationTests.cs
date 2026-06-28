using Tensotron;

namespace Tensotron.Tests;

public class MovementValidationTests
{
    [Fact]
    public void Permute_RejectsWrongCount()
    {
        var x = Tensor.Zeros(new Shape(2, 3, 4));
        Assert.Throws<InvalidOperationException>(() => TensorOps.Permute(x, 0, 1));
    }

    [Fact]
    public void Permute_RejectsDuplicateAxis()
    {
        var x = Tensor.Zeros(new Shape(2, 3));
        Assert.Throws<InvalidOperationException>(() => TensorOps.Permute(x, 0, 0));
    }

    [Fact]
    public void Permute_RejectsOutOfRangeAxis()
    {
        var x = Tensor.Zeros(new Shape(2, 3));
        Assert.Throws<InvalidOperationException>(() => TensorOps.Permute(x, 0, 5));
    }

    [Fact]
    public void Transpose_RejectsOutOfRangeAxis()
    {
        var x = Tensor.Zeros(new Shape(2, 3));
        Assert.Throws<InvalidOperationException>(() => TensorOps.Transpose(x, 0, 9));
    }
}
