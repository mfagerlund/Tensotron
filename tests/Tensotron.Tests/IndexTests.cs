using Tensotron;

namespace Tensotron.Tests;

public class IndexTests
{
    [Fact]
    public void Index_MatchTorch()
    {
        var fix = Fixtures.Load("index");
        foreach (var c in fix.Cases)
        {
            var m = c.Meta!;
            switch (m.Op)
            {
                case "index_select":
                {
                    var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                    var y = TensorOps.IndexSelect(x, m.Dim, m.Index!);
                    Check1(c, y, x);
                    break;
                }
                case "gather":
                {
                    var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                    var y = TensorOps.Gather(x, m.Dim, m.Index!, m.IndexShape!);
                    Check1(c, y, x);
                    break;
                }
                case "scatter_add":
                {
                    var self = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                    var src = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
                    var y = TensorOps.ScatterAdd(self, m.Dim, m.Index!, m.IndexShape!, src);
                    Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                    y.Backward(Fixtures.ToTensor(c.GradOutput));
                    Fixtures.AssertMatches($"{c.Name} grad self", c.Grads[0], self.Grad!);
                    Fixtures.AssertMatches($"{c.Name} grad src", c.Grads[1], src.Grad!);
                    break;
                }
                case "repeat":
                {
                    var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                    var y = TensorOps.Repeat(x, m.Sizes!);
                    Check1(c, y, x);
                    break;
                }
                default:
                    throw new InvalidOperationException(m.Op);
            }
        }
    }

    private static void Check1(Case c, Tensor y, Tensor x)
    {
        Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
        y.Backward(Fixtures.ToTensor(c.GradOutput));
        Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
    }

    // gather(dim, Tensor index) convenience overload bridges argmax output into a gather.
    [Fact]
    public void GatherWithArgmax_RoundTrips()
    {
        var x = Tensor.FromShaped(new[] { 1f, 5f, 2f, 9f, 3f, 4f }, new[] { 2, 3 });
        var idx = x.Argmax(dim: 1, keepdim: true);          // float tensor [[1],[0]]
        var picked = TensorOps.Gather(x, 1, idx);            // -> [[5],[9]]
        Assert.Equal(new[] { 5f, 9f }, picked.ToArray());
    }
}
