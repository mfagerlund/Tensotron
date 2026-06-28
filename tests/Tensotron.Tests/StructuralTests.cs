using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

public class StructuralTests
{
    [Fact]
    public void Structural_MatchTorch()
    {
        var fix = Fixtures.Load("structural");
        foreach (var c in fix.Cases)
        {
            var p = c.Meta!.Params;
            int P(int i) => (int)p![i];

            if (c.Meta.Op is "cat" or "stack")
            {
                var a = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
                var b = Fixtures.ToTensor(c.Inputs[1]).RequireGrad();
                var y = c.Meta.Op == "cat"
                    ? TensorOps.Cat(new[] { a, b }, P(0))
                    : TensorOps.Stack(new[] { a, b }, P(0));
                Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);
                y.Backward(Fixtures.ToTensor(c.GradOutput));
                Fixtures.AssertMatches($"{c.Name} grad a", c.Grads[0], a.Grad!);
                Fixtures.AssertMatches($"{c.Name} grad b", c.Grads[1], b.Grad!);
                continue;
            }

            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            Tensor o = c.Meta.Op switch
            {
                "squeeze" => p is null ? TensorOps.Squeeze(x) : TensorOps.Squeeze(x, P(0)),
                "unsqueeze" => TensorOps.Unsqueeze(x, P(0)),
                "flatten" => TensorOps.Flatten(x, P(0), P(1)),
                "expand" => TensorOps.Expand(x, p!.Select(v => (int)v).ToArray()),
                "narrow" => TensorOps.Narrow(x, P(0), P(1), P(2)),
                _ => throw new InvalidOperationException(c.Meta.Op),
            };
            Fixtures.AssertMatches($"{c.Name} forward", c.Output, o);
            o.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
        }
    }

    // chunk/split are pure compositions of the (torch-grounded) Narrow op, so a
    // reconstruction check is sufficient: the pieces re-concatenate to the input,
    // and gradient flows back through the split.
    [Fact]
    public void ChunkSplit_ReconstructAndGrad()
    {
        var data = Enumerable.Range(0, 2 * 7).Select(i => (float)i).ToArray();
        var x = Tensor.FromShaped(data, new[] { 2, 7 }).RequireGrad();

        var chunks = TensorOps.Chunk(x, 3, dim: 1); // sizes 3,3,1
        Assert.Equal(3, chunks.Length);
        Assert.Equal(new[] { 2, 3 }, chunks[0].Shape.Dims);
        Assert.Equal(new[] { 2, 1 }, chunks[2].Shape.Dims);
        var recat = TensorOps.Cat(chunks, dim: 1);
        Assert.Equal(data, recat.ToArray());

        var splits = TensorOps.Split(x, new[] { 4, 3 }, dim: 1);
        Assert.Equal(new[] { 2, 4 }, splits[0].Shape.Dims);
        Assert.Equal(new[] { 2, 3 }, splits[1].Shape.Dims);

        // grad: sum of all chunk elements -> d/dx = 1 everywhere.
        var loss = TensorOps.Add(chunks[0].Sum(), TensorOps.Add(chunks[1].Sum(), chunks[2].Sum()));
        loss.Backward();
        var g = x.Grad!.ToArray();
        Assert.All(g, v => Assert.Equal(1f, v, 3));
    }
}
