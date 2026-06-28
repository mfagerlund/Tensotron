using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

public class DataLoaderTests
{
    private static Tensor Range2D(int n, int c)
        => Tensor.FromShaped(Enumerable.Range(0, n * c).Select(i => (float)i).ToArray(), new[] { n, c });

    [Fact]
    public void Sequential_ReconstructsDataInOrder()
    {
        var x = Range2D(10, 3);
        var y = Tensor.FromShaped(Enumerable.Range(0, 10).Select(i => (float)i).ToArray(), new[] { 10 });
        var dl = new DataLoader(x, y, batchSize: 4, shuffle: false);

        Assert.Equal(3, dl.BatchCount); // 4 + 4 + 2

        var xs = new List<float>();
        var sizes = new List<int>();
        foreach (var batch in dl.Epoch())
        {
            sizes.Add(batch[0].Shape.Dims[0]);
            xs.AddRange(batch[0].ToArray());
        }
        Assert.Equal(new[] { 4, 4, 2 }, sizes);
        Assert.Equal(x.ToArray(), xs.ToArray()); // exact concatenation recovers the data
    }

    [Fact]
    public void Shuffle_IsAPermutationAndKeepsXyAligned()
    {
        int n = 12;
        // y[i] encodes its own row index so we can check x/y stay paired after shuffle.
        var x = Range2D(n, 2);
        var y = Tensor.FromShaped(Enumerable.Range(0, n).Select(i => (float)i).ToArray(), new[] { n });
        var dl = new DataLoader(x, y, batchSize: 5, shuffle: true, seed: 99);

        var seenRows = new List<int>();
        foreach (var batch in dl.Epoch())
        {
            var xb = batch[0].ToArray();
            var yb = batch[1].ToArray();
            int rows = batch[0].Shape.Dims[0];
            for (int r = 0; r < rows; r++)
            {
                int idx = (int)yb[r];
                // x row r must be [2*idx, 2*idx+1] — same row that y points at.
                Assert.Equal(2f * idx, xb[r * 2]);
                Assert.Equal(2f * idx + 1, xb[r * 2 + 1]);
                seenRows.Add(idx);
            }
        }
        Assert.Equal(Enumerable.Range(0, n), seenRows.OrderBy(v => v)); // every row exactly once
    }

    [Fact]
    public void DropLast_DropsTheShortTail()
    {
        var x = Range2D(10, 1);
        var dl = new DataLoader(new[] { x }, batchSize: 4, shuffle: false, dropLast: true);
        Assert.Equal(2, dl.BatchCount);
        var sizes = dl.Epoch().Select(b => b[0].Shape.Dims[0]).ToList();
        Assert.Equal(new[] { 4, 4 }, sizes);
    }

    [Fact]
    public void Batches_FlowGradientToSource()
    {
        var w = Range2D(6, 2).RequireGrad();
        var dl = new DataLoader(new[] { w }, batchSize: 3, shuffle: false);
        foreach (var batch in dl.Epoch())
            batch[0].Sum().Backward();
        // sum over all batches = sum over all elements -> grad 1 everywhere.
        Assert.All(w.Grad!.ToArray(), v => Assert.Equal(1f, v, 3));
    }
}
