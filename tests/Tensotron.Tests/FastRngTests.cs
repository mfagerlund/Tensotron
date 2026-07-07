using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// <see cref="FastRng"/> — the xorshift128 generator used on the RNG hot paths. Proves the properties the
/// callers rely on: deterministic per seed (so seeded runs stay reproducible), in-range uniforms, and
/// standard-normal gaussians (mean 0, std 1) so it is a correct drop-in for the old System.Random Box–Muller.
/// </summary>
public class FastRngTests
{
    [Fact]
    public void Same_seed_yields_the_same_sequence()
    {
        var a = new FastRng(12345);
        var b = new FastRng(12345);
        for (int i = 0; i < 1000; i++)
        {
            Assert.Equal(a.NextUInt(), b.NextUInt());
        }
    }

    [Fact]
    public void SetSeed_resets_the_stream()
    {
        var r = new FastRng(1);
        var first = Enumerable.Range(0, 100).Select(_ => r.NextUInt()).ToArray();
        for (int i = 0; i < 50; i++) r.NextGaussian();   // advance + populate the gaussian spare
        r.SetSeed(1);
        var again = Enumerable.Range(0, 100).Select(_ => r.NextUInt()).ToArray();
        Assert.Equal(first, again);
    }

    [Fact]
    public void Different_seeds_diverge()
    {
        var a = new FastRng(1);
        var b = new FastRng(2);
        int matches = Enumerable.Range(0, 100).Count(_ => a.NextUInt() == b.NextUInt());
        Assert.True(matches < 5, $"streams from different seeds should not track each other ({matches}/100 matched)");
    }

    [Fact]
    public void NextFloat_stays_in_unit_interval()
    {
        var r = new FastRng(7);
        for (int i = 0; i < 1_000_000; i++)
        {
            float v = r.NextFloat();
            Assert.InRange(v, 0f, 0.9999999f);
        }
    }

    [Fact]
    public void NextFloat_range_respects_bounds()
    {
        var r = new FastRng(99);
        for (int i = 0; i < 100_000; i++) Assert.InRange(r.NextFloat(-3f, 5f), -3f, 5f);
    }

    [Fact]
    public void Next_int_is_in_range()
    {
        var r = new FastRng(3);
        for (int i = 0; i < 100_000; i++)
        {
            Assert.InRange(r.Next(10), 0, 9);
            Assert.InRange(r.Next(5, 8), 5, 7);
        }
        Assert.Equal(0, r.Next(0));   // empty range is a valid no-op
    }

    [Fact]
    public void Gaussian_has_unit_mean_and_variance()
    {
        var r = new FastRng(2024);
        const int n = 4_000_000;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < n; i++)
        {
            float g = r.NextGaussian();
            sum += g; sumSq += (double)g * g;
        }
        double mean = sum / n;
        double var = sumSq / n - mean * mean;
        Assert.Equal(0.0, mean, 2);   // ~0 to 2 decimals
        Assert.Equal(1.0, var, 2);    // ~1 to 2 decimals
    }

    [Fact]
    public void Gaussian_applies_mean_and_std()
    {
        var r = new FastRng(55);
        const int n = 2_000_000;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < n; i++)
        {
            float g = r.NextGaussian(mean: 3f, std: 2f);
            sum += g; sumSq += (double)g * g;
        }
        double mean = sum / n;
        double std = Math.Sqrt(sumSq / n - mean * mean);
        Assert.Equal(3.0, mean, 1);
        Assert.Equal(2.0, std, 1);
    }
}
