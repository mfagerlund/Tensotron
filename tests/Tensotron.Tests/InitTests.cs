using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

public class InitTests
{
    // Scale formulas (fan / gain / bound) match torch's recorded values exactly.
    [Fact]
    public void InitScales_MatchTorch()
    {
        var fix = Fixtures.Load("init");
        foreach (var c in fix.Cases)
        {
            var m = c.Meta!;
            var exp = c.Output.Data;
            switch (m.Op)
            {
                case "fan":
                {
                    var (fanIn, fanOut) = Init.FanInFanOut(new Shape(m.Dims!));
                    Assert.Equal(exp[0], fanIn, 3);
                    Assert.Equal(exp[1], fanOut, 3);
                    break;
                }
                case "gain":
                    Assert.Equal(exp[0], Init.CalculateGain(m.Reduction!, m.Params![0]), 4);
                    break;
                case "kaiming_uniform_bound":
                    Assert.Equal(exp[0], Init.KaimingUniformBound(new Shape(m.Dims!), m.Params![0],
                        "fan_in", m.Reduction!), 4);
                    break;
                case "xavier_uniform_bound":
                    Assert.Equal(exp[0], Init.XavierUniformBound(new Shape(m.Dims!), m.Params![0]), 4);
                    break;
                default:
                    throw new InvalidOperationException(m.Op);
            }
        }
    }

    // The fillers actually apply their scale: large-sample statistics match the formula.
    [Fact]
    public void Fillers_ProduceExpectedStatistics()
    {
        Init.Seed(7);
        var n = new Shape(200, 200);

        var normal = Init.Normal_(Tensor.Zeros(n), mean: 1f, std: 2f).ToArray();
        Assert.Equal(1f, Mean(normal), 1);
        Assert.Equal(2f, Std(normal), 1);

        var uni = Init.Uniform_(Tensor.Zeros(n), -3f, 3f).ToArray();
        Assert.All(uni, v => Assert.InRange(v, -3f, 3f));
        Assert.Equal(0f, Mean(uni), 1);

        // Kaiming-normal std should track gain/sqrt(fan_in).
        var w = new Shape(256, 128);
        float expected = Init.KaimingNormalStd(w, a: 0f, nonlinearity: "relu");
        var kn = Init.KaimingNormal_(Tensor.Zeros(w), a: 0f, nonlinearity: "relu").ToArray();
        Assert.Equal(expected, Std(kn), 2);
    }

    private static float Mean(float[] x) => x.Average();

    private static float Std(float[] x)
    {
        float m = x.Average();
        double s = x.Sum(v => (double)(v - m) * (v - m)) / x.Length;
        return (float)Math.Sqrt(s);
    }
}
