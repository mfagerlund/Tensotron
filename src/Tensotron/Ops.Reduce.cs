namespace Tensotron;

/// <summary>Struct-generic reduction operator (identity + associative combine).</summary>
public interface IReduceOp
{
    float Identity { get; }
    float Combine(float a, float b);
}

public readonly struct SumReduce : IReduceOp
{
    public float Identity => 0f;
    public float Combine(float a, float b) => a + b;
}

public readonly struct MaxReduce : IReduceOp
{
    public float Identity => float.NegativeInfinity;
    public float Combine(float a, float b) => a > b ? a : b;
}

public readonly struct MinReduce : IReduceOp
{
    public float Identity => float.PositiveInfinity;
    public float Combine(float a, float b) => a < b ? a : b;
}

public readonly struct ProdReduce : IReduceOp
{
    public float Identity => 1f;
    public float Combine(float a, float b) => a * b;
}
