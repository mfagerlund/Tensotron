namespace Tensotron;

/// <summary>
/// Struct-generic elementwise operator. A single elementwise kernel is parameterized
/// by one of these structs, so each op inlines at JIT time. (ILGPU kernels can't take
/// a `Func`/delegate, so the operation is carried as a struct type parameter.)
/// </summary>
public interface IBinaryOp
{
    float Apply(float a, float b);
}

public readonly struct AddOp : IBinaryOp
{
    public float Apply(float a, float b) => a + b;
}

public readonly struct MulOp : IBinaryOp
{
    public float Apply(float a, float b) => a * b;
}

public readonly struct SubOp : IBinaryOp
{
    public float Apply(float a, float b) => a - b;
}

public readonly struct DivOp : IBinaryOp
{
    public float Apply(float a, float b) => a / b;
}

public readonly struct PowOp : IBinaryOp
{
    public float Apply(float a, float b) => ILGPU.Algorithms.XMath.Pow(a, b);
}

public readonly struct MaximumOp : IBinaryOp
{
    public float Apply(float a, float b) => a > b ? a : b;
}

public readonly struct MinimumOp : IBinaryOp
{
    public float Apply(float a, float b) => a < b ? a : b;
}

// Comparisons return 1f/0f and carry no gradient.
public readonly struct GtOp : IBinaryOp { public float Apply(float a, float b) => a > b ? 1f : 0f; }
public readonly struct GeOp : IBinaryOp { public float Apply(float a, float b) => a >= b ? 1f : 0f; }
public readonly struct LtOp : IBinaryOp { public float Apply(float a, float b) => a < b ? 1f : 0f; }
public readonly struct LeOp : IBinaryOp { public float Apply(float a, float b) => a <= b ? 1f : 0f; }
public readonly struct EqOp : IBinaryOp { public float Apply(float a, float b) => a == b ? 1f : 0f; }
public readonly struct NeOp : IBinaryOp { public float Apply(float a, float b) => a != b ? 1f : 0f; }
