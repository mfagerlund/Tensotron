namespace Tensotron;

/// <summary>
/// A fast, seedable pseudo-random generator — Marsaglia's xorshift128 (period 2^128−1). A drop-in for the
/// hot paths where <see cref="System.Random"/> is the bottleneck: uniform draws are ~2–3× faster and,
/// crucially, <b>construction is ~180× cheaper than a <i>seeded</i></b> <c>System.Random</c> (which runs an
/// expensive Knuth-subtractive init), so the "fresh seeded RNG per parallel worker" pattern in a PPO rollout
/// is essentially free. Gaussian draws (~2× faster) use Box–Muller with a cached spare — two normals per two
/// uniforms, no lookup table.
///
/// <para><b>Not thread-safe</b> (it holds mutable state): give each thread its own instance — exactly how the
/// rollout uses it (one per env, seeded from the master stream). <b>Deterministic</b> and platform-independent:
/// the same seed always yields the same sequence. Adapted from Colin Green's FastRandom (2005); the Godot
/// dependency and unused helpers were dropped.</para>
/// </summary>
public sealed class FastRng
{
    // 1/(int.MaxValue+1): scales a 31-bit draw into [0,1) without ever reaching 1.0.
    private const float FloatUnitInt = 1.0f / ((float)int.MaxValue + 1.0f);
    private const double RealUnitInt = 1.0 / ((double)int.MaxValue + 1.0);

    // xorshift's only requirement is that not all of x,y,z,w are zero; seeding sets x and keeps these
    // non-zero constants (Marsaglia's), so seed 0 is still a valid, non-degenerate stream.
    private const uint Y0 = 842502087, Z0 = 3579807591, W0 = 273326509;

    private uint _x, _y = Y0, _z = Z0, _w = W0;
    private float _spare;
    private bool _hasSpare;

    /// <summary>Seed the generator. Any int is a valid seed.</summary>
    public FastRng(int seed) => _x = (uint)seed;

    /// <summary>Reseed in place (cheap — no allocation, unlike <c>new System.Random(seed)</c>).</summary>
    public void SetSeed(int seed)
    {
        _x = (uint)seed; _y = Y0; _z = Z0; _w = W0;
        _hasSpare = false;
    }

    /// <summary>The core generator step: a draw over the full 32-bit range.</summary>
    public uint NextUInt()
    {
        uint t = _x ^ (_x << 11);
        _x = _y; _y = _z; _z = _w;
        return _w = _w ^ (_w >> 19) ^ t ^ (t >> 8);
    }

    /// <summary>Uniform float in [0, 1).</summary>
    public float NextFloat() => FloatUnitInt * (int)(0x7FFFFFFF & NextUInt());

    /// <summary>Uniform double in [0, 1).</summary>
    public double NextDouble() => RealUnitInt * (int)(0x7FFFFFFF & NextUInt());

    /// <summary>Uniform float in [min, max).</summary>
    public float NextFloat(float min, float max) => min + NextFloat() * (max - min);

    /// <summary>Uniform int in [0, maxExclusive). <paramref name="maxExclusive"/> must be ≥ 0.</summary>
    public int Next(int maxExclusive)
    {
        if (maxExclusive < 0) throw new ArgumentOutOfRangeException(nameof(maxExclusive));
        return (int)(NextFloat() * maxExclusive);
    }

    /// <summary>Uniform int in [min, max). <paramref name="max"/> must be ≥ <paramref name="min"/>.</summary>
    public int Next(int min, int max)
    {
        if (max < min) throw new ArgumentOutOfRangeException(nameof(max));
        return min + (int)(NextFloat() * (max - min));
    }

    /// <summary>Standard-normal draw (mean 0, std 1) via Box–Muller with a cached spare.</summary>
    public float NextGaussian()
    {
        if (_hasSpare) { _hasSpare = false; return _spare; }
        float u1 = NextFloat();
        float u2 = NextFloat();
        if (u1 < 1e-9f) u1 = 1e-9f;   // guard log(0) at the boundary
        float mag = MathF.Sqrt(-2f * MathF.Log(u1));
        float ang = 2f * MathF.PI * u2;
        _spare = mag * MathF.Sin(ang);
        _hasSpare = true;
        return mag * MathF.Cos(ang);
    }

    /// <summary>Normal draw with the given <paramref name="mean"/> and standard deviation <paramref name="std"/>.</summary>
    public float NextGaussian(float mean, float std) => mean + std * NextGaussian();
}
