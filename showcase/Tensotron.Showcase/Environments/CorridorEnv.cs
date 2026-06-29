using System.Numerics;
using Tensotron.Showcase.Rendering;

namespace Tensotron.Showcase.Environments;

/// <summary>
/// Corridor / track follower. A point "car" drives at constant speed around a closed wavy
/// loop; steering is the only control. It senses the walls through five raycast "whiskers"
/// fanned ahead and must keep itself inside the corridor band. Reward is +1 per surviving step
/// (as in the pole-cart), so survival ⟺ taking the bends: the geometry forbids circling in
/// place — the tightest turn the car can make (radius Speed/MaxTurn = 1.25) is wider than the
/// corridor (HalfWidth 0.9), so a closed loop won't fit inside the band. The reactive
/// whisker→steer policy that solves this is exactly what PPO discovers, and it drives laps.
/// </summary>
public sealed class CorridorEnv : IEnvironment
{
    // --- track geometry: closed wavy ring r(φ) = BaseRadius + Waviness·sin(Lobes·φ) ---
    private const float BaseRadius = 6f;
    private const float Waviness = 1.0f;
    private const int Lobes = 3;
    private const float HalfWidth = 0.9f;
    private const int Samples = 240;

    // --- car dynamics (constant forward speed; steer only) ---
    private const float Speed = 0.10f;     // world units / step
    private const float MaxTurn = 0.08f;   // rad / step at full steering → min radius 1.25 > HalfWidth
    private const float SensorRange = 3.0f;
    private static readonly float[] WhiskerAngles = { -1.0f, -0.5f, 0f, 0.5f, 1.0f };

    private readonly Random _rng;
    private readonly int _maxSteps;

    private readonly Vector2[] _center = new Vector2[Samples];
    private readonly Vector2[] _wallA = new Vector2[Samples];
    private readonly Vector2[] _wallB = new Vector2[Samples];
    private readonly Vector2[] _midA = new Vector2[Samples];
    private readonly Vector2[] _midB = new Vector2[Samples];

    private Vector2 _pos;
    private float _heading;

    public int Steps { get; private set; }
    public List<Vector2> Trajectory { get; } = new();

    public int StateSize => WhiskerAngles.Length;
    public int ActionSize => 1;

    public CorridorEnv(Random rng, int maxSteps = 800)
    {
        _rng = rng;
        _maxSteps = maxSteps;
        BuildTrack();
    }

    private void BuildTrack()
    {
        for (int i = 0; i < Samples; i++)
        {
            float phi = i * 2f * MathF.PI / Samples;
            float r = BaseRadius + Waviness * MathF.Sin(Lobes * phi);
            _center[i] = new Vector2(r * MathF.Cos(phi), r * MathF.Sin(phi));
        }
        for (int i = 0; i < Samples; i++)
        {
            var nrm = LeftNormal(i);
            _wallA[i] = _center[i] - nrm * HalfWidth;
            _wallB[i] = _center[i] + nrm * HalfWidth;
        }
        for (int i = 0; i < Samples; i++)
        {
            _midA[i] = (_wallA[i] + _wallA[(i + 1) % Samples]) * 0.5f;
            _midB[i] = (_wallB[i] + _wallB[(i + 1) % Samples]) * 0.5f;
        }
    }

    private Vector2 Tangent(int i)
    {
        var prev = _center[(i - 1 + Samples) % Samples];
        var next = _center[(i + 1) % Samples];
        return Vector2.Normalize(next - prev);
    }

    private Vector2 LeftNormal(int i)
    {
        var t = Tangent(i);
        return new Vector2(-t.Y, t.X);
    }

    public float[] Reset()
    {
        int i0 = _rng.Next(Samples);
        var tan = Tangent(i0);
        _heading = MathF.Atan2(tan.Y, tan.X) + (float)(_rng.NextDouble() - 0.5) * 0.3f;
        var nrm = new Vector2(-tan.Y, tan.X);
        _pos = _center[i0] + nrm * (float)((_rng.NextDouble() - 0.5) * HalfWidth * 0.6f);
        Steps = 0;
        Trajectory.Clear();
        Trajectory.Add(_pos);
        return GetState();
    }

    public float[] GetState()
    {
        var s = new float[WhiskerAngles.Length];
        for (int k = 0; k < WhiskerAngles.Length; k++)
        {
            float ang = _heading + WhiskerAngles[k];
            var dir = new Vector2(MathF.Cos(ang), MathF.Sin(ang));
            float dist = RayWallDistance(_pos, dir);
            s[k] = 1f - dist / SensorRange; // proximity: 1 = wall at the bumper, 0 = clear
        }
        return s;
    }

    public (float reward, bool done) Step(float action)
    {
        action = Math.Clamp(action, -1f, 1f);
        Steps++;
        _heading += action * MaxTurn;
        _pos += new Vector2(MathF.Cos(_heading), MathF.Sin(_heading)) * Speed;
        Trajectory.Add(_pos);

        bool crashed = DistanceToCenterline(_pos) > HalfWidth;
        bool done = crashed || Steps >= _maxSteps;
        return (crashed ? 0f : 1f, done); // survival reward; the crashing step earns nothing
    }

    private float RayWallDistance(Vector2 o, Vector2 d)
    {
        float cullR2 = (SensorRange + 0.3f) * (SensorRange + 0.3f);
        float best = SensorRange;
        best = MinHit(_wallA, _midA, o, d, best, cullR2);
        best = MinHit(_wallB, _midB, o, d, best, cullR2);
        return best;
    }

    private static float MinHit(Vector2[] wall, Vector2[] mid, Vector2 o, Vector2 d, float best, float cullR2)
    {
        int n = wall.Length;
        for (int i = 0; i < n; i++)
        {
            if (Vector2.DistanceSquared(o, mid[i]) > cullR2) continue; // broad-phase: skip far segments
            float t = RaySegment(o, d, wall[i], wall[(i + 1) % n]);
            if (t > 0 && t < best) best = t;
        }
        return best;
    }

    // Ray o + t·d (t ≥ 0, d unit ⇒ t is distance) vs segment a→b. Returns t at the hit, else -1.
    private static float RaySegment(Vector2 o, Vector2 d, Vector2 a, Vector2 b)
    {
        Vector2 e = b - a;
        float denom = d.X * e.Y - d.Y * e.X;
        if (MathF.Abs(denom) < 1e-9f) return -1f;
        Vector2 ao = a - o;
        float t = (ao.X * e.Y - ao.Y * e.X) / denom;
        float u = (ao.X * d.Y - ao.Y * d.X) / denom;
        return (t >= 0f && u >= 0f && u <= 1f) ? t : -1f;
    }

    private float DistanceToCenterline(Vector2 p)
    {
        float best = float.MaxValue;
        for (int i = 0; i < Samples; i++)
        {
            float dd = Vector2.DistanceSquared(p, _center[i]);
            if (dd < best) best = dd;
        }
        return MathF.Sqrt(best);
    }

    public void RenderToSvg(string fileName)
    {
        const float S = 40f;
        var svg = new Svg();
        DrawClosed(svg, _wallA, S);
        DrawClosed(svg, _wallB, S);
        int n = Trajectory.Count;
        for (int i = 1; i < n; i++)
        {
            float t = n > 2 ? (float)(i - 1) / (n - 2) : 0f;
            svg.AddLine(Trajectory[i - 1] * S, Trajectory[i] * S).SetStroke(Ramp(t)).SetStrokeWidth(2.0f);
        }
        if (n > 0)
            svg.AddCircle(Trajectory[0] * S, 4f).SetStroke("#222222").SetStrokeWidth(1.5f).SetFill("#ffffff");
        svg.Save(fileName);
    }

    private static void DrawClosed(Svg svg, Vector2[] pts, float s)
    {
        for (int i = 0; i < pts.Length; i++)
            svg.AddLine(pts[i] * s, pts[(i + 1) % pts.Length] * s).SetStroke("#b0b4ba").SetStrokeWidth(1.2f);
    }

    // Time gradient teal → amber → orange (matches the README performance charts).
    private static string Ramp(float t)
    {
        float r, g, b;
        if (t < 0.5f) { float u = t * 2f; r = Lerp(0x2a, 0xe9, u); g = Lerp(0x9d, 0xc4, u); b = Lerp(0x8f, 0x6a, u); }
        else { float u = (t - 0.5f) * 2f; r = Lerp(0xe9, 0xe7, u); g = Lerp(0xc4, 0x6f, u); b = Lerp(0x6a, 0x51, u); }
        return $"#{(int)r:x2}{(int)g:x2}{(int)b:x2}";
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}
