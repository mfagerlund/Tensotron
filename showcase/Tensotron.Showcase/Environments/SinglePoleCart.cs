using System.Numerics;
using Tensotron.Showcase.Rendering;

namespace Tensotron.Showcase.Environments;

/// <summary>
/// Classic single pole-cart (cart-pole). Physics follows the standard cart-pole model
/// (www.igi.tugraz.at/lehre/MLA/WS99/pole.c). State = [cartX, cartV, poleAngle, poleAngV],
/// continuous force action in [-1,1]. Self-contained — no external dependencies.
/// </summary>
public sealed class SinglePoleCart : IEnvironment
{
    private const float Gravity = 9.8f;
    private const float MassCart = 1.0f;
    private const float MassPole = 0.1f;
    private const float TotalMass = MassPole + MassCart;
    private const float Length = 0.5f; // half the pole's length
    private const float PoleMassLength = MassPole * Length;
    private const float ForceMag = 10.0f;
    private const float FourThirds = 4f / 3f;
    private const float Tau = 0.02f;
    private const float RailLengthHalf = 2.4f;
    private const float MaxAngleRad = 12f * MathF.PI / 180f;

    private readonly Random _rng;
    private readonly int _maxSteps;

    public SinglePoleCart(Random rng, int maxSteps = 500)
    {
        _rng = rng;
        _maxSteps = maxSteps;
    }

    public int StateSize => 4;
    public int ActionSize => 1;

    public float CartPosition { get; private set; }
    public float CartSpeed { get; private set; }
    public float PoleAngle { get; private set; }
    public float PoleAngleSpeed { get; private set; }
    public int Steps { get; private set; }

    public List<float[]> RenderStateHistory { get; } = new();

    private float Uniform(float lo, float hi) => lo + (float)_rng.NextDouble() * (hi - lo);

    public bool IsOutOfBounds =>
        MathF.Abs(CartPosition) > RailLengthHalf || MathF.Abs(PoleAngle) > MaxAngleRad;

    public float[] Reset()
    {
        CartPosition = Uniform(-0.05f, 0.05f);
        CartSpeed = Uniform(-0.05f, 0.05f);
        PoleAngle = Uniform(-0.05f, 0.05f);
        PoleAngleSpeed = Uniform(-0.05f, 0.05f);
        Steps = 0;
        RenderStateHistory.Clear();
        return GetState();
    }

    public float[] GetState() => new[] { CartPosition, CartSpeed, PoleAngle, PoleAngleSpeed };

    public (float reward, bool done) Step(float action)
    {
        action = Math.Clamp(action, -1f, 1f);
        Steps++;
        float force = action * ForceMag;
        float cos = MathF.Cos(PoleAngle);
        float sin = MathF.Sin(PoleAngle);

        float temp = (force + PoleMassLength * PoleAngleSpeed * PoleAngleSpeed * sin) / TotalMass;
        float thetaAcc = (Gravity * sin - cos * temp) /
                         (Length * (FourThirds - MassPole * cos * cos / TotalMass));
        float xAcc = temp - PoleMassLength * thetaAcc * cos / TotalMass;

        CartPosition += Tau * CartSpeed;
        CartSpeed += Tau * xAcc;
        PoleAngle += Tau * PoleAngleSpeed;
        PoleAngleSpeed += Tau * thetaAcc;

        RenderStateHistory.Add(GetState());

        bool done = IsOutOfBounds || Steps >= _maxSteps;
        return (1f, done); // +1 per surviving step
    }

    public void RenderToSvg(string fileName)
    {
        var svg = new Svg();
        const float scale = 100f;
        foreach (var s in RenderStateHistory)
        {
            var start = new Vector2(s[0], 0);
            var tip = start + new Vector2(MathF.Cos(s[2] - MathF.PI / 2), MathF.Sin(s[2] - MathF.PI / 2)) * Length;
            svg.AddLine(start * scale, tip * scale).SetStrokeWidth(1f).SetStroke("steelblue");
        }
        svg.AddLine(new Vector2(-RailLengthHalf, 0) * scale, new Vector2(RailLengthHalf, 0) * scale)
            .SetStroke("black").SetStrokeWidth(1f);
        svg.Save(fileName);
    }
}
