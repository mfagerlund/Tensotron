using System.Numerics;
using Tensotron.Showcase.Rendering;

namespace Tensotron.Showcase.Environments;

/// <summary>
/// Double pole-cart: a cart balancing two poles of different lengths at once — the classic
/// benchmark that's quick for NEAT but notoriously hard for gradient RL. Physics uses the
/// standard RK4 double-pole model. State = [x, xDot, θ1, θ1Dot, θ2, θ2Dot], continuous
/// force action in [-1,1]. Deterministic start (θ1 = 4°) — only the controller varies.
/// </summary>
public sealed class DoublePoleCart : IEnvironment
{
    private const float Gravity = -9.8f;
    private const float MassCart = 1.0f;
    private const float Length1 = 0.5f;
    private const float MassPole1 = 0.1f;
    private const float Length2 = 0.05f;
    private const float MassPole2 = 0.01f;
    private const float ForceMag = 10.0f;
    private const float TimeDelta = 0.01f;
    private const float Mup = 0.000002f;

    private const float Pi = MathF.PI;
    private const float FourDegrees = Pi / 45f;
    private const float ThirtySixDegrees = Pi / 5f;

    private readonly int _maxSteps;
    private readonly float _trackLengthHalf;
    private readonly float _poleAngleThreshold;
    private float[] _state = new float[6];

    public DoublePoleCart(int maxSteps = 1000, float trackLength = 4.8f, float poleAngleThreshold = ThirtySixDegrees)
    {
        _maxSteps = maxSteps;
        _trackLengthHalf = trackLength / 2f;
        _poleAngleThreshold = poleAngleThreshold;
    }

    public int StateSize => 6;
    public int ActionSize => 1;
    public int Steps { get; private set; }

    public List<float[]> RenderStateHistory { get; } = new();

    public bool IsOutOfBounds =>
        _state[0] < -_trackLengthHalf || _state[0] > _trackLengthHalf ||
        _state[2] > _poleAngleThreshold || _state[2] < -_poleAngleThreshold ||
        _state[4] > _poleAngleThreshold || _state[4] < -_poleAngleThreshold;

    public float[] Reset()
    {
        _state = new float[6];
        _state[2] = FourDegrees; // deterministic start: long pole tilted 4°
        Steps = 0;
        RenderStateHistory.Clear();
        return GetState();
    }

    public float[] GetState() => (float[])_state.Clone();

    public (float reward, bool done) Step(float action)
    {
        action = Math.Clamp(action, -1f, 1f);
        Steps++;
        var dydx = new float[6];
        for (int i = 0; i < 2; i++) // two RK4 sub-steps per tick
        {
            dydx[0] = _state[1];
            dydx[2] = _state[3];
            dydx[4] = _state[5];
            ComputeDerivs(action, _state, dydx);
            Rk4(action, _state, dydx, _state);
        }

        RenderStateHistory.Add((float[])_state.Clone());
        bool done = IsOutOfBounds || Steps >= _maxSteps;
        return (1f, done);
    }

    private static void ComputeDerivs(float action, float[] st, float[] derivs)
    {
        float force = action * ForceMag;
        float cos1 = MathF.Cos(st[2]), sin1 = MathF.Sin(st[2]), gsin1 = Gravity * sin1;
        float cos2 = MathF.Cos(st[4]), sin2 = MathF.Sin(st[4]), gsin2 = Gravity * sin2;

        float ml1 = Length1 * MassPole1;
        float ml2 = Length2 * MassPole2;
        float temp1 = Mup * st[3] / ml1;
        float temp2 = Mup * st[5] / ml2;

        float fi1 = ml1 * st[3] * st[3] * sin1 + 0.75f * MassPole1 * cos1 * (temp1 + gsin1);
        float fi2 = ml2 * st[5] * st[5] * sin2 + 0.75f * MassPole2 * cos2 * (temp2 + gsin2);

        float mi1 = MassPole1 * (1 - 0.75f * cos1 * cos1);
        float mi2 = MassPole2 * (1 - 0.75f * cos2 * cos2);

        derivs[1] = (force + fi1 + fi2) / (mi1 + mi2 + MassCart);
        derivs[3] = -0.75f * (derivs[1] * cos1 + gsin1 + temp1) / Length1;
        derivs[5] = -0.75f * (derivs[1] * cos2 + gsin2 + temp2) / Length2;
    }

    private static void Rk4(float action, float[] y, float[] dydx, float[] yout)
    {
        var dym = new float[6];
        var dyt = new float[6];
        var yt = new float[6];
        float hh = TimeDelta / 2, h6 = TimeDelta / 6;

        for (int i = 0; i <= 5; i++) yt[i] = y[i] + hh * dydx[i];
        ComputeDerivs(action, yt, dyt);
        dyt[0] = yt[1]; dyt[2] = yt[3]; dyt[4] = yt[5];

        for (int i = 0; i <= 5; i++) yt[i] = y[i] + hh * dyt[i];
        ComputeDerivs(action, yt, dym);
        dym[0] = yt[1]; dym[2] = yt[3]; dym[4] = yt[5];

        for (int i = 0; i <= 5; i++) { yt[i] = y[i] + TimeDelta * dym[i]; dym[i] += dyt[i]; }
        ComputeDerivs(action, yt, dyt);
        dyt[0] = yt[1]; dyt[2] = yt[3]; dyt[4] = yt[5];

        for (int i = 0; i <= 5; i++) yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2 * dym[i]);
    }

    public void RenderToSvg(string fileName)
    {
        var svg = new Svg();
        const float scale = 70f;
        foreach (var s in RenderStateHistory)
        {
            var baseP = new Vector2(s[0], 0);
            var tip1 = baseP + new Vector2(MathF.Cos(s[2] - Pi / 2), MathF.Sin(s[2] - Pi / 2)) * Length1;
            var tip2 = baseP + new Vector2(MathF.Cos(s[4] - Pi / 2), MathF.Sin(s[4] - Pi / 2)) * Length2;
            svg.AddLine(baseP * scale, tip1 * scale).SetStroke("steelblue").SetStrokeWidth(1f);
            svg.AddLine(baseP * scale, tip2 * scale).SetStroke("black").SetStrokeWidth(1.4f);
        }
        svg.AddLine(new Vector2(-_trackLengthHalf, 0) * scale, new Vector2(_trackLengthHalf, 0) * scale)
            .SetStroke("black").SetStrokeWidth(1f);
        svg.Save(fileName);
    }
}
