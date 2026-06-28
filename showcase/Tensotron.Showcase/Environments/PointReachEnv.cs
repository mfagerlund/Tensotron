namespace Tensotron.Showcase.Environments;

/// <summary>
/// A trivial continuous-control task with a KNOWN analytic optimum, used to benchmark PPO
/// correctness rather than just "did it learn something". Each episode is a single step:
/// the state is a random target t in [-0.8, 0.8], the action a in [-1, 1], and the reward is
/// -(a - t)^2. The optimal policy is therefore the identity map a* = t, with optimal return 0.
///
/// Because episodes are one step long, a rollout is a single batched forward (not a long
/// sequential chain), so PPO converges to the optimum in well under a second — making this a
/// fast, always-on soundness gate for the actor-critic + GAE + clipped-surrogate machinery.
/// </summary>
public sealed class PointReachEnv : IEnvironment
{
    private readonly Random _rng;
    private float _target;

    public PointReachEnv(Random rng) => _rng = rng;

    public int StateSize => 1;
    public int ActionSize => 1;

    public float[] Reset()
    {
        _target = (float)(_rng.NextDouble() * 1.6 - 0.8); // U[-0.8, 0.8]
        return GetState();
    }

    public float[] GetState() => new[] { _target };

    public (float reward, bool done) Step(float action)
    {
        float d = action - _target;
        return (-(d * d), true); // single-step episode; optimum reward is 0 at a == t
    }

    // No spatial replay to draw for a 1-step bandit; the soundness test asserts on returns.
    public void RenderToSvg(string fileName) { }
}
