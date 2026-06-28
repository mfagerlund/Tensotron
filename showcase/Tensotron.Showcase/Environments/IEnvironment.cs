namespace Tensotron.Showcase.Environments;

/// <summary>
/// Minimal continuous-control environment contract for the PPO showcase. One scalar
/// action in [-1,1]; reward is +1 per surviving step; episodes end when the agent leaves
/// the valid region or hits the step cap.
/// </summary>
public interface IEnvironment
{
    int StateSize { get; }
    int ActionSize { get; }

    /// <summary>Reset to a fresh start state and return it.</summary>
    float[] Reset();

    float[] GetState();

    /// <summary>Advance one step. Returns the reward and whether the episode terminated.</summary>
    (float reward, bool done) Step(float action);

    /// <summary>Write an SVG replay of the state history recorded since the last Reset.</summary>
    void RenderToSvg(string fileName);
}
