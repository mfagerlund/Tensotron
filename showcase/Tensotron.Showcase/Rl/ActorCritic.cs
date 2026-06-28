namespace Tensotron.Showcase.Rl;

/// <summary>
/// Gaussian actor + value critic for continuous PPO. The actor maps state → action mean
/// (a small tanh MLP) and carries a state-independent learnable log-σ per action dim
/// (standard PPO practice). The critic is a separate tanh MLP mapping state → scalar value.
/// </summary>
public sealed class ActorCritic
{
    private readonly Sequential _policy;
    private readonly Sequential _value;

    /// <summary>Learnable log standard deviation, one per action dim (state-independent).</summary>
    public Tensor LogStd { get; }

    public int StateSize { get; }
    public int ActionSize { get; }

    public ActorCritic(int stateSize, int actionSize, int hidden = 64, float initLogStd = -0.5f)
    {
        StateSize = stateSize;
        ActionSize = actionSize;

        _policy = new Sequential(
            new Linear(stateSize, hidden), Activation.Tanh(),
            new Linear(hidden, hidden), Activation.Tanh(),
            new Linear(hidden, actionSize));

        _value = new Sequential(
            new Linear(stateSize, hidden), Activation.Tanh(),
            new Linear(hidden, hidden), Activation.Tanh(),
            new Linear(hidden, 1));

        var logStd = new float[actionSize];
        Array.Fill(logStd, initLogStd);
        LogStd = Tensor.FromShaped(logStd, new[] { actionSize }).RequireGrad();
    }

    /// <summary>Action mean for a batch of states: (B, stateSize) → (B, actionSize).</summary>
    public Tensor PolicyMean(Tensor states) => _policy.Forward(states);

    /// <summary>Value estimate for a batch of states: (B, stateSize) → (B,).</summary>
    public Tensor Value(Tensor states) => _value.Forward(states).Reshape(states.Shape.Dims[0]);

    public IReadOnlyList<Tensor> Parameters()
    {
        var ps = new List<Tensor>();
        ps.AddRange(_policy.Parameters());
        ps.AddRange(_value.Parameters());
        ps.Add(LogStd);
        return ps;
    }
}
