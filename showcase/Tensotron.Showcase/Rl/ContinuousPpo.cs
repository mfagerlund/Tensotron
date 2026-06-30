using Tensotron.Showcase.Environments;

namespace Tensotron.Showcase.Rl;

/// <summary>
/// Continuous-action PPO (clipped surrogate + GAE), built entirely on Tensotron tensors.
/// Mirrors the standard actor-critic PPO: collect fixed-horizon rollouts from parallel
/// environments, estimate advantages with GAE, then take several clipped-surrogate epochs
/// over minibatches. Rollouts run launch-free from a one-shot CPU weight snapshot; only the
/// update builds an autograd graph.
///
/// <para>Two robustness features keep long runs stable. <b>Value-target normalization:</b> the
/// critic regresses onto <c>return / σ_ret</c> (a running RMS of returns, <see cref="ReturnScale"/>)
/// and its output is scaled back up wherever GAE consumes it — so the value target stays ~unit scale
/// and can't random-walk into a loss-exploding regime, while advantages (and thus the policy
/// objective) are byte-for-byte unchanged. <b>Crash guards:</b> the Gaussian log-σ is clamped to
/// [<see cref="LogStdMin"/>,<see cref="LogStdMax"/>] at sampling and in the update, a non-finite-loss
/// minibatch is skipped rather than back-propagated, and the optimizer steps only on a finite global
/// grad norm (<see cref="LastSkippedUpdates"/> counts any guarded minibatch).</para>
/// </summary>
public sealed class ContinuousPpo
{
    private static readonly float Log2Pi = MathF.Log(2f * MathF.PI);
    private static readonly float HalfLog2PiE = 0.5f * MathF.Log(2f * MathF.PI * MathF.E);

    /// <summary>Safe band for the Gaussian log-σ, applied identically when sampling the rollout and in
    /// the autograd update. Keeps σ ∈ [e^-5, e^2] ≈ [0.0067, 7.4] so the log-prob never divides by a
    /// near-zero σ (→ ±∞) nor explodes it — the two numeric blow-up paths in continuous PPO. Clamping at
    /// point of use also zeroes the gradient outside the band, so log-σ self-limits instead of drifting.</summary>
    public const float LogStdMin = -5f;
    public const float LogStdMax = 2f;

    public float Gamma = 0.99f;
    public float Lambda = 0.95f;
    public float ClipEps = 0.2f;
    public int Epochs = 10;
    public int MinibatchSize = 512;
    public float LearningRate = 2e-3f;
    public float EntropyCoef = 0.0f;
    public float ValueCoef = 0.5f;
    public float MaxGradNorm = 0.5f;
    public int NumEnvs = 16;
    public int Horizon = 256;

    private readonly ActorCritic _ac;
    private readonly Func<IEnvironment> _envFactory;
    private readonly Adam _opt;
    private readonly Random _rng;

    private float _retStd = 1f;          // running RMS of returns: the value head learns ret/_retStd (bounded)
    private bool _retStdInit;
    private const float RetStdBeta = 0.1f;

    /// <summary>Mean total PPO loss across applied minibatches of the most recent iteration (NaN if all skipped).</summary>
    public float LastLoss { get; private set; }
    /// <summary>Minibatch updates the non-finite crash guard skipped in the most recent iteration (0 = healthy).</summary>
    public int LastSkippedUpdates { get; private set; }
    /// <summary>Running RMS of returns the value targets are normalized by (keeps the critic from diverging).</summary>
    public float ReturnScale => _retStd;

    public ContinuousPpo(ActorCritic ac, Func<IEnvironment> envFactory, Random rng)
    {
        _ac = ac;
        _envFactory = envFactory;
        _rng = rng;
        _opt = new Adam(ac.Parameters(), lr: LearningRate);

        // A single minibatch update (3-layer MLP forward+backward + PPO loss + clip + Adam) is
        // well under 1024 launches and makes no host pull. FlushEvery=1024 spans a whole update so
        // the safety drain doesn't sync mid-update; drains still fire across minibatches, keeping
        // the in-flight queue bounded. Pure MLP: no parked data-dependent index buffers, so nothing
        // accumulates between drains.
        TensorRuntime.Instance.FlushEvery = 1024;
    }

    private float NextGaussian()
    {
        // Box–Muller
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>Run training for the given number of PPO iterations. The callback (if any)
    /// receives (iteration, meanRolloutEpisodeReturn) for logging.</summary>
    public void Train(int iterations, Action<int, float>? onIteration = null)
    {
        _opt.LearningRate = LearningRate;
        int S = _ac.StateSize, A = _ac.ActionSize;
        var envs = Enumerable.Range(0, NumEnvs).Select(_ => _envFactory()).ToArray();
        var cur = envs.Select(e => e.Reset()).ToArray();

        for (int iter = 0; iter < iterations; iter++)
        {
            int T = Horizon, E = NumEnvs, B = T * E;
            var bStates = new float[B * S];
            var bActions = new float[B * A];
            var bLogp = new float[B];
            var bRewards = new float[B];
            var bValues = new float[B];
            var bDones = new float[B];

            var logStd = _ac.LogStd.ToArray();
            ClampLogStd(logStd);                                  // sample with the enforced σ band
            var std = logStd.Select(MathF.Exp).ToArray();

            float epReturnSum = 0f; int epCount = 0;
            var running = new float[E];

            // ---- rollout: launch-free CPU inference from a one-shot weight snapshot ----
            // (per-step device forwards would be hundreds of tiny launch+sync round-trips).
            var cpu = _ac.SnapshotCpu();
            for (int t = 0; t < T; t++)
            {
                var (means, values) = ForwardCpu(cpu, cur, E);
                for (int e = 0; e < E; e++)
                {
                    int row = (t * E + e);
                    Array.Copy(cur[e], 0, bStates, row * S, S);
                    bValues[row] = values[e] * _retStd;          // value head is normalized; raw units for GAE

                    float logp = 0f;
                    for (int k = 0; k < A; k++)
                    {
                        float mean = means[e * A + k];
                        float act = mean + std[k] * NextGaussian();
                        act = Math.Clamp(act, -1f, 1f);
                        bActions[row * A + k] = act;
                        float d = (act - mean) / std[k];
                        logp += -0.5f * d * d - logStd[k] - 0.5f * Log2Pi;
                    }
                    bLogp[row] = logp;

                    float action = bActions[row * A]; // single-action envs
                    var (reward, done) = envs[e].Step(action);
                    bRewards[row] = reward;
                    bDones[row] = done ? 1f : 0f;
                    running[e] += reward;

                    if (done)
                    {
                        epReturnSum += running[e]; epCount++;
                        running[e] = 0f;
                        cur[e] = envs[e].Reset();
                    }
                    else
                    {
                        cur[e] = envs[e].GetState();
                    }
                }
            }

            // bootstrap value for the final state of each env (same snapshot)
            var (_, finalValues) = ForwardCpu(cpu, cur, E);
            for (int e = 0; e < E; e++) finalValues[e] *= _retStd;  // raw-return units, like bValues

            // ---- GAE advantages + returns ----
            var adv = new float[B];
            var ret = new float[B];
            for (int e = 0; e < E; e++)
            {
                float lastGae = 0f;
                for (int t = T - 1; t >= 0; t--)
                {
                    int row = t * E + e;
                    float nextValue = t == T - 1 ? finalValues[e] : bValues[(t + 1) * E + e];
                    float nextNonTerminal = 1f - bDones[row];
                    float delta = bRewards[row] + Gamma * nextValue * nextNonTerminal - bValues[row];
                    lastGae = delta + Gamma * Lambda * nextNonTerminal * lastGae;
                    adv[row] = lastGae;
                    ret[row] = lastGae + bValues[row];
                }
            }

            // recalibrate the running return scale from THIS batch's raw returns, then put the value
            // target into normalized units below (the critic regresses onto ret/σ, which stays ~unit
            // scale, so the value loss can't drift to huge — the divergence the raw MSE target invites).
            UpdateReturnScale(ret);

            // normalize advantages (unchanged — advantages are never divided by σ, so the policy
            // objective is byte-for-byte what it was; only the value target moves to unit scale).
            float advMean = adv.Average();
            float advStd = MathF.Sqrt(adv.Select(x => (x - advMean) * (x - advMean)).Sum() / adv.Length) + 1e-8f;
            for (int i = 0; i < B; i++) adv[i] = (adv[i] - advMean) / advStd;
            for (int i = 0; i < B; i++) ret[i] /= _retStd;

            // ---- PPO update epochs ----
            var idx = Enumerable.Range(0, B).ToArray();
            float lossSum = 0f; int lossCount = 0, skipped = 0;
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                Shuffle(idx);
                for (int start = 0; start < B; start += MinibatchSize)
                {
                    int mb = Math.Min(MinibatchSize, B - start);
                    var mbStates = new float[mb * S];
                    var mbActions = new float[mb * A];
                    var mbLogp = new float[mb];
                    var mbAdv = new float[mb];
                    var mbRet = new float[mb];
                    for (int j = 0; j < mb; j++)
                    {
                        int r = idx[start + j];
                        Array.Copy(bStates, r * S, mbStates, j * S, S);
                        Array.Copy(bActions, r * A, mbActions, j * A, A);
                        mbLogp[j] = bLogp[r];
                        mbAdv[j] = adv[r];
                        mbRet[j] = ret[r];
                    }
                    float l = UpdateMinibatch(mb, S, A, mbStates, mbActions, mbLogp, mbAdv, mbRet, out bool stepped);
                    if (stepped && float.IsFinite(l)) { lossSum += l; lossCount++; } else skipped++;
                }
            }
            LastLoss = lossCount > 0 ? lossSum / lossCount : float.NaN;
            LastSkippedUpdates = skipped;

            float meanReturn = epCount > 0 ? epReturnSum / epCount : running.Average();
            onIteration?.Invoke(iter, meanReturn);
        }
    }

    // Launch-free rollout inference: evaluate the snapshotted policy+value on the host for E
    // states. No tensors, no kernel launches, no Synchronize — the whole point of the split.
    private static (float[] means, float[] values) ForwardCpu(CpuActorCritic cpu, float[][] states, int E)
    {
        int A = cpu.ActionSize;
        var means = new float[E * A];
        var values = new float[E];
        var m = new float[A];
        for (int e = 0; e < E; e++)
        {
            cpu.Forward(states[e], m, out float v);
            for (int k = 0; k < A; k++) means[e * A + k] = m[k];
            values[e] = v;
        }
        return (means, values);
    }

    // One clipped-surrogate + value + entropy step. Returns the scalar total loss and sets
    // <paramref name="stepped"/> to whether the optimizer actually stepped. The two guards are the crash
    // protection: a non-finite loss is never back-propagated, and a step is applied only when the
    // (post-clip) global grad norm is finite — so a single diverged/NaN minibatch can't write NaN into
    // the weights and poison every later forward pass.
    private float UpdateMinibatch(int mb, int S, int A,
        float[] states, float[] actions, float[] oldLogp, float[] adv, float[] ret, out bool stepped)
    {
        var st = Tensor.FromShaped(states, new[] { mb, S });
        var actT = Tensor.FromShaped(actions, new[] { mb, A });
        var oldLogpT = Tensor.FromShaped(oldLogp, new[] { mb });
        var advT = Tensor.FromShaped(adv, new[] { mb });
        var retT = Tensor.FromShaped(ret, new[] { mb });

        var mean = _ac.PolicyMean(st);                       // (mb, A)
        var logStd = _ac.LogStd.Clamp(LogStdMin, LogStdMax); // (A,) bound σ so the Gaussian stays finite
        var stdT = logStd.Exp();                             // (A,)

        // log N(act; mean, std) summed over action dims -> (mb,)
        var diff = (actT - mean) / stdT;                     // (mb, A) broadcast std
        var logpTerms = -0.5f * diff.Square() - logStd - 0.5f * Log2Pi;
        var logp = logpTerms.Sum(new[] { 1 });               // (mb,)

        var ratio = (logp - oldLogpT).Exp();                 // (mb,)
        var surr1 = ratio * advT;
        var surr2 = ratio.Clamp(1f - ClipEps, 1f + ClipEps) * advT;
        var policyLoss = TensorOps.Minimum(surr1, surr2).Mean().Neg();

        var value = _ac.Value(st);                           // (mb,)
        var valueLoss = (value - retT).Square().Mean();

        // differential entropy of the Gaussian policy = sum_k (logStd_k + 0.5 log 2πe)
        var entropy = (logStd + HalfLog2PiE).Sum();

        var loss = policyLoss + ValueCoef * valueLoss - EntropyCoef * entropy;
        float lossValue = loss.ToArray()[0];

        _opt.ZeroGrad();
        if (!float.IsFinite(lossValue))
        {
            // A non-finite loss would back-propagate NaN/∞ into every weight. Skip the whole minibatch;
            // grads stay zeroed and the next minibatch (or iteration) trains from intact weights.
            stepped = false;
            return lossValue;
        }

        loss.Backward();
        // ∞ grads were already scaled toward 0 by the clip, but a NaN norm means NaN grads — applying
        // either (especially NaN) would corrupt the weights, so step only on a finite norm.
        float gradNorm = GradUtils.ClipGradNorm(_ac.Parameters(), MaxGradNorm, returnTotalNorm: true);
        stepped = float.IsFinite(gradNorm);
        if (stepped) _opt.Step();
        return lossValue;
    }

    /// <summary>Greedy evaluation (mean action, no exploration). Returns the average number
    /// of steps survived across the given number of episodes.</summary>
    public float EvaluateMeanSteps(int episodes, int maxSteps)
    {
        var cpu = _ac.SnapshotCpu();
        var mean = new float[cpu.ActionSize];
        float total = 0f;
        for (int ep = 0; ep < episodes; ep++)
        {
            var env = _envFactory();
            var s = env.Reset();
            int steps = 0;
            for (; steps < maxSteps; steps++)
            {
                cpu.Forward(s, mean, out _);
                var (_, done) = env.Step(Math.Clamp(mean[0], -1f, 1f));
                s = env.GetState();
                if (done) { steps++; break; }
            }
            total += steps;
        }
        return total / episodes;
    }

    private void Shuffle(int[] a)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = _rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }

    // Clamp a host log-σ array into the LogStdMin..LogStdMax band in place, so the noise the rollout
    // samples with matches the band the autograd update enforces.
    private static void ClampLogStd(float[] logStd)
    {
        for (int i = 0; i < logStd.Length; i++)
            logStd[i] = Math.Clamp(logStd[i], LogStdMin, LogStdMax);
    }

    // Track the running RMS of returns so value targets stay ~unit scale (the critic can't diverge).
    // Calibrated directly on the first healthy batch, then EMA; a NaN/degenerate batch keeps the prior.
    private void UpdateReturnScale(float[] ret)
    {
        float rms = ReturnRms(ret);
        if (!float.IsFinite(rms) || rms <= 0f) return;
        _retStd = _retStdInit ? (1f - RetStdBeta) * _retStd + RetStdBeta * rms : rms;
        _retStd = MathF.Max(_retStd, 1e-4f);
        _retStdInit = true;
    }

    // sqrt(mean(ret^2)), accumulated in double — the scale value targets are normalized by.
    private static float ReturnRms(float[] ret)
    {
        double sumSq = 0;
        for (int i = 0; i < ret.Length; i++) sumSq += (double)ret[i] * ret[i];
        return (float)Math.Sqrt(sumSq / ret.Length);
    }
}
