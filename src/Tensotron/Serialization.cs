namespace Tensotron;

/// <summary>
/// Parameter (state-dict) serialization. Writes a compact binary file of named tensors;
/// load matches by name into a module's parameters and copies in place (preserving leaf
/// identity, so optimizers already bound to the params keep working).
/// </summary>
public static class Serialization
{
    private const int Magic = 0x534E4554; // "TENS"

    /// <summary>Save a module's full state_dict — learnable parameters AND persistent buffers
    /// (e.g. BatchNorm running mean/var). torch saves both; omitting buffers silently breaks
    /// eval-after-load for normalization layers.</summary>
    public static void Save(Module module, string path)
        => SaveTensors(module.StateDict(), path);

    public static void SaveTensors(IEnumerable<(string name, Tensor param)> tensors, string path)
    {
        var list = tensors.ToList();
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);
        w.Write(Magic);
        w.Write(list.Count);
        foreach (var (name, t) in list)
        {
            w.Write(name);
            w.Write(t.Rank);
            foreach (var d in t.Shape.Dims) w.Write(d);
            var data = t.ToArray();
            w.Write(data.Length);
            foreach (var f in data) w.Write(f);
        }
    }

    /// <summary>Read a file into a name → tensor dictionary.</summary>
    public static Dictionary<string, Tensor> LoadTensors(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        if (r.ReadInt32() != Magic)
            throw new InvalidOperationException($"{path} is not a Tensotron state file.");
        int count = r.ReadInt32();
        var dict = new Dictionary<string, Tensor>(count);
        for (int i = 0; i < count; i++)
        {
            string name = r.ReadString();
            int rank = r.ReadInt32();
            var dims = new int[rank];
            for (int j = 0; j < rank; j++) dims[j] = r.ReadInt32();
            int len = r.ReadInt32();
            var data = new float[len];
            for (int j = 0; j < len; j++) data[j] = r.ReadSingle();
            dict[name] = Tensor.FromShaped(data, dims);
        }
        return dict;
    }

    /// <summary>
    /// Load parameters from <paramref name="path"/> into <paramref name="module"/> by name.
    /// strict (default) requires an exact key match in both directions.
    /// </summary>
    public static void Load(Module module, string path, bool strict = true)
    {
        var dict = LoadTensors(path);
        var seen = new HashSet<string>();
        foreach (var (name, p) in module.StateDict())
        {
            if (dict.TryGetValue(name, out var src))
            {
                if (!src.Shape.Equals(p.Shape))
                    throw new InvalidOperationException($"Param '{name}' shape {src.Shape} != model {p.Shape}.");
                p.CopyInPlace(src);
                seen.Add(name);
            }
            else if (strict)
            {
                throw new InvalidOperationException($"Missing parameter '{name}' in {path}.");
            }
        }
        if (strict && seen.Count != dict.Count)
        {
            var extra = dict.Keys.Where(k => !seen.Contains(k));
            throw new InvalidOperationException($"Unexpected parameters in {path}: {string.Join(", ", extra)}.");
        }
    }

    /// <summary>
    /// Save a full training checkpoint — model state_dict (params + buffers), optimizer state
    /// (momentum / Adam moments + step), and optional LR-scheduler state — into one file, so
    /// <see cref="LoadCheckpoint"/> resumes training exactly. Namespaces are prefixed
    /// (<c>model.</c> / <c>optim.</c> / <c>sched.</c>) so they never collide.
    /// </summary>
    public static void SaveCheckpoint(Module module, Optimizer optimizer, LrScheduler? scheduler, string path)
    {
        var all = new List<(string, Tensor)>();
        foreach (var (n, t) in module.StateDict()) all.Add(("model." + n, t));
        foreach (var (n, t) in optimizer.StateDict()) all.Add(("optim." + n, t));
        if (scheduler is not null)
            foreach (var (n, t) in scheduler.StateDict()) all.Add(("sched." + n, t));
        SaveTensors(all, path);
    }

    /// <summary>Restore a checkpoint written by <see cref="SaveCheckpoint"/> into a model + optimizer
    /// (+ optional scheduler) of the same shape. The model's params and buffers are required to be
    /// present; optimizer/scheduler state is applied if present.</summary>
    public static void LoadCheckpoint(Module module, Optimizer optimizer, LrScheduler? scheduler, string path)
    {
        var dict = LoadTensors(path);
        foreach (var (name, p) in module.StateDict())
        {
            if (!dict.TryGetValue("model." + name, out var src))
                throw new InvalidOperationException($"Missing model tensor '{name}' in checkpoint {path}.");
            if (!src.Shape.Equals(p.Shape))
                throw new InvalidOperationException($"Tensor '{name}' shape {src.Shape} != model {p.Shape}.");
            p.CopyInPlace(src);
        }
        optimizer.LoadStateDict(SubDict(dict, "optim."));
        scheduler?.LoadStateDict(SubDict(dict, "sched."));
    }

    private static Dictionary<string, Tensor> SubDict(Dictionary<string, Tensor> dict, string prefix)
        => dict.Where(kv => kv.Key.StartsWith(prefix))
               .ToDictionary(kv => kv.Key[prefix.Length..], kv => kv.Value);
}
