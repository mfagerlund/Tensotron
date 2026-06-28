namespace Tensotron;

/// <summary>
/// Parameter (state-dict) serialization. Writes a compact binary file of named tensors;
/// load matches by name into a module's parameters and copies in place (preserving leaf
/// identity, so optimizers already bound to the params keep working).
/// </summary>
public static class Serialization
{
    private const int Magic = 0x534E4554; // "TENS"

    public static void Save(Module module, string path)
        => SaveTensors(module.NamedParameters(), path);

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
        foreach (var (name, p) in module.NamedParameters())
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
}
