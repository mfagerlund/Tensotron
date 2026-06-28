using System.Text.Json;
using System.Text.Json.Serialization;
using Tensotron;

namespace Tensotron.Tests;

// ---- JSON fixture model (snake_case from tools/fixtures/gen.py) ----

public sealed class Fixture
{
    [JsonPropertyName("op")] public string Op { get; set; } = "";
    [JsonPropertyName("torch_version")] public string TorchVersion { get; set; } = "";
    [JsonPropertyName("source")] public string Source { get; set; } = "";
    [JsonPropertyName("cases")] public List<Case> Cases { get; set; } = new();
}

public sealed class Case
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("meta")] public Meta? Meta { get; set; }
    [JsonPropertyName("inputs")] public List<TData> Inputs { get; set; } = new();
    [JsonPropertyName("grad_output")] public TData GradOutput { get; set; } = new();
    [JsonPropertyName("output")] public TData Output { get; set; } = new();
    [JsonPropertyName("grads")] public List<TData> Grads { get; set; } = new();
    [JsonPropertyName("running_mean")] public TData? RunningMean { get; set; }
    [JsonPropertyName("running_var")] public TData? RunningVar { get; set; }
}

public sealed class Meta
{
    [JsonPropertyName("op")] public string? Op { get; set; }
    [JsonPropertyName("dims")] public int[]? Dims { get; set; }
    [JsonPropertyName("keepdim")] public bool Keepdim { get; set; }
    [JsonPropertyName("params")] public float[]? Params { get; set; }
    [JsonPropertyName("dim")] public int Dim { get; set; }
    [JsonPropertyName("index")] public int[]? Index { get; set; }
    [JsonPropertyName("index_shape")] public int[]? IndexShape { get; set; }
    [JsonPropertyName("sizes")] public int[]? Sizes { get; set; }
    [JsonPropertyName("reduction")] public string? Reduction { get; set; }
    [JsonPropertyName("config")] public Dictionary<string, float>? Config { get; set; }
}

public sealed class TData
{
    [JsonPropertyName("shape")] public int[] Shape { get; set; } = Array.Empty<int>();
    [JsonPropertyName("data")] public float[] Data { get; set; } = Array.Empty<float>();
}

public static class Fixtures
{
    private static readonly string Dir =
        Path.Combine(AppContext.BaseDirectory, "Fixtures");

    public static Fixture Load(string op)
    {
        var path = Path.Combine(Dir, op + ".json");
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<Fixture>(json)
            ?? throw new InvalidOperationException($"Failed to load fixture {op}.");
    }

    public static Tensor ToTensor(TData td) => Tensor.FromShaped(td.Data, td.Shape);

    /// <summary>Assert a Tensotron tensor matches a torch-recorded expectation.</summary>
    public static void AssertMatches(string what, TData expected, Tensor actual,
        float atol = 2e-4f, float rtol = 2e-4f)
    {
        Assert.True(expected.Shape.SequenceEqual(actual.Shape.Dims),
            $"{what}: shape [{string.Join(",", actual.Shape.Dims)}] != expected [{string.Join(",", expected.Shape)}]");

        var got = actual.ToArray();
        Assert.Equal(expected.Data.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
        {
            float tol = atol + rtol * MathF.Abs(expected.Data[i]);
            Assert.True(MathF.Abs(got[i] - expected.Data[i]) <= tol,
                $"{what}: element {i} = {got[i]} != expected {expected.Data[i]} (tol {tol})");
        }
    }
}
