using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

public class SerializationTests
{
    private static Sequential MakeMlp()
        => new(new Linear(4, 8), Activation.Relu(), new Linear(8, 2));

    [Fact]
    public void SaveLoad_RoundTrips()
    {
        Init.Seed(42);
        var model = MakeMlp();

        var path = Path.Combine(Path.GetTempPath(), $"tensotron_{Guid.NewGuid():N}.tns");
        try
        {
            Serialization.Save(model, path);

            // Fresh model with different random init, then load the saved weights.
            Init.Seed(999);
            var loaded = MakeMlp();
            Serialization.Load(loaded, path);

            // Every named parameter must match bit-for-bit after load.
            var a = model.NamedParameters().ToDictionary(p => p.name, p => p.param);
            var b = loaded.NamedParameters().ToDictionary(p => p.name, p => p.param);
            Assert.Equal(a.Keys.OrderBy(k => k), b.Keys.OrderBy(k => k));
            foreach (var k in a.Keys)
                Assert.Equal(a[k].ToArray(), b[k].ToArray());

            // And the models must produce identical output.
            var x = Tensor.FromShaped(Enumerable.Range(0, 12).Select(i => i * 0.1f).ToArray(), new[] { 3, 4 });
            Assert.Equal(model.Forward(x).ToArray(), loaded.Forward(x).ToArray());
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Load_ShapeMismatch_Throws()
    {
        Init.Seed(1);
        var model = MakeMlp();
        var path = Path.Combine(Path.GetTempPath(), $"tensotron_{Guid.NewGuid():N}.tns");
        try
        {
            Serialization.Save(model, path);
            var wrong = new Sequential(new Linear(4, 16), Activation.Relu(), new Linear(16, 2));
            Assert.Throws<InvalidOperationException>(() => Serialization.Load(wrong, path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
