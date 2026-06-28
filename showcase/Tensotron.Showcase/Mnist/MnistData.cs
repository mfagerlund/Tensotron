using System.IO.Compression;
using System.Net.Http;

namespace Tensotron.Showcase.Mnist;

/// <summary>
/// Downloads and caches the MNIST dataset (idx.gz from the CVDF mirror) and parses it into
/// flat float arrays. Images are normalized to [0,1]. Cached under the local app-data dir so
/// the download happens only once; <see cref="TryLoad"/> returns null if the data can't be
/// obtained (e.g. offline) so the showcase test can skip gracefully.
/// </summary>
public sealed class MnistData
{
    public float[] TrainImages = Array.Empty<float>(); // N*784, row-major, [0,1]
    public int[] TrainLabels = Array.Empty<int>();
    public float[] TestImages = Array.Empty<float>();
    public int[] TestLabels = Array.Empty<int>();
    public int TrainCount, TestCount;
    public const int ImageSize = 28 * 28;

    private const string Mirror = "https://storage.googleapis.com/cvdf-datasets/mnist/";

    private static readonly (string file, string kind)[] Files =
    {
        ("train-images-idx3-ubyte.gz", "train-img"),
        ("train-labels-idx1-ubyte.gz", "train-lbl"),
        ("t10k-images-idx3-ubyte.gz",  "test-img"),
        ("t10k-labels-idx1-ubyte.gz",  "test-lbl"),
    };

    public static string CacheDir =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Tensotron", "mnist");

    /// <summary>Load MNIST, downloading+caching on first use. Returns null on failure.</summary>
    public static MnistData? TryLoad(Action<string>? log = null)
    {
        try
        {
            Directory.CreateDirectory(CacheDir);
            foreach (var (file, _) in Files)
            {
                var path = Path.Combine(CacheDir, file);
                if (!File.Exists(path))
                {
                    log?.Invoke($"downloading {file} ...");
                    using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
                    var bytes = http.GetByteArrayAsync(Mirror + file).GetAwaiter().GetResult();
                    File.WriteAllBytes(path, bytes);
                }
            }

            var data = new MnistData();
            (data.TrainImages, data.TrainCount) = ReadImages(Path.Combine(CacheDir, "train-images-idx3-ubyte.gz"));
            data.TrainLabels = ReadLabels(Path.Combine(CacheDir, "train-labels-idx1-ubyte.gz"));
            (data.TestImages, data.TestCount) = ReadImages(Path.Combine(CacheDir, "t10k-images-idx3-ubyte.gz"));
            data.TestLabels = ReadLabels(Path.Combine(CacheDir, "t10k-labels-idx1-ubyte.gz"));
            return data;
        }
        catch (Exception ex)
        {
            log?.Invoke($"MNIST load failed: {ex.Message}");
            return null;
        }
    }

    private static byte[] Gunzip(string path)
    {
        using var fs = File.OpenRead(path);
        using var gz = new GZipStream(fs, CompressionMode.Decompress);
        using var ms = new MemoryStream();
        gz.CopyTo(ms);
        return ms.ToArray();
    }

    private static int BigEndian(byte[] b, int o) => (b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3];

    private static (float[] images, int count) ReadImages(string path)
    {
        var raw = Gunzip(path);
        if (BigEndian(raw, 0) != 2051) throw new InvalidDataException($"Bad idx image magic in {path}.");
        int count = BigEndian(raw, 4);
        int rows = BigEndian(raw, 8), cols = BigEndian(raw, 12);
        int n = rows * cols;
        var images = new float[count * n];
        int offset = 16;
        for (int i = 0; i < images.Length; i++) images[i] = raw[offset + i] / 255f;
        return (images, count);
    }

    private static int[] ReadLabels(string path)
    {
        var raw = Gunzip(path);
        if (BigEndian(raw, 0) != 2049) throw new InvalidDataException($"Bad idx label magic in {path}.");
        int count = BigEndian(raw, 4);
        var labels = new int[count];
        for (int i = 0; i < count; i++) labels[i] = raw[8 + i];
        return labels;
    }
}
