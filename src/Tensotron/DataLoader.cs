namespace Tensotron;

/// <summary>
/// Minibatch iterator over one or more tensors that share a leading (sample) axis
/// (torch.utils.data.DataLoader, simplified). Each <see cref="Epoch"/> yields batches as
/// a Tensor[] parallel to the tensors it was given; shuffling reorders via index_select,
/// sequential access via narrow. Data tensors carry no gradient, so batches are plain.
/// </summary>
public sealed class DataLoader
{
    private readonly Tensor[] _tensors;
    private readonly int _n;
    private readonly int _batchSize;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private Random _rng;

    public DataLoader(IReadOnlyList<Tensor> tensors, int batchSize,
        bool shuffle = true, bool dropLast = false, int seed = 0)
    {
        if (tensors.Count == 0) throw new InvalidOperationException("DataLoader needs at least one tensor.");
        if (batchSize < 1) throw new InvalidOperationException("batchSize must be >= 1.");
        _tensors = tensors.ToArray();
        foreach (var t in _tensors)
            if (t.Rank == 0)
                throw new InvalidOperationException("DataLoader tensors must have rank >= 1 (a leading sample dimension).");
        _n = _tensors[0].Shape.Dims[0];
        foreach (var t in _tensors)
            if (t.Shape.Dims[0] != _n)
                throw new InvalidOperationException($"DataLoader tensors disagree on sample count ({t.Shape.Dims[0]} vs {_n}).");
        _batchSize = batchSize;
        _shuffle = shuffle;
        _dropLast = dropLast;
        _rng = new Random(seed);
    }

    public DataLoader(Tensor x, Tensor y, int batchSize,
        bool shuffle = true, bool dropLast = false, int seed = 0)
        : this(new[] { x, y }, batchSize, shuffle, dropLast, seed) { }

    /// <summary>Number of batches produced per epoch.</summary>
    public int BatchCount => _dropLast ? _n / _batchSize : (_n + _batchSize - 1) / _batchSize;

    /// <summary>Iterate one epoch's batches (reshuffled each call when shuffle is on).</summary>
    public IEnumerable<Tensor[]> Epoch()
    {
        int[]? perm = null;
        if (_shuffle)
        {
            perm = new int[_n];
            for (int i = 0; i < _n; i++) perm[i] = i;
            for (int i = _n - 1; i > 0; i--) // Fisher–Yates
            {
                int j = _rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }
        }

        for (int start = 0; start < _n; start += _batchSize)
        {
            int len = Math.Min(_batchSize, _n - start);
            if (_dropLast && len < _batchSize) break;

            var batch = new Tensor[_tensors.Length];
            if (perm is null)
            {
                for (int t = 0; t < _tensors.Length; t++)
                    batch[t] = TensorOps.Narrow(_tensors[t], 0, start, len);
            }
            else
            {
                var idx = new int[len];
                Array.Copy(perm, start, idx, 0, len);
                for (int t = 0; t < _tensors.Length; t++)
                    batch[t] = TensorOps.IndexSelect(_tensors[t], 0, idx);
            }
            yield return batch;
        }
    }
}
