namespace Tensotron;

/// <summary>
/// Host-side shape: logical dimensions plus row-major (contiguous) strides.
/// Backend-agnostic. Tensotron tensors are always stored contiguous row-major on
/// the device, so strides are derived, not stored as views.
/// </summary>
public sealed class Shape : IEquatable<Shape>
{
    public int[] Dims { get; }
    public int[] Strides { get; }
    public int Size { get; }
    public int Rank => Dims.Length;

    public Shape(params int[] dims)
    {
        for (int i = 0; i < dims.Length; i++)
            if (dims[i] < 0)
                throw new ArgumentException($"Shape dimensions must be non-negative; got [{string.Join(",", dims)}].", nameof(dims));
        Dims = dims;
        Strides = new int[dims.Length];
        int stride = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            Strides[i] = stride;
            stride *= dims[i];
        }
        Size = stride;
    }

    public Shape Reshape(int[] dims)
    {
        // Resolve a single -1 inferred dimension (PyTorch semantics).
        int inferred = Array.IndexOf(dims, -1);
        var resolved = (int[])dims.Clone();
        // Validate: at most one -1, and no other negative dimensions.
        int negCount = 0;
        foreach (var d in dims) if (d < 0) negCount++;
        if (negCount > 1)
            throw new InvalidOperationException($"Cannot reshape {this} to ({string.Join(",", dims)}): only one dimension can be inferred (-1).");
        for (int i = 0; i < dims.Length; i++)
            if (dims[i] < 0 && dims[i] != -1)
                throw new InvalidOperationException($"Cannot reshape {this} to ({string.Join(",", dims)}): invalid negative dimension {dims[i]}.");
        if (inferred >= 0)
        {
            int known = 1;
            for (int i = 0; i < dims.Length; i++)
                if (i != inferred) known *= dims[i];
            if (known == 0 || Size % known != 0)
                throw new InvalidOperationException($"Cannot reshape {this} to ({string.Join(",", dims)}): inferred dimension is not integral.");
            resolved[inferred] = Size / known;
        }

        var s = new Shape(resolved);
        if (s.Size != Size)
            throw new InvalidOperationException($"Cannot reshape {this} to ({string.Join(",", resolved)}): size mismatch.");
        return s;
    }

    public bool Equals(Shape? other)
    {
        if (other is null || other.Rank != Rank) return false;
        for (int i = 0; i < Rank; i++)
            if (Dims[i] != other.Dims[i]) return false;
        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as Shape);

    public override int GetHashCode()
    {
        var h = new HashCode();
        foreach (var d in Dims) h.Add(d);
        return h.ToHashCode();
    }

    public override string ToString() => $"[{string.Join(",", Dims)}]";
}
