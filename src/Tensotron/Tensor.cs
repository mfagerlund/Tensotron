namespace Tensotron;

/// <summary>
/// A backward node in the autograd graph: a named op, its inputs, and a closure
/// that, given this tensor's gradient, deposits gradients into the inputs.
/// Named so the graph prints and traces are readable.
/// </summary>
public sealed class GradNode
{
    public string OpName { get; }
    public Tensor[] Inputs { get; }
    public Action<Tensor> Backward { get; }

    public GradNode(string opName, Tensor[] inputs, Action<Tensor> backward)
    {
        OpName = opName;
        Inputs = inputs;
        Backward = backward;
    }
}

/// <summary>
/// Device-resident float32 tensor. Data lives in an ILGPU buffer; shape/strides
/// stay host-side. Autograd is define-by-run forward + explicit topological-sort
/// backward (no gradient counting). PyTorch-faithful by design.
/// </summary>
public sealed partial class Tensor : IDisposable
{
    [ThreadStatic] private static bool _noGrad;

    /// <summary>
    /// True while autograd is suspended on the current thread. Read-only: enter a no-grad
    /// region with <see cref="NoGradScope"/> (PyTorch's <c>torch.no_grad()</c>) rather than
    /// flipping a flag, so the previous state is always restored even on exceptions.
    /// </summary>
    public static bool NoGrad => _noGrad;

    /// <summary>
    /// Suspend autograd on the current thread until the returned token is disposed — the
    /// faithful equivalent of <c>with torch.no_grad():</c>. Nesting is safe: disposal restores
    /// the prior state rather than unconditionally re-enabling grad. Use with <c>using</c>.
    /// </summary>
    public static IDisposable NoGradScope()
    {
        var token = new NoGradToken(_noGrad);
        _noGrad = true;
        return token;
    }

    private sealed class NoGradToken : IDisposable
    {
        private readonly bool _prev;
        private bool _disposed;
        public NoGradToken(bool prev) => _prev = prev;
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _noGrad = _prev;
        }
    }

    private static int _nextId;
    public long Id { get; } = Interlocked.Increment(ref _nextId);

    public Shape Shape { get; private set; }
    internal TensorStorage Buffer { get; private set; }

    /// <summary>
    /// True if this tensor owns its device buffer (the common case: any freshly allocated
    /// result). False for zero-copy views (<see cref="Detach"/>, <see cref="Reshape"/>) that
    /// share another tensor's buffer — those must never free it. Disposal honors this flag.
    /// </summary>
    internal bool OwnsBuffer { get; private set; } = true;
    private bool _disposed;
    public bool IsDisposed => _disposed;

    public bool RequiresGrad { get; set; }
    public Tensor? Grad { get; set; }
    public GradNode? Node { get; internal set; }
    public string Name { get; set; } = "t";

    internal static TensorRuntime Runtime => TensorRuntime.Instance;

    internal Tensor(Shape shape, TensorStorage buffer)
    {
        Shape = shape;
        Buffer = buffer;
    }

    /// <summary>
    /// Deterministically release the device buffer. Frees it only if this tensor owns it
    /// (views over a shared buffer just drop their reference). Idempotent. Disposing a tensor
    /// whose buffer is still referenced by a live view is a caller error (same hazard as
    /// sharing storage in PyTorch); the common path — disposing freshly-allocated intermediate
    /// results — is always safe. Autograd intermediates remain reachable until backward and are
    /// otherwise reclaimed by the GC (ILGPU buffers carry finalizers); Dispose is the
    /// deterministic opt-in for inference / no-grad loops that want to bound device memory.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Only the data buffer is freed, and only if owned. Grad is intentionally NOT disposed:
        // gradients can be aliased across inputs (e.g. Add passes one upstream grad to both
        // operands), so freeing here would risk a use-after-free. Grads are GC-reclaimed.
        // The buffer is returned to the runtime's pool for reuse (it really frees over the cap).
        if (OwnsBuffer) Runtime.ReturnToPool(Buffer);
    }

    /// <summary>
    /// Recycle every interior (op-produced) buffer reachable from this tensor back to the runtime's
    /// allocator pool — the deterministic "free the graph" PyTorch does after a non-retained backward.
    /// Leaves (parameters / inputs, Node == null) and this tensor itself are left intact. Call AFTER
    /// backward + optimizer step, when the forward activations are dead; with a bounded buffer pool
    /// this turns large-activation training from cudaMalloc-bound into reuse.
    /// Caller owns the no-live-view contract, same as <see cref="Dispose"/>.
    /// </summary>
    public void DisposeGraph()
    {
        var seen = new HashSet<Tensor>();
        void Go(Tensor t)
        {
            if (!seen.Add(t)) return;
            if (t.Node == null) return;                 // leaf: stop, never free params/inputs
            foreach (var inp in t.Node.Inputs) Go(inp);
            // Never recycle a buffer the root tensor shares: when `this` is a zero-copy view
            // (Reshape/Squeeze/Flatten/Detach/Mv/1D-dot output), its owning input backs the
            // root's storage — freeing it would return live memory the caller still reads to the pool.
            if (!ReferenceEquals(t, this) && !ReferenceEquals(t.Buffer, Buffer)
                && t.OwnsBuffer && !t.IsDisposed)
                t.Dispose();                            // recycle this interior activation
        }
        Go(this);
    }

    /// <summary>True if this tensor participates in gradient flow.</summary>
    internal static bool NeedsGrad(Tensor t) => t.RequiresGrad || t.Node != null;

    // ---------------- creators ----------------

    internal static Tensor Allocate(Shape shape) => new(shape, Runtime.Allocate(shape.Size));

    public static Tensor FromArray(float[] data, params int[] dims)
    {
        var shape = dims.Length == 0 ? new Shape(data.Length) : new Shape(dims);
        if (shape.Size != data.Length)
            throw new InvalidOperationException($"Data length {data.Length} != shape size {shape.Size} {shape}.");
        var t = Allocate(shape);
        Runtime.NoteHostUpload();
        t.Buffer.CopyFromHost(data);
        return t;
    }

    public static Tensor FromArray2D(float[,] data)
    {
        int r = data.GetLength(0), c = data.GetLength(1);
        var flat = new float[r * c];
        System.Buffer.BlockCopy(data, 0, flat, 0, flat.Length * sizeof(float));
        return FromArray(flat, r, c);
    }

    /// <summary>Create from flat data with an explicit shape (supports rank-0).</summary>
    public static Tensor FromShaped(float[] data, int[] dims)
    {
        var shape = new Shape(dims);
        if (shape.Size != data.Length)
            throw new InvalidOperationException($"Data length {data.Length} != shape size {shape.Size} {shape}.");
        var t = Allocate(shape);
        Runtime.NoteHostUpload();
        t.Buffer.CopyFromHost(data);
        return t;
    }

    public static Tensor Zeros(Shape shape)
    {
        var t = Allocate(shape);
        Runtime.ZeroBuffer(t.Buffer); // device-side memset, not a host→device copy of zeros
        return t;
    }

    public static Tensor Ones(Shape shape)
    {
        var t = Allocate(shape);
        Runtime.NoteHostUpload();
        var ones = new float[shape.Size];
        Array.Fill(ones, 1f);
        t.Buffer.CopyFromHost(ones);
        return t;
    }

    /// <summary>
    /// A persistent rank-0 scalar whose value you can change between captured-graph replays. Use it as a
    /// broadcast operand inside a captured body (e.g. <c>loss = policy + value - entCoef * entropy</c>);
    /// writing a new value with <see cref="Upload"/> before the next <see cref="CapturedGraph.Replay"/> is
    /// honoured — exactly like feeding the next minibatch into a stable input tensor. This is how a
    /// per-iteration coefficient (an annealed entropy / clip-ε weight, a grad-clip threshold) stays live
    /// under capture, the same way the capturable optimizer keeps the learning rate live. A plain
    /// <c>float</c> operand, by contrast, is baked into the kernel at capture and frozen — route anything
    /// you anneal through this instead. Eagerly it is just a normal scalar tensor.
    /// </summary>
    public static Tensor ScalarInput(float value, string? name = null)
    {
        var t = FromShaped(new[] { value }, Array.Empty<int>());
        return name == null ? t : t.SetName(name);
    }

    public Tensor RequireGrad(bool value = true)
    {
        RequiresGrad = value;
        return this;
    }

    public Tensor SetName(string name)
    {
        Name = name;
        return this;
    }

    // ---------------- host transfer ----------------

    /// <summary>Pull data to host. Expensive: only sync point. Use sparingly.</summary>
    public float[] ToArray()
    {
        Runtime.Sync();
        return Buffer.ToHost();
    }

    public float Item()
    {
        if (Shape.Size != 1)
            throw new InvalidOperationException($"Item() requires a single-element tensor, got {Shape}.");
        return ToArray()[0];
    }

    /// <summary>Returns a view sharing this buffer but detached from autograd.</summary>
    public Tensor Detach() => new(Shape, Buffer) { OwnsBuffer = false };

    internal Tensor Clone()
    {
        var t = Allocate(Shape);
        Runtime.DeviceCopy(Buffer, t.Buffer); // device→device, no host round-trip
        return t;
    }

    /// <summary>Overwrite this tensor's data in place (keeps identity/buffer; used by
    /// optimizers to update parameters without rebuilding the leaf each step).</summary>
    internal void CopyInPlace(Tensor src)
    {
        if (!src.Shape.Equals(Shape))
            throw new InvalidOperationException($"CopyInPlace shape {src.Shape} != {Shape}.");
        Runtime.DeviceCopy(src.Buffer, Buffer); // device→device, no host round-trip
    }

    // CPU-sourced in-place overwrite (host data → this buffer), e.g. weight init / deserialize.
    internal void CopyFromHost(float[] data)
    {
        if (data.Length != Shape.Size)
            throw new InvalidOperationException($"CopyFromHost length {data.Length} != {Shape.Size} {Shape}.");
        Runtime.NoteHostUpload();
        Buffer.CopyFromHost(data);
    }

    /// <summary>Overwrite this tensor's contents from host data, in place, keeping the same device
    /// buffer (so a captured trace's launches stay valid). Used to feed the next minibatch into a
    /// <see cref="CapturedGraph"/>'s stable input tensor between replays.</summary>
    public Tensor Upload(float[] data)
    {
        CopyFromHost(data);
        return this;
    }

    /// <summary>In-place copy of another tensor's data into this one (torch's <c>copy_</c>).</summary>
    public Tensor Copy_(Tensor src)
    {
        CopyInPlace(src);
        return this;
    }

    // ---------------- shape ----------------

    /// <summary>Zero-copy reshape (shares the contiguous buffer). Tracked.</summary>
    public Tensor Reshape(params int[] dims)
    {
        var ns = Shape.Reshape(dims);
        var result = new Tensor(ns, Buffer) { OwnsBuffer = false };
        if (!NoGrad && NeedsGrad(this))
        {
            var self = this;
            result.Node = new GradNode("Reshape", new[] { this },
                g => self.AddGrad(g.Reshape(self.Shape.Dims)));
        }
        return result;
    }

    // ---------------- autograd ----------------

    public void AddGrad(Tensor g)
    {
        if (!g.Shape.Equals(Shape))
            throw new InvalidOperationException($"Gradient shape {g.Shape} != tensor shape {Shape} ({Name}).");
        if (Grad == null)
            Grad = g;
        else
            TensorOps.AddInto(Grad, g);
    }

    /// <summary>
    /// Reverse-mode backward via explicit topological sort. By the time we reach a
    /// node in reverse order, every downstream consumer has already accumulated into
    /// it — so accumulation is just AddGrad. No expected/received counting.
    /// </summary>
    public void Backward(Tensor? grad = null)
    {
        if (!NeedsGrad(this))
            throw new InvalidOperationException($"{Name} does not require grad — nothing to back-propagate.");

        var topo = new List<Tensor>();
        var visited = new HashSet<Tensor>();
        Build(this, visited, topo);

        // Clear grads on interior nodes (those produced by an op) so a repeated backward
        // recomputes them fresh instead of accumulating stale contributions through the
        // graph. Leaf grads (Node == null) are retained and accumulate across calls, which
        // matches torch's retain_graph semantics — call ZeroGrad between passes to reset.
        foreach (var t in topo)
            if (t.Node != null)
                t.Grad = null;

        // torch: an implicit gradient can be created only for a scalar output. A non-scalar
        // root must be given an explicit `grad`, else a vector loss is silently summed.
        if (grad == null && Shape.Size != 1)
            throw new InvalidOperationException(
                $"Backward on a non-scalar tensor {Shape} requires an explicit gradient argument " +
                "(grad can be implicitly created only for scalar outputs).");
        Grad = grad ?? Ones(Shape);

        using (NoGradScope())
        {
            for (int i = topo.Count - 1; i >= 0; i--)
            {
                var t = topo[i];
                if (t.Node != null && t.Grad != null)
                    t.Node.Backward(t.Grad);
            }
        }
    }

    private static void Build(Tensor t, HashSet<Tensor> visited, List<Tensor> topo)
    {
        if (!visited.Add(t)) return;
        if (t.Node != null)
            foreach (var inp in t.Node.Inputs)
                Build(inp, visited, topo);
        topo.Add(t); // post-order: inputs precede outputs
    }

    public void ZeroGrad()
    {
        var seen = new HashSet<Tensor>();
        void Go(Tensor t)
        {
            if (!seen.Add(t)) return;
            t.Grad = null;
            if (t.Node != null)
                foreach (var i in t.Node.Inputs) Go(i);
        }
        Go(this);
    }

    public override string ToString() => $"{Name}#{Id} {Shape}";
}
