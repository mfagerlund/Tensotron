using ILGPU;
using ILGPU.Runtime;

namespace Tensotron;

/// <summary>
/// Backend-agnostic handle to a tensor's contiguous float32 storage. The runtime is
/// single-backend per process, so exactly one concrete kind is live at a time:
/// <see cref="DeviceStorage"/> (an ILGPU device buffer) or <see cref="HostStorage"/> (a managed
/// <c>float[]</c>). The op layer holds tensors as <see cref="TensorStorage"/> and never downcasts;
/// only the active runtime downcasts, to the concrete type it owns.
/// </summary>
public abstract class TensorStorage : IDisposable
{
    public abstract int Length { get; }

    /// <summary>Copy the whole buffer to a fresh host array. On device backends the caller must
    /// have <see cref="TensorRuntime.Sync"/>'d first (the device→host pull is the sync point).</summary>
    public abstract float[] ToHost();

    /// <summary>Overwrite the whole buffer from host data (length must equal <see cref="Length"/>).</summary>
    public abstract void CopyFromHost(float[] data);

    public abstract void Dispose();
}

/// <summary>ILGPU device-buffer storage (CUDA or ILGPU's scalar CPUAccelerator).</summary>
internal sealed class DeviceStorage : TensorStorage
{
    public MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }
    public DeviceStorage(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Buffer = buffer;

    public ArrayView<float> View => Buffer.View;
    public override int Length => (int)Buffer.Length;
    public override float[] ToHost() => Buffer.GetAsArray1D();      // caller has already Sync'd
    public override void CopyFromHost(float[] data) => Buffer.CopyFromCPU(data);
    public override void Dispose() => Buffer.Dispose();
}

/// <summary>Managed-array storage for the hand-written CPU/SIMD backend — no ILGPU in the loop.</summary>
internal sealed class HostStorage : TensorStorage
{
    public float[] Data { get; }
    public HostStorage(float[] data) => Data = data;

    public override int Length => Data.Length;
    public override float[] ToHost()
    {
        var copy = new float[Data.Length];
        Array.Copy(Data, copy, Data.Length);
        return copy;
    }
    public override void CopyFromHost(float[] data) => Array.Copy(data, Data, Data.Length);
    public override void Dispose() { }   // managed array: reclaimed by the GC
}
