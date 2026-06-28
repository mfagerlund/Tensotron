using ILGPU.Runtime;

namespace Tensotron;

/// <summary>
/// PyTorch-flavored CUDA availability probe (mirrors <c>torch.cuda.*</c>).
///
/// Tensotron deliberately runs on a single, process-wide accelerator (see
/// <see cref="TensorRuntime"/>) — there is no per-tensor device and no <c>tensor.to(device)</c>.
/// So this is an honest availability/diagnostic surface, not a device-placement API: it answers
/// "is a CUDA GPU present, and which one?" so callers can gate GPU-only work (e.g. the showcase
/// convergence demos) instead of silently grinding on the CPU fallback.
/// </summary>
public static class Cuda
{
    private static IEnumerable<Device> CudaDevices =>
        TensorRuntime.Instance.Context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda);

    /// <summary>True if ILGPU enumerates at least one CUDA device (cf. <c>torch.cuda.is_available()</c>).</summary>
    public static bool IsAvailable() => CudaDevices.Any();

    /// <summary>Number of CUDA devices ILGPU sees (cf. <c>torch.cuda.device_count()</c>).</summary>
    public static int DeviceCount() => CudaDevices.Count();

    /// <summary>
    /// Name of the CUDA device at <paramref name="index"/> (cf. <c>torch.cuda.get_device_name()</c>).
    /// Throws if no CUDA device exists at that index — like torch, this is for when you already
    /// know a GPU is present; guard with <see cref="IsAvailable"/> otherwise.
    /// </summary>
    public static string GetDeviceName(int index = 0)
    {
        var devices = CudaDevices.ToList();
        if (index < 0 || index >= devices.Count)
            throw new ArgumentOutOfRangeException(nameof(index),
                $"CUDA device {index} requested but {devices.Count} CUDA device(s) are available.");
        return devices[index].Name;
    }
}

/// <summary>
/// Diagnostic listing of every ILGPU device of any kind (CUDA, CPU fallback, …) and which one
/// the runtime actually selected. Useful for "what is Tensotron running on?" logging; for the
/// common "do I have a GPU?" question prefer <see cref="Cuda.IsAvailable"/>.
/// </summary>
public static class Accelerators
{
    /// <summary>The accelerator type + name of every device ILGPU enumerated, in ILGPU order.</summary>
    public static IReadOnlyList<(AcceleratorType Type, string Name)> List() =>
        TensorRuntime.Instance.Context.Devices
            .Select(d => (d.AcceleratorType, d.Name))
            .ToList();

    /// <summary>The accelerator the runtime is actually using (CUDA-preferred, CPU fallback).</summary>
    public static (AcceleratorType Type, string Name) Active() =>
        (TensorRuntime.Instance.Accelerator.AcceleratorType, TensorRuntime.Instance.DeviceName);
}
