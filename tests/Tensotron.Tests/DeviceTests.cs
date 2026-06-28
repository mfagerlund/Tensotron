using ILGPU.Runtime;
using Tensotron;

namespace Tensotron.Tests;

/// <summary>
/// Behavioral tests for the PyTorch-flavored device-availability surface (<see cref="Cuda"/>,
/// <see cref="Accelerators"/>). These can't be torch-parity fixtures — GPU presence is a
/// property of the host, not of torch — so they assert internal consistency and consistency
/// with what the runtime actually selected, holding on both GPU and CPU-only machines.
/// </summary>
public class DeviceTests
{
    [Fact]
    public void IsAvailable_MatchesDeviceCount()
    {
        Assert.True(Cuda.DeviceCount() >= 0);
        Assert.Equal(Cuda.DeviceCount() > 0, Cuda.IsAvailable());
    }

    [Fact]
    public void Accelerators_AlwaysListsAtLeastTheCpuFallback()
    {
        var all = Accelerators.List();
        Assert.NotEmpty(all); // ILGPU's CPU accelerator is always present
        Assert.All(all, d => Assert.False(string.IsNullOrWhiteSpace(d.Name)));
    }

    [Fact]
    public void Active_IsOneOfTheListedDevices()
    {
        var active = Accelerators.Active();
        Assert.Contains(Accelerators.List(), d => d.Type == active.Type && d.Name == active.Name);
    }

    [Fact]
    public void ActiveCudaDevice_ImpliesCudaIsAvailable()
    {
        // The runtime prefers a GPU; if it actually selected CUDA, the probe must agree.
        if (Accelerators.Active().Type == AcceleratorType.Cuda)
            Assert.True(Cuda.IsAvailable());
    }

    [Fact]
    public void GetDeviceName_ValidWhenPresent_ThrowsOutOfRange()
    {
        if (Cuda.IsAvailable())
        {
            Assert.False(string.IsNullOrWhiteSpace(Cuda.GetDeviceName(0)));
            Assert.Throws<ArgumentOutOfRangeException>(() => Cuda.GetDeviceName(Cuda.DeviceCount()));
        }
        else
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Cuda.GetDeviceName(0));
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => Cuda.GetDeviceName(-1));
    }
}
