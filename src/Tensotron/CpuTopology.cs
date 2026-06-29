using System.Runtime.InteropServices;

namespace Tensotron;

/// <summary>
/// Physical-core count for sizing CPU parallelism. SMT/hyperthread siblings share a physical core's
/// vector units and L1/L2, so they barely help compute-bound SIMD GEMM (measured: <c>t16 ≈ t32</c>
/// on a 16-core/32-thread box, and <c>t32</c> is the *worst* option under load). Row-parallel matmul
/// therefore targets physical cores, not <see cref="Environment.ProcessorCount"/> (= logical).
///
/// Best-effort: Windows <c>GetLogicalProcessorInformation</c>; falls back to the logical count on
/// other platforms or any error. (That API is single-processor-group, i.e. ≤64 logical CPUs — fine
/// for desktops; the fallback covers the rest.)
/// </summary>
internal static class CpuTopology
{
    private static int? _cached;

    public static int PhysicalCoreCount()
    {
        if (_cached is int c) return c;
        int cores = 0;
        try
        {
            if (OperatingSystem.IsWindows()) cores = CountWindowsPhysicalCores();
        }
        catch { cores = 0; }
        _cached = cores > 0 ? cores : Environment.ProcessorCount;
        return _cached.Value;
    }

    private const int RelationProcessorCore = 0;

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcInfo
    {
        public UIntPtr ProcessorMask;
        public int Relationship;
        public ulong UnionPart0;   // 16-byte union (Reserved[2] is the largest member)
        public ulong UnionPart1;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformation([Out] ProcInfo[]? buffer, ref uint returnLength);

    private static int CountWindowsPhysicalCores()
    {
        uint len = 0;
        GetLogicalProcessorInformation(null, ref len);   // first call: ask for the required byte count
        if (len == 0) return 0;
        int stride = Marshal.SizeOf<ProcInfo>();
        var buf = new ProcInfo[len / stride];
        if (!GetLogicalProcessorInformation(buf, ref len)) return 0;
        int cores = 0;
        foreach (var e in buf) if (e.Relationship == RelationProcessorCore) cores++;
        return cores;
    }
}
