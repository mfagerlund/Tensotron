"""
Render the README performance figures as PNGs from benchmark output.

  cpu_inference.png  — the headline: small control-net forward latency, Tensotron's
                       managed/SIMD CPU backend vs PyTorch CPU, plus a batch-1 CPU-vs-GPU
                       panel. Built from FRESH measurements parsed out of the RESULT lines
                       emitted by `BenchExample.InferenceLatency` (run under TENSOTRON_BACKEND
                       =simd and =cuda) and `tools/bench/torch_infer.py`.

  gpu_training.png   — secondary: GPU training ms/step and FP32 GEMM throughput vs PyTorch.
                       Uses the documented clean-RTX-4090 numbers from
                       docs/PERFORMANCE_VS_PYTORCH.md (reproduce on an *unloaded* GPU via
                       `... -- ladder` and `tools/bench/torch_bench.py`).

Usage:  python tools/bench/plot_perf.py <bench_dir> <out_dir>
"""
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TENS = "#2a9d8f"   # Tensotron
TORCH = "#e76f51"  # PyTorch
TENS_GPU = "#264653"
TORCH_GPU = "#b5651d"
BATCHES = [1, 8, 64]

RESULT_RE = re.compile(r"config=(\S+)\s+batch=(\d+)\s+us=([\d.]+)")


def parse(path):
    """{(config, batch): us} from a file's `RESULT infer ...` lines."""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "RESULT infer" not in line:
                    continue
                m = RESULT_RE.search(line)
                if m:
                    out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    except FileNotFoundError:
        pass
    return out


def label_bars(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 1),
                    textcoords="offset points")


def cpu_inference(bench_dir, out_dir):
    simd = parse(f"{bench_dir}/infer_simd.txt")   # tensotron CPU
    cuda = parse(f"{bench_dir}/infer_cuda.txt")   # tensotron GPU
    torch = parse(f"{bench_dir}/infer_torch.txt")

    tens_cpu = [simd[("tensotron", b)] for b in BATCHES]
    torch_cpu = [torch[("torch-cpu1", b)] for b in BATCHES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})

    # --- Panel 1: CPU inference latency by batch ---
    x = range(len(BATCHES))
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], tens_cpu, w, label="Tensotron (SIMD CPU)", color=TENS)
    b2 = ax1.bar([i + w / 2 for i in x], torch_cpu, w, label="PyTorch (CPU, 1 thread)", color=TORCH)
    ax1.set_yscale("log")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"batch {b}" for b in BATCHES])
    ax1.set_ylabel("microseconds / forward  (log, lower is better)")
    ax1.set_title("CPU inference: small policy net  8→64→64→2 (tanh)", fontsize=11)
    label_bars(ax1, b1); label_bars(ax1, b2)
    # speedup annotations centered over each group
    for i, b in enumerate(BATCHES):
        r = torch_cpu[i] / tens_cpu[i]
        txt = f"{r:.1f}× faster" if r >= 1 else f"{1/r:.1f}× slower"
        top = max(tens_cpu[i], torch_cpu[i])
        ax1.annotate(txt, (i, top * 1.6), ha="center", fontsize=8.5, fontweight="bold",
                     color=(TENS if r >= 1 else TORCH))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_ylim(top=ax1.get_ylim()[1] * 2.2)
    ax1.margins(x=0.08)

    # --- Panel 2: batch-1 latency, CPU vs GPU (both libraries) ---
    names = ["Tensotron\nSIMD CPU", "PyTorch\nCPU", "Tensotron\nCUDA", "PyTorch\nCUDA"]
    vals = [simd[("tensotron", 1)], torch[("torch-cpu1", 1)],
            cuda[("tensotron", 1)], torch[("torch-cuda", 1)]]
    colors = [TENS, TORCH, TENS_GPU, TORCH_GPU]
    bb = ax2.bar(names, vals, color=colors, width=0.7)
    ax2.set_yscale("log")
    ax2.set_ylabel("µs / forward  (log)")
    ax2.set_title("Batch-1 latency: CPU ≪ GPU\n(launch overhead dominates a tiny net)", fontsize=11)
    label_bars(ax2, bb)
    ax2.set_ylim(top=max(vals) * 2.4)
    ax2.tick_params(axis="x", labelsize=8.5)

    fig.suptitle("A single-agent control-net forward is ~8× faster on Tensotron's CPU backend than PyTorch CPU—"
                 "and the GPU is the wrong tool for it",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = f"{out_dir}/cpu_inference.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


def gpu_training(out_dir):
    # Documented clean-RTX-4090 numbers (docs/PERFORMANCE_VS_PYTORCH.md). FP32 strict baseline.
    train_labels = ["MLP small\n(b256)", "MLP large\n(b1024)", "CNN\n(b64)", "CNN\n(b256)"]
    tens = [1.22, 3.34, 4.79, 5.04]
    torch_fp32 = [2.04, 2.04, 2.84, 2.66]
    torch_tf32 = [1.47, 1.40, 1.92, 1.96]

    gemm_sizes = ["1024³", "2048³", "4096³"]
    g_tens = [41.7, 48.7, 48.7]
    g_torch = [34.8, 42.5, 45.7]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})

    x = range(len(train_labels))
    w = 0.27
    ax1.bar([i - w for i in x], tens, w, label="Tensotron (FP32)", color=TENS)
    ax1.bar(list(x), torch_fp32, w, label="PyTorch (FP32)", color=TORCH)
    ax1.bar([i + w for i in x], torch_tf32, w, label="PyTorch (TF32)", color="#f4a261")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(train_labels, fontsize=9)
    ax1.set_ylabel("ms / training step  (lower is better)")
    ax1.set_title("GPU training step  (fwd + bwd + Adam)", fontsize=11)
    ax1.legend(fontsize=8.5, framealpha=0.9)

    bb = ax2.bar([i - 0.2 for i in range(3)], g_tens, 0.4, label="Tensotron FP32", color=TENS)
    bb2 = ax2.bar([i + 0.2 for i in range(3)], g_torch, 0.4, label="PyTorch FP32", color=TORCH)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(gemm_sizes)
    ax2.set_ylabel("TFLOP/s  (higher is better)")
    ax2.set_title("FP32 GEMM throughput\n(both call cuBLAS Sgemm)", fontsize=11)
    ax2.legend(fontsize=8.5, framealpha=0.9)
    label_bars(ax2, bb, "{:.0f}"); label_bars(ax2, bb2, "{:.0f}")

    fig.suptitle("On the GPU, Tensotron is in PyTorch's league: even on FP32 GEMM, ahead on small MLPs, "
                 "~1.5–2× behind on conv  (RTX 4090)", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = f"{out_dir}/gpu_training.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    bench_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/img"
    cpu_inference(bench_dir, out_dir)
    gpu_training(out_dir)
