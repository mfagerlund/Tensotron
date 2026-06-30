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
REPLAY_RE = re.compile(
    r"RESULT replay config=(\S+).*?eager_us=([\d.]+)\s+replay_us=([\d.]+)\s+speedup=([\d.]+)")


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


def parse_replay(path):
    """[(config, eager_us, replay_us, speedup)] from `RESULT replay ...` lines (BenchExample.ReplayBench)."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = REPLAY_RE.search(line)
                if m:
                    rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    except FileNotFoundError:
        pass
    return rows


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
    # Clean RTX-4090 measurements on an unloaded GPU (median of 3 Tensotron `... -- ladder` runs / 2
    # PyTorch `tools/bench/torch_bench.py` runs). TF32 off for FP32 rows; see docs/PERFORMANCE_VS_PYTORCH.md.
    # TF32 is omitted from the training panel because it does NOT help these overhead-bound small steps
    # (torch strict ≈ default); it only matters on the compute-bound GEMM, where it's shown as the ceiling.
    train_labels = ["MLP small\n(b256)", "MLP large\n(b1024)", "CNN\n(b64)", "CNN\n(b256)"]
    tens = [0.71, 1.62, 3.56, 6.22]
    torch_fp32 = [1.32, 1.37, 1.86, 1.89]

    gemm_sizes = ["1024³", "2048³", "4096³"]
    g_tens = [41.6, 50.1, 53.2]
    g_torch_fp32 = [42.2, 50.3, 53.0]
    g_torch_tf32 = [64.5, 72.2, 80.3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1.05]})

    # --- training step: Tensotron FP32 vs PyTorch FP32 ---
    x = range(len(train_labels))
    w = 0.36
    b1 = ax1.bar([i - w / 2 for i in x], tens, w, label="Tensotron (FP32)", color=TENS)
    b2 = ax1.bar([i + w / 2 for i in x], torch_fp32, w, label="PyTorch (FP32)", color=TORCH)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(train_labels, fontsize=9)
    ax1.set_ylabel("ms / training step  (lower is better)")
    ax1.set_title("GPU training step  (fwd + bwd + Adam)", fontsize=11)
    label_bars(ax1, b1, "{:.2f}"); label_bars(ax1, b2, "{:.2f}")
    ax1.set_ylim(top=max(max(tens), max(torch_fp32)) * 1.28)
    for i in x:
        r = torch_fp32[i] / tens[i]
        txt = f"{r:.1f}× faster" if r >= 1 else f"{1 / r:.1f}× slower"
        top = max(tens[i], torch_fp32[i])
        ax1.annotate(txt, (i, top + 0.55), ha="center", fontsize=8, fontweight="bold",
                     color=(TENS if r >= 1 else TORCH))
    ax1.legend(fontsize=8.5, framealpha=0.9, loc="upper left")

    # --- FP32 GEMM: Tensotron ≈ PyTorch (both cuBLAS Sgemm); TF32 = the tensor-core ceiling we forgo ---
    gx = range(3)
    gw = 0.27
    bb = ax2.bar([i - gw for i in gx], g_tens, gw, label="Tensotron FP32", color=TENS)
    bb2 = ax2.bar(list(gx), g_torch_fp32, gw, label="PyTorch FP32", color=TORCH)
    bb3 = ax2.bar([i + gw for i in gx], g_torch_tf32, gw, label="PyTorch TF32", color="#f4a261")
    ax2.set_xticks(list(gx)); ax2.set_xticklabels(gemm_sizes)
    ax2.set_ylabel("TFLOP/s  (higher is better)")
    ax2.set_title("FP32 GEMM throughput\n(TF32 = the ceiling FP32-only forgoes)", fontsize=11)
    ax2.legend(fontsize=8, framealpha=0.9, loc="upper left")
    label_bars(ax2, bb, "{:.0f}"); label_bars(ax2, bb2, "{:.0f}"); label_bars(ax2, bb3, "{:.0f}")
    ax2.set_ylim(top=max(g_torch_tf32) * 1.22)

    fig.suptitle("On the GPU, Tensotron is in PyTorch's league: faster on small-batch MLP steps, ~par on "
                 "FP32 GEMM, ~2–3× behind on small conv  (RTX 4090)", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = f"{out_dir}/gpu_training.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


def capture_speedup(bench_dir, out_dir):
    """eager vs native-CUDA-graph replay per training-step config (from RESULT replay lines)."""
    rows = parse_replay(f"{bench_dir}/replay.txt")
    if not rows:
        print("no replay data in", f"{bench_dir}/replay.txt", "- skipping capture_speedup")
        return
    names = [r[0] for r in rows]
    eager = [r[1] for r in rows]
    replay = [r[2] for r in rows]
    speed = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = range(len(names))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], eager, w, label="eager (rebuild autograd graph each step)", color=TORCH)
    b2 = ax.bar([i + w / 2 for i in x], replay, w, label="capture + replay (one cuGraphLaunch)", color=TENS)
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("microseconds / training step  (log, lower is better)")
    ax.set_title("Step capture: eager vs native CUDA-graph replay  (fwd + bwd + clip + Adam)", fontsize=11)
    label_bars(ax, b1); label_bars(ax, b2)
    for i in x:
        top = max(eager[i], replay[i])
        ax.annotate(f"{speed[i]:.1f}× faster", (i, top * 1.45), ha="center", fontsize=9.5,
                    fontweight="bold", color=TENS)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.4)
    ax.margins(x=0.08)

    fig.suptitle("A fixed-shape training step replays as a single CUDA-graph launch — host dispatch erased",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    p = f"{out_dir}/capture_speedup.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    bench_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/img"
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    if which in ("all", "cpu"):
        cpu_inference(bench_dir, out_dir)
    if which in ("all", "gpu"):
        gpu_training(out_dir)
    if which in ("all", "capture"):
        capture_speedup(bench_dir, out_dir)
