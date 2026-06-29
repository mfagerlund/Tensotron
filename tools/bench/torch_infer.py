"""
PyTorch side of the inference-latency comparison. Mirrors examples/Tensotron.Examples
BenchExample.InferenceLatency(): a small PPO-style policy net (8 -> 64 -> 64 -> 2, tanh),
forward-only / no-grad, the realistic deployment path (marshal obs in -> action out per call,
including readback). Reports microseconds per forward at batch 1 / 8 / 64.

PyTorch is measured at its *best* for this size, not strawmanned: single-thread (threads=1,
which is fastest for tiny nets — thread-pool wakeups cost more than the arithmetic saves) and,
separately, the out-of-box default thread count. A CUDA row is included to show the same
launch-overhead wall Tensotron's GPU backend hits at batch 1.

Prints the same `RESULT infer ...` schema as the C# bench so plot_perf.py can merge both.
Usage:  python -u tools/bench/torch_infer.py
"""
import os
import time
import numpy as np
import torch

OBS, H1, H2, ACT = 8, 64, 64, 2
BATCHES = [1, 8, 64]


def make_model():
    m = torch.nn.Sequential(
        torch.nn.Linear(OBS, H1), torch.nn.Tanh(),
        torch.nn.Linear(H1, H2), torch.nn.Tanh(),
        torch.nn.Linear(H2, ACT),
    )
    m.eval()
    return m


@torch.no_grad()
def bench(model, device, batch, iters, warm):
    xs = (np.random.rand(batch, OBS).astype(np.float32) - 0.5)
    # Realistic path: fresh host array -> tensor -> forward -> read action back to host.
    t0 = 0.0
    for i in range(iters + warm):
        if i == warm:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        xt = torch.from_numpy(xs).to(device)
        out = model(xt)
        _ = out.cpu().numpy() if device.type == "cuda" else out.numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / iters  # microseconds / forward


def run(tag, device):
    model = make_model().to(device)
    for b in BATCHES:
        iters = 20000 if b == 1 else 10000 if b == 8 else 4000
        us = bench(model, device, b, iters, iters // 10)
        print(f"RESULT infer config={tag} batch={b} us={us:.3f}", flush=True)


def main():
    print(f"# torch {torch.__version__}", flush=True)

    torch.set_num_threads(1)
    run("torch-cpu1", torch.device("cpu"))

    torch.set_num_threads(os.cpu_count() or 1)  # out-of-box default thread count
    run("torch-cpu", torch.device("cpu"))

    if torch.cuda.is_available():
        run("torch-cuda", torch.device("cuda"))


if __name__ == "__main__":
    main()
