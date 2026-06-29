"""
PyTorch side of the Tensotron-vs-PyTorch ladder. Mirrors examples/Tensotron.Examples
BenchExample.Ladder() exactly — same architectures, batch sizes, optimizer (Adam 1e-3),
step counts, warmup, and timing protocol (single cuda.synchronize() around the timed region) —
and prints the SAME `RESULT …` line schema so the two outputs merge into one table.

Three rungs:
  1. MLP hot path        — overhead-bound; Tensotron's recent work targets this regime.
  2. MNIST-CNN           — the showcase arch; conv via cuDNN, the hardest matchup for us.
  3. large GEMM forward  — compute-bound; PyTorch matmul is FP32 by default (TF32 off).

Two PyTorch configs (the ONLY difference is TF32, with cudnn.benchmark=True for both so
conv-algo selection is fair and TF32 is isolated):
  strict  — TF32 off everywhere   (same numerical class as Tensotron: FP32, no tensor cores)
  default — torch defaults        (matmul TF32 off, cuDNN conv TF32 on → tensor cores on conv)

GEMM additionally reports a `tf32` row (matmul TF32 on) to show what tensor cores buy — the
ceiling we forgo by being FP32-only by design.

Usage:  python -u tools/bench/torch_bench.py
"""
import time
import torch

assert torch.cuda.is_available(), "need CUDA"
DEV = torch.device("cuda")


def sync():
    torch.cuda.synchronize()


def set_config(name: str):
    # benchmark=True for both so cuDNN autotunes conv either way; TF32 is the only variable.
    torch.backends.cudnn.benchmark = True
    if name == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif name == "default":
        torch.backends.cuda.matmul.allow_tf32 = False  # torch default
        torch.backends.cudnn.allow_tf32 = True          # torch default (conv tensor cores)
    else:
        raise ValueError(name)


def bench_mlp(batch, in_dim, width, depth, steps=300, warmup=30):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]
        d = width
    layers += [torch.nn.Linear(d, 1)]
    model = torch.nn.Sequential(*layers).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = torch.nn.MSELoss()
    x = torch.rand(batch, in_dim, device=DEV) - 0.5
    target = torch.zeros(batch, 1, device=DEV)
    t0 = 0.0
    for step in range(steps):
        if step == warmup:
            sync(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = lossfn(model(x), target)
        loss.backward()
        opt.step()
    sync(); t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / (steps - warmup)


def bench_cnn(batch, steps=150, warmup=20):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 7 * 7, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = torch.nn.CrossEntropyLoss()
    x = torch.rand(batch, 1, 28, 28, device=DEV)
    y = torch.randint(0, 10, (batch,), device=DEV)
    t0 = 0.0
    for step in range(steps):
        if step == warmup:
            sync(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = lossfn(model(x), y)
        loss.backward()
        opt.step()
    sync(); t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / (steps - warmup)


def bench_gemm(m, n, k, iters=50, warmup=10):
    a = torch.rand(m, k, device=DEV) - 0.5
    b = torch.rand(k, n, device=DEV) - 0.5
    t0 = 0.0
    for it in range(iters + warmup):
        if it == warmup:
            sync(); t0 = time.perf_counter()
        c = a @ b
    sync(); t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / iters
    gflops = 2.0 * m * n * k / (ms / 1000.0) / 1e9
    return ms, gflops


def main():
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    mlp_cfgs = [(256, 32, 128, 2), (1024, 128, 512, 2)]
    cnn_cfgs = [64, 256]
    gemm_cfgs = [1024, 2048, 4096]

    for cfg in ("strict", "default"):
        set_config(cfg)
        for (b, ind, w, d) in mlp_cfgs:
            ms = bench_mlp(b, ind, w, d)
            print(f"RESULT mlp config={cfg} batch={b} in={ind} width={w} depth={d} ms_per_step={ms:.3f}")
        for b in cnn_cfgs:
            ms = bench_cnn(b)
            print(f"RESULT cnn config={cfg} batch={b} ms_per_step={ms:.3f}")

    # GEMM: matmul TF32 is off in both strict & default, so report FP32 once, plus a TF32 ceiling.
    torch.backends.cuda.matmul.allow_tf32 = False
    for n in gemm_cfgs:
        ms, gf = bench_gemm(n, n, n)
        print(f"RESULT gemm config=fp32 m={n} n={n} k={n} ms={ms:.3f} gflops={gf:.1f}")
    torch.backends.cuda.matmul.allow_tf32 = True
    for n in gemm_cfgs:
        ms, gf = bench_gemm(n, n, n)
        print(f"RESULT gemm config=tf32 m={n} n={n} k={n} ms={ms:.3f} gflops={gf:.1f}")


if __name__ == "__main__":
    main()
