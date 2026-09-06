---
oneliner: PyTorch-faithful float32 tensor and autograd library for .NET with CUDA and SIMD-CPU backends
tags: [tensor library, autograd, deep learning, dotnet, csharp, cuda, ilgpu, simd, reinforcement learning, gpu]
stack: [.NET 8, C#, ILGPU, CUDA]
generated: 2026-09-06
commit: e6d5ef8
placeholder: false
---
Reimplements PyTorch's op surface, autograd semantics, and gradients (down to kink and tie behavior) in native .NET, running on CUDA via ILGPU/cuBLAS or a hand-written managed/SIMD CPU backend from the same code. Every op ships with a torch-generated golden fixture proving forward/backward parity; a CUDA-graph capture path erases per-step host dispatch for small RL-scale training loops. Working library published to NuGet, exercised by spiral/regression training demos and a self-driving PPO corridor showcase.
