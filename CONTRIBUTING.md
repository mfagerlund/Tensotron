# Contributing to Tensotron

Thanks for your interest. Tensotron is a **PyTorch-faithful, float32-only** tensor + autograd library
for .NET. The bar for changes is correctness parity with PyTorch — please read this before opening a PR.

## The law

For every op it implements, Tensotron matches PyTorch **exactly** — naming, semantics, broadcasting,
and gradients, down to behavior at kinks, ties, and special values. If an implemented op diverges from
torch, that's a bug, not a design choice. When in doubt about an op's name, signature, or edge
behavior, match torch — don't invent.

Tensotron is deliberately scoped to feed-forward training/inference (MLPs, CNNs, small RL nets). New
op families are welcome, but see the README **Scope** section for what's intentionally out of scope
today.

## Build & test

No PyTorch needed to build or test — only to regenerate fixtures (see below).

```
dotnet build Tensotron.sln --configuration Release
dotnet test  Tensotron.sln --filter "Category!=Showcase"     # parity + fast smoke (runs without a GPU)
tools/run-tests.ps1            # preferred on Windows: torch-parity + smoke at low priority
tools/run-tests.ps1 -Simd      # same suite on the managed/SIMD CPU backend
```

Tests run on whatever backend is active (CUDA if present, else the managed CPU backend), so the full
parity suite passes without a GPU. The slow `Category=Showcase` convergence demos are GPU-gated and
skip on a CPU-only box.

## Adding or changing an op (the core workflow)

**Every op ships a golden-fixture parity test, or it doesn't land** — asserting both forward *and*
backward match torch within tolerance.

1. Implement the op in the relevant `src/Tensotron/TensorOps.*.cs` partial. Compute the forward via a
   kernel, then (unless `NoGrad`) attach a named `GradNode` whose closure deposits input gradients;
   broadcast reductions in backward go through `ReduceGradToShape`.
2. Add cases to `tools/fixtures/gen.py` and run it (**the only place torch is needed**) to emit JSON
   under `tests/Tensotron.Tests/Fixtures/`:
   ```
   python tools/fixtures/gen.py
   ```
   The generator self-embeds its own source into each JSON for provenance.
3. **Probe the hard parts, not just `randn` interior:** boundary inputs that hit the kink exactly
   (e.g. `x==0` for activations, `x==bound` for clamp), parameter sweeps that break shortcuts, and
   ties/special values. Edge cases use a deterministic `grad_output` (ones) so the recorded boundary
   gradient is unambiguous.
4. Add/extend the C# test that reads the committed JSON and asserts forward + backward parity. It
   never imports torch.

The fixtures were generated with a pinned torch version recorded in each JSON's `torch_version`
field. If you regenerate, keep the diff scoped to the op you changed where possible.

## Conventions

- **float32 only.** No other dtype — it keeps the kernel surface small.
- **Single-threaded across ops.** Don't call ops from multiple threads concurrently (the one
  exception is the managed matmul's opt-in internal row parallelism, which is bit-identical to serial).
- Match the surrounding code's style; keep comments sparse and meaningful. Don't add yourself as a
  commit co-author.
- Keep build and test output clean (0 warnings).

## Pull requests

- Branch off `main`, keep PRs focused.
- Green CI (build + parity tests on Linux and Windows) is required.
- Describe what op/behavior changed and how you verified parity with torch.
