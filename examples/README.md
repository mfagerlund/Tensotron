# Tensotron examples

Minimal, **runnable** programs that train a model from scratch and write an SVG you can
open in a browser. This is the place to start if you're new to the library — each file is
a small, self-contained training loop using the same `Sequential` / `Linear` / optimizer /
loss API you'd use for real work.

```bash
# from the repo root:
dotnet run --project examples/Tensotron.Examples              # runs all three
dotnet run --project examples/Tensotron.Examples xor          # XOR — the smallest loop
dotnet run --project examples/Tensotron.Examples spiral       # spiral classification (+ spiral.svg)
dotnet run --project examples/Tensotron.Examples regression   # sine curve fit (+ regression.svg)
```

No GPU required — without CUDA, Tensotron falls back to the fast managed/SIMD CPU backend, so these
run anywhere (just slower than a GPU). SVGs are written to the current working directory.

| Example | What it shows | Output |
|---|---|---|
| **xor** (`XorExample.cs`) | The smallest complete training loop: a 1-hidden-layer MLP learning a non-linearly-separable function with `CrossEntropy` + `Adam`. Read this first. | console only |
| **spiral** (`SpiralExample.cs`) | A deeper ReLU MLP carving a curved decision boundary across 3 intertwined spirals. Classification + accuracy. | `spiral.svg` |
| **regression** (`RegressionExample.cs`) | The regression path: fitting a noisy sine with `MseLoss` (continuous targets, no softmax). | `regression.svg` |

The training loop is identical everywhere:

```csharp
var loss = TensorOps.CrossEntropy(model.Forward(x), labels); // or MseLoss(pred, target)
opt.ZeroGrad();
loss.Backward();
opt.Step();
```

`Plot.cs` is a tiny SVG writer used only to visualize results — it's not part of the
Tensotron library.
