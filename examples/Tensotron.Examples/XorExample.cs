using Tensotron;

namespace Tensotron.Examples;

/// <summary>
/// The "hello world" of neural nets: learn XOR. Four points that no linear model can
/// separate, so a single hidden layer with a nonlinearity is the minimal thing that works.
/// This is the smallest complete Tensotron training loop — read it first.
/// </summary>
public static class XorExample
{
    public static void Run()
    {
        Console.WriteLine("== XOR ==");
        Init.Seed(0);

        // 4 samples, 2 features each; labels are the class index (0 or 1).
        var x = Tensor.FromArray(new[]
        {
            0f, 0f,
            0f, 1f,
            1f, 0f,
            1f, 1f,
        }, 4, 2);
        int[] y = { 0, 1, 1, 0 };

        // 2 -> 8 -> 2 MLP. Cross-entropy wants raw logits, so no final activation.
        var model = new Sequential(
            new Linear(2, 8), Activation.Tanh(),
            new Linear(8, 2));
        var opt = new Adam(model.Parameters().ToList(), lr: 5e-2f);

        for (int step = 1; step <= 2000; step++)
        {
            var loss = TensorOps.CrossEntropy(model.Forward(x), y);
            opt.ZeroGrad();
            loss.Backward();
            opt.Step();
            if (step % 500 == 0) Console.WriteLine($"  step {step,4}: loss={loss.Item():0.0000}");
        }

        // Inference: no autograd, just read the predicted class per row.
        using (Tensor.NoGradScope())
        {
            float[] logits = model.Forward(x).ToArray(); // (4, 2) row-major
            for (int i = 0; i < 4; i++)
            {
                int pred = logits[i * 2 + 1] > logits[i * 2] ? 1 : 0;
                Console.WriteLine($"  ({x.ToArray()[i * 2]:0}, {x.ToArray()[i * 2 + 1]:0}) -> {pred}  (target {y[i]})");
            }
        }
        Console.WriteLine();
    }
}
