using System.Linq;
using Tensotron;

namespace Tensotron.Tests;

public class MovementTests
{
    [Fact]
    public void Movement_MatchTorch()
    {
        var fix = Fixtures.Load("movement");
        foreach (var c in fix.Cases)
        {
            var x = Fixtures.ToTensor(c.Inputs[0]).RequireGrad();
            var p = c.Meta!.Params;

            Tensor y = c.Meta.Op switch
            {
                "t" => TensorOps.T(x),
                "transpose" => TensorOps.Transpose(x, (int)p![0], (int)p![1]),
                "permute" => TensorOps.Permute(x, p!.Select(v => (int)v).ToArray()),
                _ => throw new InvalidOperationException(c.Meta.Op),
            };

            Fixtures.AssertMatches($"{c.Name} forward", c.Output, y);

            y.Backward(Fixtures.ToTensor(c.GradOutput));
            Fixtures.AssertMatches($"{c.Name} grad", c.Grads[0], x.Grad!);
        }
    }
}
