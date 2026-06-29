using Tensotron;
using Tensotron.Examples;

// Minimal launcher: pick a demo by name, or run them all.
//   dotnet run --project examples/Tensotron.Examples -- spiral
var which = (args.Length > 0 ? args[0] : "all").ToLowerInvariant();

Console.WriteLine($"Tensotron examples — device: {Accelerators.Active().Name}\n");

switch (which)
{
    case "xor": XorExample.Run(); break;
    case "spiral": SpiralExample.Run(); break;
    case "regression": case "reg": RegressionExample.Run(); break;
    case "bench": BenchExample.Run(); break;
    case "benchsweep": case "sweep": BenchExample.Sweep(); break;
    case "benchpool": case "pool": BenchExample.Pool(); break;
    case "benchladder": case "ladder": BenchExample.Ladder(); break;
    case "convbreakdown": case "convbd": BenchExample.ConvBreakdown(); break;
    case "cnnphases": BenchExample.CnnPhases(); break;
    case "gemmtf32": case "tf32": BenchExample.GemmTf32(); break;
    case "inference": case "infer": BenchExample.InferenceLatency(); break;
    case "stepbreakdown": case "breakdown": BenchExample.StepBreakdown(); break;
    case "replaybench": case "replay": BenchExample.ReplayBench(); break;
    case "all":
        XorExample.Run();
        SpiralExample.Run();
        RegressionExample.Run();
        break;
    default:
        Console.WriteLine($"Unknown example '{which}'. Try: xor | spiral | regression | bench | benchsweep | benchpool | ladder | all");
        Environment.ExitCode = 1;
        break;
}
