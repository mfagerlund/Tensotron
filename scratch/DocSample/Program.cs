using Tensotron;

var x = Tensor.FromArray(new[] { 1f, 2f, 3f }, 3).RequireGrad();
var y = (x * x).Sum();
y.Backward();
var grad = x.Grad!.ToArray();
Console.WriteLine(string.Join(",", grad));
