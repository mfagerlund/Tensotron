using System.Globalization;
using System.Text;

namespace Tensotron.Examples;

/// <summary>
/// Tiny self-contained SVG scatter/line plotter for the examples — filled rectangles
/// (decision-boundary cells), circles (data points) and polylines (fitted curves), with
/// an auto-computed viewBox. SVG's y-axis points down, so we flip y on input: pass plain
/// math coordinates (y up) and the picture comes out the right way round.
///
/// Not part of the Tensotron library — just enough to make the demos produce something
/// you can open in a browser.
/// </summary>
public sealed class Plot
{
    private readonly StringBuilder _body = new();
    private float _minX = float.MaxValue, _minY = float.MaxValue;
    private float _maxX = float.MinValue, _maxY = float.MinValue;

    private static string F(float v) => v.ToString("0.###", CultureInfo.InvariantCulture);

    private void Grow(float x, float y)
    {
        _minX = MathF.Min(_minX, x); _maxX = MathF.Max(_maxX, x);
        _minY = MathF.Min(_minY, y); _maxY = MathF.Max(_maxY, y);
    }

    public void FilledCell(float x, float y, float half, string color)
    {
        float x0 = x - half, y0 = -(y + half), x1 = x + half, y1 = -(y - half);
        Grow(x0, y0); Grow(x1, y1);
        _body.AppendLine(
            $"  <rect x='{F(x0)}' y='{F(y0)}' width='{F(x1 - x0)}' height='{F(y1 - y0)}' " +
            $"fill='{color}' stroke='none' />");
    }

    public void Dot(float x, float y, float r, string color)
    {
        Grow(x - r, -y - r); Grow(x + r, -y + r);
        _body.AppendLine($"  <circle cx='{F(x)}' cy='{F(-y)}' r='{F(r)}' fill='{color}' stroke='none' />");
    }

    public void Curve(IEnumerable<(float x, float y)> points, string color, float width)
    {
        var sb = new StringBuilder("  <polyline points='");
        foreach (var (x, y) in points) { sb.Append($"{F(x)},{F(-y)} "); Grow(x, -y); }
        sb.Append($"' fill='none' stroke='{color}' stroke-width='{F(width)}' />");
        _body.AppendLine(sb.ToString());
    }

    public void Save(string fileName)
    {
        float padX = (_maxX - _minX) * 0.05f + 0.1f;
        float padY = (_maxY - _minY) * 0.05f + 0.1f;
        float x = _minX - padX, y = _minY - padY;
        float w = (_maxX - _minX) + 2 * padX, h = (_maxY - _minY) + 2 * padY;

        var svg = new StringBuilder();
        svg.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{F(x)} {F(y)} {F(w)} {F(h)}'>"
            .Replace("'", "\""));
        svg.Append(_body);
        svg.AppendLine("</svg>");
        File.WriteAllText(fileName, svg.ToString());
        Console.WriteLine($"  wrote {Path.GetFullPath(fileName)}");
    }
}
