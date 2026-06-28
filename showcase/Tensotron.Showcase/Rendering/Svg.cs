using System.Globalization;
using System.Numerics;
using System.Text;

namespace Tensotron.Showcase.Rendering;

/// <summary>
/// Minimal self-contained SVG writer for replay visualization: lines / rectangles /
/// circles over System.Numerics.Vector2, with auto-computed viewBox. No external
/// dependencies — this is showcase scaffolding, not part of the Tensotron library.
/// </summary>
public sealed class Svg
{
    private readonly List<Item> _items = new();

    private static string F(float v) => v.ToString(CultureInfo.InvariantCulture);

    public Item AddLine(Vector2 a, Vector2 b) => Add(new Item
    {
        Body = $"<line x1='{F(a.X)}' y1='{F(a.Y)}' x2='{F(b.X)}' y2='{F(b.Y)}'",
        Points = new[] { a, b },
    });

    public Item AddRect(Vector2 from, Vector2 to) => Add(new Item
    {
        Body = $"<rect x='{F(MathF.Min(from.X, to.X))}' y='{F(MathF.Min(from.Y, to.Y))}' " +
               $"width='{F(MathF.Abs(to.X - from.X))}' height='{F(MathF.Abs(to.Y - from.Y))}'",
        Points = new[] { from, to },
    });

    public Item AddCircle(Vector2 c, float r) => Add(new Item
    {
        Body = $"<circle cx='{F(c.X)}' cy='{F(c.Y)}' r='{F(r)}'",
        Points = new[] { c + new Vector2(r, r), c - new Vector2(r, r) },
    });

    private Item Add(Item item)
    {
        item.SetStroke("steelblue").SetStrokeWidth(0.5f).SetFill("none");
        _items.Add(item);
        return item;
    }

    public void Save(string fileName)
    {
        var dir = Path.GetDirectoryName(fileName);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var pts = _items.SelectMany(i => i.Points).ToList();
        var min = new Vector2(pts.Min(p => p.X), pts.Min(p => p.Y));
        var max = new Vector2(pts.Max(p => p.X), pts.Max(p => p.Y));
        var pad = (max - min) * 0.05f + Vector2.One;
        min -= pad; max += pad;
        var size = max - min;

        var sb = new StringBuilder();
        sb.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{F(min.X)} {F(min.Y)} {F(size.X)} {F(size.Y)}'>");
        foreach (var item in _items) sb.AppendLine("  " + item.Render());
        sb.AppendLine("</svg>");
        File.WriteAllText(fileName, sb.ToString().Replace("'", "\""));
    }

    public sealed class Item
    {
        public string Body = "";
        public Vector2[] Points = Array.Empty<Vector2>();
        private readonly Dictionary<string, string> _attrs = new();

        public Item SetStroke(string v) { _attrs["stroke"] = v; return this; }
        public Item SetStrokeWidth(float v) { _attrs["stroke-width"] = F(v); return this; }
        public Item SetFill(string v) { _attrs["fill"] = v; return this; }

        public string Render()
        {
            var sb = new StringBuilder(Body);
            foreach (var (k, v) in _attrs) sb.Append($" {k}='{v}'");
            sb.Append(" />");
            return sb.ToString();
        }
    }
}
