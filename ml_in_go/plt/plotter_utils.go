package plt

import (
    "image/color"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/palette"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/vg/draw"

    "gonum.org/v1/gonum/mat"
)

func HEX2RGBA(Hex int) (uint8, uint8, uint8, uint8) {
    r := (Hex >> 24) & 255
    g := (Hex >> 16) & 255
    b := (Hex >> 8) & 255
    a := Hex & 255
    return uint8(r), uint8(g), uint8(b), uint8(a)
}

func RGBA2HEX(R, G, B, A uint32) int {
    // These uint32 values are of the form 0xSTUV where ST is the color, UV is the alpha blend.
    // Therefore with some bit operation we mask (0xff00) the alpha blend out and have only the colour, then shift to
    // reach numbers in the range 0-255.
    R_, G_, B_, A_ := int((R & 0xff00) >> 8), int((G & 0xff00) >> 8), int((B & 0xff00) >> 8), int((A & 0xff00) >> 8)
    return  (R_ << 24) | (G_ << 16) | (B_ << 8) | (A_)
}


func MakeXY(X, Y []float64) plotter.XYs {
    points := make(plotter.XYs, len(X))
    for i := range points {
        points[i].X = X[i]
        points[i].Y = Y[i]
    }
    return points
}

func MakeScatter(X, Y []float64) *plotter.Scatter {
    points := MakeXY(X, Y)
    scatter, _ := plotter.NewScatter(points)
    return scatter
}

func MakeLine(X, Y []float64) *plotter.Line {
    points := MakeXY(X, Y)
    line, _ := plotter.NewLine(points)
    return line
}

const CIRCLE_POINT_MARKER = 0
const CROSS_POINT_MARKER = 1
const PLUS_POINT_MARKER = 2
const PYRAMID_POINT_MARKER = 3
const RING_POINT_MARKER = 4
const SQUARE_POINT_MARKER = 5
const TRIANGLE_POINT_MARKER = 6

func MakeScatterUnicorn(X, Y []float64, MarkerType int, MarkerRadius float64, Palette palette.Palette) *plotter.Scatter {
    scatter := MakeScatter(X, Y)
    colours := Palette.Colors()
    var shape draw.GlyphDrawer
    switch MarkerType {
        case CIRCLE_POINT_MARKER:
            shape = draw.CircleGlyph{}
        case CROSS_POINT_MARKER:
            shape = draw.CrossGlyph{}
        case PLUS_POINT_MARKER:
            shape = draw.PlusGlyph{}
        case PYRAMID_POINT_MARKER:
            shape = draw.PyramidGlyph{}
        case RING_POINT_MARKER:
            shape = draw.RingGlyph{}
        case SQUARE_POINT_MARKER:
            shape = draw.SquareGlyph{}
        case TRIANGLE_POINT_MARKER:
            shape = draw.TriangleGlyph{}
    }
    scatter.Shape = shape
    scatter.Color = colours[0]
    scatter.Radius = vg.Points(MarkerRadius)
    scatter.GlyphStyleFunc = func(i int) draw.GlyphStyle {
        return draw.GlyphStyle{
            Color: colours[i],
            Radius: vg.Points(MarkerRadius),
            Shape: shape,
        }
	}
    return scatter
}

func MakeLineUnicorn(X, Y []float64, LineWidth float64, HexColour int, Dashes []float64) *plotter.Line {
    line := MakeLine(X, Y)
    r, g, b, a := HEX2RGBA(HexColour)
    dashes := make([]vg.Length, len(Dashes))
    for i := range dashes {
        dashes[i] = vg.Points(Dashes[i])
    }
    line.LineStyle = draw.LineStyle{Color: color.RGBA{r, g, b, a}, Width: vg.Points(LineWidth), Dashes: dashes}
    return line
}


func MakeMultiLineUnicorn(X []float64, Y *mat.Dense, LineWidth float64, Palette palette.Palette, Dashes []float64) []*plotter.Line {
    _, Num := Y.Dims()
    colours := Palette.Colors()
    lines := make([]*plotter.Line, Num)
    for j:=0; j<Num; j++ {
        r, g, b, a := colours[j].RGBA()
        lines[j] = MakeLineUnicorn(X, mat.Col(nil, j, Y), LineWidth, RGBA2HEX(r, g, b, a), Dashes)
    }
    return lines
}







