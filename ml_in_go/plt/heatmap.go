package plt

import (
    "image"
//     "gonum.org/v1/plot"
    "gonum.org/v1/plot/palette"
//     "gonum.org/v1/plot/vg/vgimg"
    "gonum.org/v1/plot/plotter"
// //     "gonum.org/v1/gonum/mat"
//     "gonum.org/v1/plot/vg"
//     "gonum.org/v1/plot/vg/draw"
//     "log"
//     "gonum.org/v1/gonum/stat/distuv"
//     "golang.org/x/exp/rand"
)

type Range struct {
    Min float64
    Max float64
}

type FuncHeatMap struct {
    Function func (x, y float64) float64
    Height int
    Width int
    XRange Range
    YRange Range
}

func (f *FuncHeatMap) Dims() (int, int) {
    return f.Height, f.Width
}
func (f *FuncHeatMap) X(c int) float64 {
    return f.XRange.Min + float64(c) / float64(f.Width) * (f.XRange.Max-f.XRange.Min)
}
func (f *FuncHeatMap) Y(r int) float64 {
    return f.YRange.Min + float64(r) / float64(f.Height) * ((f.YRange.Max-f.YRange.Min))
}
func (f *FuncHeatMap) Z(c, r int) float64 {
    return f.Function(f.Y(c), f.X(r))
}

func FillImage (data plotter.GridXYZ, pal palette.Palette) *image.RGBA64 {
    n, m := data.Dims()
    img := image.NewRGBA64(image.Rectangle{
        Min: image.Point{X: 0, Y: 0},
        Max: image.Point{X: n, Y: m},
    })
    colors := pal.Colors()

    max := data.Z(0, 0)
    min := data.Z(0, 0)
    for i := 0; i < n; i++ {
        for j := 0; j < m; j++ {
            if data.Z(i, j) > max {
            max = data.Z(i, j)
            }

            if data.Z(i, j) < min {
                min = data.Z(i, j)
            }
        }
    }

    for i := 0; i < n; i++ {
        for j := 0; j < m; j++ {
            v := data.Z(i, j)
            colorIdx := int((v - min) * float64(len(colors)-1) / (max - min))
            img.Set(i, n-j, colors[colorIdx])
        }
    }
    return img
}


// func main() {
//     gaussian := distuv.Normal{0, 1, rand.NewSource(69)}
//     m := FuncHeatMap{Function: func (x,y float64) float64 {return gaussian.Prob(x)*gaussian.Prob(y*2)},
//                      Height: 200,
//                      Width: 200,
//                      XRange: Range{-3, 3},
//                      YRange: Range{-3, 3},
//     }
//     pal := palette.Heat(12, 1)
//     heatmap := plotter.NewHeatMap(&m, pal)
//     contour := plotter.NewContour(&m, []float64{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09}, pal)
//
//     p := plot.New()
//     p.Title.Text = "Heat map"
//
// //     p.X.Tick.Marker = integerTicks{}
// //     p.Y.Tick.Marker = integerTicks{}
//
// //     p.Add(heatmap)
//     p.Add(contour)
//     p.Save(5*vg.Inch, 5*vg.Inch, "heatmap.pdf")
//
//     // Create a legend.
//     l := plot.NewLegend()
//     thumbs := plotter.PaletteThumbnailers(pal)
//     for i := len(thumbs) - 1; i >= 0; i-- {
//         t := thumbs[i]
//         if i != 0 && i != len(thumbs)-1 {
//             l.Add("", t)
//             continue
//         }
//         var val float64
//         switch i {
//         case 0:
//             val = heatmap.Min
//         case len(thumbs) - 1:
//             val = heatmap.Max
//         }
//         l.Add(fmt.Sprintf("%.2g", val), t)
//     }
//
//     p.X.Padding = 0
//     p.Y.Padding = 0
// //     p.X.Max = 1.5
// //     p.Y.Max = 1.5
//
//     img := vgimg.New(250, 175)
//     dc := draw.New(img)
//
//     l.Top = true
//     // Calculate the width of the legend.
//     r := l.Rectangle(dc)
//     legendWidth := r.Max.X - r.Min.X
//     l.YOffs = -p.Title.TextStyle.FontExtents().Height // Adjust the legend down a little.
//
//     l.Draw(dc)
//     dc = draw.Crop(dc, 0, -legendWidth-vg.Millimeter, 0, 0) // Make space for the legend.
//     p.Draw(dc)
//     w, err := os.Create("heatMap.png")
//     if err != nil {
//         log.Panic(err)
//     }
//     png := vgimg.PngCanvas{Canvas: img}
//     if _, err = png.WriteTo(w); err != nil {
//         log.Panic(err)
//     }
//
//
// }