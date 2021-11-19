package plt

import (
// 	"fmt"
	"log"
	"image/color"
// 	"os"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
// 	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
// 	"gonum.org/v1/plot/vg/vgimg"
// 	"gonum.org/v1/plot/vg/vgsvg"
)

// https://pkg.go.dev/gonum.org/v1/plot/plotter#Scatter
func FunctionMultiPlot(x, curves *mat.Dense, title, width, height string, filename string) {
    N, Num := curves.Dims()
    p := plot.New()
    p.Title.Text = title
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
    for j:=0; j<Num; j++ {
        scatterData := make(plotter.XYs, N)
        for i := range scatterData {
            scatterData[i].X = x.At(i, 0)
            scatterData[i].Y = curves.At(i, j)
        }
        sc, err := plotter.NewLine(scatterData)
        if err != nil { log.Panic(err) }
        sc.LineStyle = draw.LineStyle{Color: color.RGBA{R: uint8(rand.Int()%256), G: uint8(rand.Int()%256), B: uint8(rand.Int()%256), A: 255}, Width: vg.Points(1)}
        p.Add(sc)
    }
	W, _ := vg.ParseLength(width)
	H, _ := vg.ParseLength(height)
	p.Save(W, H, filename)
}
