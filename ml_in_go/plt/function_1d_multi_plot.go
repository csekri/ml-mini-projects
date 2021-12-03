package plt

import (
	"log"
	"image/color"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// https://pkg.go.dev/gonum.org/v1/plot/plotter#Scatter
/*
SUMMARY
    Plots multiple curves on the same canvas of the form f(x)=y. Each curve is assigned a random colour.
PARAMETERS
    x *mat.Dense: N number of x coordinates
    curves *mat.Dense: an N by Num matrix where Num is the number of curves, N is the number of points per curve
    title string: the name of the figure
    width string: the width of the plot e.g. "1cm"
    height string: the height of the plot e.g. "1cm"
    filename string: the name/path of the file where the figure is saved into
RETURN
    N/A
*/
func FunctionMultiPlot(x, curves *mat.Dense, title, width, height string, filename string) {
    rand.Seed(8)
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
