package plt

import (
	"log"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette/moreland"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

/*
SUMMARY
    Makes a scatter plot where each point is labelled with colour.
    Code influenced by https://pkg.go.dev/gonum.org/v1/plot/plotter#Scatter
PARAMETERS
    xs []float64: x coordinates
    ys []float64: y coordinates
    cs []float64: colour values
    width string: width of the image e.g. "1cm"
    height string: height of the image e.g. "1cm"
    title string: the title of the figure
    filename string: the name/path of the file we save the image into
RETURN
    N/A
*/
func ScatterPlotWithLabels(xs, ys []float64, cs []int, width, height, title string, filename string) {
    scatterData := make(plotter.XYZs, len(xs))
    for i := range scatterData {
        scatterData[i].X = xs[i]
        scatterData[i].Y = ys[i]
        scatterData[i].Z = float64(1+cs[i])
	}
	// Calculate the range of Z values.
	minZ, maxZ := math.Inf(1), math.Inf(-1)
	for _, xyz := range scatterData {
		if xyz.Z > maxZ {
			maxZ = xyz.Z
		}
		if xyz.Z < minZ {
			minZ = xyz.Z
		}
	}
	colors := moreland.Kindlmann() // Initialize a color map.
	colors.SetMax(maxZ)
	colors.SetMin(minZ)

	p := plot.New()
	p.Title.Text = title
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Add(plotter.NewGrid())

	sc, err := plotter.NewScatter(scatterData)
	if err != nil {
		log.Panic(err)
	}

	// Specify style and color for individual points.
	sc.GlyphStyleFunc = func(i int) draw.GlyphStyle {
		_, _, z := scatterData.XYZ(i)
		d := (z - minZ) / (maxZ - minZ)
		rng := maxZ - minZ
		k := d*rng + minZ
		c, err := colors.At(k)
		if err != nil {
			log.Panic(err)
		}
		return draw.GlyphStyle{Color: c, Radius: vg.Points(3), Shape: draw.CircleGlyph{}}
	}
	p.Add(sc)
	W, _ := vg.ParseLength(width)
	H, _ := vg.ParseLength(height)
	p.Save(W, H, filename)
}
