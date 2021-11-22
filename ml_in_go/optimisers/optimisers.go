package main

import (
    "fmt"
    "math"
    "os"
    "image"
    "image/color"
    "image/gif"

    "gonum.org/v1/gonum/floats"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/vg/vgimg"
    "gonum.org/v1/plot/vg/draw"
    "gonum.org/v1/plot/palette"

    "ml_playground/plt"
    "ml_playground/pic"
)

func ConstantStepsizeOptimizer(input float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    StepSize := input
    Steps := 0
    if StepSize <= 0 {
        panic("No negative stepsize.")
    }
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Gradient := Derivative(At)
        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            NewAt[i] = At[i] - StepSize * Gradient[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < 1e-3 {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func ConstantStepsizeWithMomentumOptimizer(input1, input2 float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    StepSize := input1
    var Velocity []float64
    Steps := 0
    Friction := input2
    if StepSize <= 0 {
        panic("No negative stepsize.")
    }
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(Velocity) == 0 {
            Velocity = make([]float64, len(At))
        }

        Gradient := Derivative(At)
        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            Velocity[i] = Friction*Velocity[i] + StepSize*Gradient[i]
            NewAt[i] = At[i] - Velocity[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < 1e-3 {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func NesterovAcceleratedGradient(input1, input2 float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    StepSize := input1
    var Velocity []float64
    Steps := 0
    Friction := input2
    if StepSize <= 0 {
        panic("No negative stepsize.")
    }
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(Velocity) == 0 {
            Velocity = make([]float64, len(At))
        }
        ModifiedAt := make([]float64, len(At))
        for i := range ModifiedAt {
            ModifiedAt[i] = At[i] - Friction * Velocity[i]
        }
        Gradient := Derivative(ModifiedAt)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            Velocity[i] = Friction*Velocity[i] + StepSize*Gradient[i]
            NewAt[i] = At[i] - Velocity[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < 1e-3 {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}


type GifMaker struct {
    Images []*image.Paletted
    Width int
    Height int
    Delay int
    Delays []int
}

func ImageToPaletted(img image.Image) *image.Paletted {
	pm, ok := img.(*image.Paletted)
	if !ok {
		b := img.Bounds()
		pm = image.NewPaletted(b, nil)
		q := &pic.MedianCutQuantizer{NumColor: 256}
		q.Quantize(pm, b, img, image.ZP)
	}
	return pm
}

func (gm *GifMaker) CollectFrames(p *plot.Plot)  {
    img := image.NewRGBA(image.Rect(0, gm.Width, 0, gm.Height))
	// Create a new canvas using the given image to specify the dimensions
	// of the plot.
	//
	// Note that modifications applied to the canvas will not be reflected on
	// the input image.
	c := vgimg.New(vg.Points(float64(gm.Width/4*3)), vg.Points(float64(gm.Height)/4*3))

	dc := draw.New(c)
	p.Draw(dc)

	c.DrawImage(vg.Rectangle{vg.Point{vg.Points(0.0), vg.Points(0.0)}, vg.Point{vg.Points(float64(gm.Width/4*3)), vg.Points(float64(gm.Height/4*3))}}, img)

    w, err := os.Create("tmp.png")
	if err != nil {
		panic(err)
	}
	defer w.Close()
	png := vgimg.PngCanvas{Canvas: c}
	if _, err = png.WriteTo(w); err != nil {
		panic(err)
	}

	infile, err := os.Open("tmp.png")
    if err != nil {
        // replace this with real error handling
        panic(err)
    }
    defer infile.Close()

    // Decode will figure out what type of image is in the file on its own.
    // We just have to be sure all the image packages we want are imported.
    src, _, err := image.Decode(infile)
    if err != nil {
        // replace this with real error handling
        panic(err)
    }

    imgPal := ImageToPaletted(src)
    gm.Images = append(gm.Images, imgPal)
    gm.Delays = append(gm.Delays, gm.Delay)
}

func (gm *GifMaker) RenderFrames(filename string) {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()
	gif.EncodeAll(f, &gif.GIF{
		Image: gm.Images,
		Delay: gm.Delays,
	})

}

func DescentPlot(function func ([]float64) float64, Ats [][]float64, XMin, XMax, YMin, YMax float64) *plot.Plot {
    m := plt.FuncHeatMap{
        Function: func (x,y float64) float64 {
            return function([]float64{x,y})
        },
        Height: 300,
        Width: 300,
        XRange: plt.Range{XMin, XMax},
        YRange: plt.Range{YMin, YMax},
    }
    p := plot.New()
    p.Title.Text = `Optimiser Comparison`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    pal := palette.Heat(100, 1)
//     heatmap := plotter.NewHeatMap(&m, pal)
    heights := make([]float64, 50)
    for i := range heights { heights[i] = 0.02 * math.Exp(float64(i+1)/3) }
    contour := plotter.NewContour(&m, heights, pal)

    ScatterData := make(plotter.XYs, len(Ats))
    for i := range ScatterData {
        ScatterData[i].X = Ats[i][0]
        ScatterData[i].Y = Ats[i][1]
	}
    sc, err := plotter.NewScatter(ScatterData)
	if err != nil {
		panic(err)
	}
    sc.GlyphStyleFunc = func(i int) draw.GlyphStyle { return draw.GlyphStyle{
                                                        Color: color.RGBA{A:255},
                                                        Radius: 5, Shape: draw.CircleGlyph{},
                                                     }
	}
    p.Add(sc)
    p.Add(contour)
    p.Add(sc)
    return p
}

func main() {
    // the function is
    // 10(x-x^2)^2+(1-x)^2+10(y-y^2)^2+(1-y)^2
    // partial x: 40 x^3 - 60 x^2 + 22 x - 2
    // partial y: 40 y^3 - 60 y^2 + 22 y - 2
    // minimum is at (1,1)
    At1 := []float64{4,-4}
    At2 := []float64{4,-4}
    At3 := []float64{4,-4}
    F := func (x []float64) float64 {
        return 10 * math.Pow(x[0]-x[0]*x[0], 2) + math.Pow(1-x[0], 2) + 10 * math.Pow(x[1]-x[1]*x[1], 2) + math.Pow(1-x[1], 2)
    }
    fmt.Println(F)
    gradient := func (x []float64) []float64 {
        partialX := 40 * x[0]*x[0]*x[0] - 60 * x[0]*x[0] + 22 * x[0] - 2
        partialY := 40 * x[1]*x[1]*x[1] - 60 * x[1]*x[1] + 22 * x[1] - 2
        return []float64{partialX, partialY}
    }

    Finished := false
    steps := 0
    optimiser1 := ConstantStepsizeOptimizer(0.001)
    optimiser2 := ConstantStepsizeWithMomentumOptimizer(0.0001, 0.9)
    optimiser3 := NesterovAcceleratedGradient(0.0001, 0.9)
    gm := GifMaker{Width: 500, Height: 500, Delay:1}
    for i:=0; i<10000; i++ {
        fmt.Println("Steps", steps)
        if i % 4 == 0 {
            Ats := make([][]float64, 3)
            Ats[0] = At1
            Ats[1] = At2
            Ats[2] = At3
            p := DescentPlot(F, Ats, -6, 6, -6, 6)
            gm.CollectFrames(p)
        }

        At1, Finished, steps = optimiser1(gradient, At1)
        At2, Finished, steps = optimiser2(gradient, At2)
        At3, Finished, steps = optimiser3(gradient, At3)

        if Finished {
            gm.RenderFrames("gradients.gif")
            break
        }
    }
//     fmt.Println("Found local minima at", At)
    fmt.Println("Used", steps, "number of steps.")






}
