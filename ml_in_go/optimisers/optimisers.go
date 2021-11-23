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
    "gonum.org/v1/plot/vg/draw"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"


    "ml_playground/plt"
    "ml_playground/pic"
)

type BlackPalette string
func (p BlackPalette) Colors() []color.Color {
    return []color.Color{color.RGBA{A:70}}
}

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

func BacktrackingLineSearch(input float64) func(func ([]float64) float64, func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Beta := input
    Steps := 0
    if Beta <= 0 || Beta >= 1 {
        panic("Beta is outsize of range (0,1)")
    }
    Terminate := false
    return func(F func ([]float64) float64, Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Gradient := Derivative(At)
        SearchAt := make([]float64, len(At))

        LeftHandSide := 1.0
        RightHandSide := 0.0
        NormSquare := math.Pow(floats.Norm(Gradient, 2), 2)
        t := 1.0
        for ; LeftHandSide > RightHandSide; {
            for i := range At {
                SearchAt[i] = At[i] - t * Gradient[i]
            }
            LeftHandSide = F(SearchAt)
            RightHandSide = F(At) - t / 2.0 * NormSquare
            t *= Beta
        }
        Steps++
        if floats.Distance(At, SearchAt, 2) < 1e-10 {
            Terminate = true
        }
        return SearchAt, Terminate, Steps
    }
}

func Adagrad(input1, input2 float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    StepSize := input1
    Epsilon := input2
    var G []float64
    Steps := 0
    if StepSize <= 0 {
        panic("No negative stepsize.")
    }
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(G) == 0 {
            G = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            G[i] = G[i] + Gradient[i] * Gradient[i]
            NewAt[i] = At[i] - StepSize / (math.Sqrt(G[i] + Epsilon)) * Gradient[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < 1e-3 {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

// coded following the original paper
// https://arxiv.org/pdf/1212.5701.pdf
// at page 3 Algorithm 1
func AdaDelta(input1, input2 float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Decay := input1
    Epsilon := input2
    var G []float64
    var Delta []float64
    Steps := 0
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(G) == 0 {
            G = make([]float64, len(At))
            Delta = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            G[i] = Decay * G[i] + (1-Decay) * Gradient[i] * Gradient[i]
            Update := - math.Sqrt((Delta[i]+Epsilon) / (G[i]+Epsilon)) * Gradient[i] // one sqrt is enough
            Delta[i] = Decay * Delta[i] + (1-Decay) * Update * Update
            NewAt[i] = At[i] + Update
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < 1e-3 {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func RMSprop(input1, input2, input3 float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    StepSize := input1
    Epsilon := input2
    Decay := input3
    var G []float64
    Steps := 0
    if StepSize <= 0 {
        panic("No negative stepsize.")
    }
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(G) == 0 {
            G = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            G[i] = Decay * G[i] + (1-Decay) * Gradient[i] * Gradient[i]
            NewAt[i] = At[i] - StepSize / (math.Sqrt(G[i] + Epsilon)) * Gradient[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < 1e-3 {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

// implemented following the original paper
// https://arxiv.org/pdf/1412.6980.pdf
// at page 2
func Adam(input1, input2, input3, input4 float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    StepSize := input1
    Beta1 := input2
    Beta2 := input3
    Epsilon := input4
    var M, MHat []float64
    var V, VHat []float64
    Steps := 0
    if StepSize <= 0 {
        panic("No negative stepsize.")
    }
    Terminate := false
    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Steps++
        if len(M) == 0 {
            M = make([]float64, len(At))
            MHat = make([]float64, len(At))
            V = make([]float64, len(At))
            VHat = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            M[i] = Beta1 * M[i] + (1-Beta1) * Gradient[i]
            V[i] = Beta2 * V[i] + (1-Beta2) * Gradient[i] * Gradient[i]
            MHat[i] = M[i] / (1 - math.Pow(Beta1, float64(Steps)))
            VHat[i] = V[i] / (1 - math.Pow(Beta2, float64(Steps)))
            NewAt[i] = At[i] - StepSize * MHat[i] / (math.Sqrt(V[i]) + Epsilon)
        }
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
    p.Save(vg.Points(float64(gm.Width/4*3)), vg.Points(float64(gm.Height/4*3)), "tmp.png")
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

func DescentPlot(function func ([]float64) float64, Ats [][][]float64, XMin, XMax, YMin, YMax float64) *plot.Plot {
    myPalette := []color.Color{color.RGBA{200, 0, 0, 255},
                             color.RGBA{0, 200, 0, 255},
                             color.RGBA{0, 0, 200, 255},
                             color.RGBA{170, 170, 0, 255},
                             color.RGBA{200, 0, 200, 255},
                             color.RGBA{0, 200, 200, 255},
                             color.RGBA{0, 0, 0, 255},
                             color.RGBA{170, 170, 170, 255},
                              }
    m := plt.FuncHeatMap{
        Function: func (x,y float64) float64 {
            return function([]float64{x,y})
        },
        Height: 300,
        Width: 300,
        XRange: plt.Range{XMin, XMax},
        YRange: plt.Range{YMin, YMax},
    }
    fonts := font.NewCache(liberation.Collection())
	plot.DefaultTextHandler = text.Latex{
		Fonts: fonts,
	}
    p := plot.New()
    p.Title.Text = `Race of Optimisers`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`

    var pal BlackPalette = ""
//     heatmap := plotter.NewHeatMap(&m, pal)
    heights := make([]float64, 50)
    for i := range heights { heights[i] = 0.01 * math.Exp(float64(i+1)/4) }
    contour := plotter.NewContour(&m, heights, pal)

    ScatterData := make(plotter.XYs, len(Ats))
    for trajectory := range ScatterData {
        LastIndex := len(Ats[trajectory]) - 1
        ScatterData[trajectory].X = Ats[trajectory][LastIndex][0]
        ScatterData[trajectory].Y = Ats[trajectory][LastIndex][1]
	}
    sc, err := plotter.NewScatter(ScatterData)
	if err != nil {
		panic(err)
	}
    sc.GlyphStyleFunc = func(i int) draw.GlyphStyle { return draw.GlyphStyle{
                                                        Color: myPalette[i],
                                                        Radius: 5, Shape: draw.CircleGlyph{},
                                                     }
	}

    p.Add(contour)
	for trajectory := range Ats {
	    LineData := make(plotter.XYs, len(Ats[trajectory]))
        for i := range Ats[trajectory] {
            LineData[i].X = Ats[trajectory][i][0]
            LineData[i].Y = Ats[trajectory][i][1]
        }
        l, err := plotter.NewLine(LineData)
        if err != nil { panic(err) }
        l.LineStyle.Width = vg.Points(3)
        l.LineStyle.Color = myPalette[trajectory]
        p.Add(l)
        switch trajectory {
            case 0:
                p.Legend.Add("SGD", l)
            case 1:
                p.Legend.Add("SGD with momentum", l)
            case 2:
                p.Legend.Add("Nesterov", l)
            case 3:
                p.Legend.Add("GD with backtracting line search", l)
            case 4:
                p.Legend.Add("Adagrad", l)
            case 5:
                p.Legend.Add("Adadelta", l)
            case 6:
                p.Legend.Add("RMSprop", l)
            case 7:
                p.Legend.Add("Adam", l)
        }
	}
    p.Add(sc)

    return p
}

func main() {
    // the function is
    // 10(x-x^2)^2+(1-x)^2+10(y-y^2)^2+(1-y)^2
    // partial x: 40 x^3 - 60 x^2 + 22 x - 2
    // partial y: 40 y^3 - 60 y^2 + 22 y - 2
    // minimum is at (1,1)
    At1 := []float64{0.9,-0.3}
    At2 := []float64{0.9,-0.3}
    At3 := []float64{0.9,-0.3}
    At4 := []float64{0.9,-0.3}
    At5 := []float64{0.9,-0.3}
    At6 := []float64{0.9,-0.3}
    At7 := []float64{0.9,-0.3}
    At8 := []float64{0.9,-0.3}
    F := func (x []float64) float64 {
        return 1.0/10.0*math.Pow(1-x[0], 2) + math.Pow(x[1]-x[0]*x[0], 2)
    }
    fmt.Println(F)
    gradient := func (x []float64) []float64 {
        partialX := 1.0 / 5.0 * (20 * x[0]*x[0]*x[0] - 20 * x[0]*x[1] + x[0] - 1)
        partialY := 2 * (x[1] - x[0]*x[0])
        return []float64{partialX, partialY}
    }

    Finished1 := false
    Finished2 := false
    Finished3 := false
    Finished4 := false
    Finished5 := false
    Finished6 := false
    Finished7 := false
    Finished8 := false
    steps := 0
    optimiser1 := ConstantStepsizeOptimizer(0.1)
    optimiser2 := ConstantStepsizeWithMomentumOptimizer(0.004, 0.97)
    optimiser3 := NesterovAcceleratedGradient(0.004, 0.97)
    optimiser4 := BacktrackingLineSearch(0.999)
    optimiser5 := Adagrad(0.3, 1e-8)
    optimiser6 := AdaDelta(0.95, 1e-6)
    optimiser7 := RMSprop(0.01, 1e-8, 0.9)
    optimiser8 := Adam(0.005, 0.9, 0.999, 1e-8)
    gm := GifMaker{Width: 600, Height: 600, Delay:1}
    Ats := make([][][]float64, 8)
    for i:=0; i<10000; i++ {
        fmt.Println("Steps", steps)
        Ats[0] = append(Ats[0], At1)
        Ats[1] = append(Ats[1], At2)
        Ats[2] = append(Ats[2], At3)
        Ats[3] = append(Ats[3], At4)
        Ats[4] = append(Ats[4], At5)
        Ats[5] = append(Ats[5], At6)
        Ats[6] = append(Ats[6], At7)
        Ats[7] = append(Ats[7], At8)
        if i % 2 == 0 {
            p := DescentPlot(F, Ats, -0.9, 1.2, -0.8, 1.4)
            gm.CollectFrames(p)
        }

        At1, Finished1, steps = optimiser1(gradient, At1)
        At2, Finished2, steps = optimiser2(gradient, At2)
        At3, Finished3, steps = optimiser3(gradient, At3)
        At4, Finished4, steps = optimiser4(F, gradient, At4)
        At5, Finished5, steps = optimiser5(gradient, At5)
        At6, Finished6, steps = optimiser6(gradient, At6)
        At7, Finished7, steps = optimiser7(gradient, At7)
        At8, Finished8, steps = optimiser8(gradient, At8)

        if (Finished1 && Finished2 && Finished3 && Finished4 && Finished5 && Finished6 && Finished7 && Finished8) || i > 400 {
            gm.RenderFrames("gradients.gif")
            break
        }
    }
//     fmt.Println("Found local minima at", At)
    fmt.Println("Used", steps, "number of steps.")






}
