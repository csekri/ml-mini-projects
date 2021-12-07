package main

import (
    "fmt"
    "math"
    "image/color"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"


    "ml_playground/plt"
    "ml_playground/pic"
    "ml_playground/optimisers"
)


/*
SUMMARY
    Creates one frame in the gradient descent animation.
PARAMETERS
    function func ([]float64) float64: the objective function
    Ats [][][]float64: the path of descending for all the optimisers
    XMin: the domain's minimal x bound
    XMax: the domain's maximal x bound
    YMin: the domain's minimal y bound
    YMax: the domain's maximal y bound
RETURN
    *plot.Plot: plot containing a snapshot of the optimisation process
*/
func DescentPlot(function func ([]float64) float64, Ats [][][]float64, XMin, XMax, YMin, YMax float64) *plot.Plot {
    // the colour of each optimiser
    myPalette := []color.Color{color.RGBA{200, 0, 0, 255},
                             color.RGBA{0, 200, 0, 255},
                             color.RGBA{0, 0, 200, 255},
                             color.RGBA{170, 170, 0, 255},
                             color.RGBA{200, 0, 200, 255},
                             color.RGBA{0, 200, 200, 255},
                             color.RGBA{0, 0, 0, 255},
                             color.RGBA{170, 170, 170, 255},
                             color.RGBA{255, 191, 0, 255},
                              }
    ballPal := plt.CustomPalette{myPalette}
    blackPal := plt.DesignedPalette{Type: plt.UNI_PALETTE, Num: 1, Extra: 0x00000044}
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

    heights := make([]float64, 50)
    for i := range heights { heights[i] = 0.01 * math.Exp(float64(i+1)/4) }
    contour := plotter.NewContour(&m, heights, blackPal)

	ballX := make([]float64, len(Ats))
	ballY := make([]float64, len(Ats))
	for i := range Ats {
        LastIndex := len(Ats[i]) - 1
	    ballX[i], ballY[i] = Ats[i][LastIndex][0], Ats[i][LastIndex][1]
	}
	sc := plt.MakeScatterUnicorn(ballX, ballY, plt.CIRCLE_POINT_MARKER, 5.0, ballPal)

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
            case 8:
                p.Legend.Add("DampenedMomentum", l)
        }
	}
    p.Add(sc)
    return p
}


/*
Creates an animation for each optimiser. The objective function:
f(x,y) = 1/10(1-x)^2 + (y-x^2)^2,
partial x:
d/dx = 1/5 (20 x^3 - 20 x y + x - 1)
partial y:
d/dy = 2 (y - x^2),
The minimum is at (1,1) with no other local minimum.
All optimisers are initialised at position (0.9, -0.3).
*/
func main() {
    XStart := 0.9
    YStart := -0.3

    At1 := []float64{XStart,YStart}
    At2 := []float64{XStart,YStart}
    At3 := []float64{XStart,YStart}
    At4 := []float64{XStart,YStart}
    At5 := []float64{XStart,YStart}
    At6 := []float64{XStart,YStart}
    At7 := []float64{XStart,YStart}
    At8 := []float64{XStart,YStart}
    F := func (x []float64) float64 {
        return 1.0/10.0*math.Pow(1-x[0], 2) + math.Pow(x[1]-x[0]*x[0], 2)
    }
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

    epsilon := 1e-4
    optimiser1 := optimisers.SGD(0.01, epsilon)
    optimiser2 := optimisers.SGDMomentum(0.01, 0.95, epsilon)
    optimiser3 := optimisers.NesterovAcceleratedGradient(0.01, 0.95, epsilon)
    optimiser4 := optimisers.BacktrackingLineSearch(0.9, epsilon)
    optimiser5 := optimisers.Adagrad(0.8, 0.1, 1e-7, epsilon)
    optimiser6 := optimisers.Adadelta(0.95, 1e-7, epsilon)
    optimiser7 := optimisers.RMSprop(0.01, 0.9, 1e-8, epsilon)
    optimiser8 := optimisers.Adam(0.5, 0.9, 0.999, 1e-8, epsilon)

    gm := pic.GifMaker{Width: 600, Height: 600, Delay:1}

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
        if i % 3 == 0 {
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

        fmt.Println(Finished1, Finished2, Finished3, Finished4, Finished5, Finished6, Finished7, Finished8)

        // optimisation finishes when the first optimiser finishes
        if (Finished1 || Finished2 || Finished3 || Finished4 || Finished5 || Finished6 || Finished7 || Finished8) {
            gm.RenderFrames("gradients.gif")
            break
        }
    }
    fmt.Println("Used", steps, "number of steps.")
}
