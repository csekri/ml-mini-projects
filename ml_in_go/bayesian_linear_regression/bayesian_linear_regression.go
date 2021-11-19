package main

import (
    "fmt"
    "strconv"
    "os"

    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/stat/distmv"

    "gonum.org/v1/gonum/mat"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/palette"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/vg/vgimg"
    "gonum.org/v1/plot/vg/draw"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"

    "ml_playground/plt"
)

var randSeed = 11
var randSrc = rand.NewSource(uint64(randSeed))

//flatten a matrix into a slice
func flatten(matrix *mat.Dense) []float64 {
    height, width := matrix.Dims()
    flattened := make([]float64, height*width)
    for y:=0; y<height; y++ {
        for x:=0; x<width; x++ {
            flattened[y*width + x] = matrix.At(y,x)
        }
    }
    return flattened
}

// returns one plot of the many subplots depicting a heatmap of the distribution
func PlotDistribution(mu []float64, sigma *mat.SymDense, filename string) *plot.Plot {
    multiNormal, _ := distmv.NewNormal(mu, sigma, randSrc)
    m := plt.FuncHeatMap{Function: func (x,y float64) float64 {return multiNormal.Prob([]float64{x, y})},
                     Height: 200,
                     Width: 200,
                     XRange: plt.Range{-1.5, 1.5},
                     YRange: plt.Range{-1.5, 1.5},
    }
    pal := palette.Heat(12, 1)
    heatmap := plotter.NewHeatMap(&m, pal)
    fonts := font.NewCache(liberation.Collection())
	plot.DefaultTextHandler = text.Latex{
		Fonts: fonts,
	}
    p := plot.New()
    p.Title.Text = `Distribution of the parameters`
    p.X.Label.Text = `$w0$`
    p.Y.Label.Text = `$w1$`
    p.Add(heatmap)
    return p
}

// computes the posterior distribution
func Posterior(X, y *mat.Dense) (*mat.Dense, *mat.SymDense) {
    beta := 1/ 0.3
    sigma_0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
    sigma_0_inv := mat.NewDense(2, 2, nil)
    _ = sigma_0_inv.Inverse(sigma_0)
    w_0 := mat.NewDense(2, 1, []float64{0, 0})
    w := mat.NewDense(2, 1, nil)
    tmp := mat.NewDense(2, 2, nil)
    tmp.Mul(X.T(), X)
    tmp.Apply(func (j,i int, v float64) float64 { return v*beta}, tmp)
    tmp.Add(sigma_0_inv, tmp)
    _ = tmp.Inverse(tmp)
    sigma := mat.NewSymDense(2, []float64{tmp.At(0,0), tmp.At(0,1), tmp.At(1,0), tmp.At(1,1)})
    tmp2 := mat.NewDense(2, 1, nil)
    tmp3 := mat.NewDense(2, 1, nil)
    tmp2.Mul(sigma_0_inv, w_0)
    tmp3.Mul(X.T(), y)
    tmp3.Apply(func (j,i int, v float64) float64 { return v*beta}, tmp3)

    w.Add(tmp2, tmp3)
    w.Mul(tmp, w)

    return w, sigma
}

// gives the marginal distribution
func PredictivePosterior(xStar, X, y *mat.Dense) (float64, float64) {
//     N, _ := X.Dims()
    meanN, sigmaN := Posterior(X, y)
    meanStar := mat.NewDense(1, 1, nil)
    meanStar.Mul(meanN.T(), xStar)
    tmp := mat.NewDense(1, 2, nil)
    tmp.Mul(xStar.T(), sigmaN)
    _sigmaStar := mat.NewDense(1, 1, nil)
    _sigmaStar.Mul(tmp, xStar)
    sigmaStar := 0.3 + _sigmaStar.At(0,0)
    return meanStar.At(0,0), sigmaStar
}

func Subplots(rows, cols int, plots [][]*plot.Plot) {
    img := vgimg.New(vg.Points(800), vg.Points(800))
    dc := draw.New(img)
	t := draw.Tiles{
		Rows:      rows,
		Cols:      cols,
		PadX:      vg.Millimeter,
		PadY:      vg.Millimeter,
		PadTop:    vg.Points(2),
		PadBottom: vg.Points(2),
		PadLeft:   vg.Points(2),
		PadRight:  vg.Points(2),
	}
	canvases := plot.Align(plots, t, dc)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			if plots[j][i] != nil {
				plots[j][i].Draw(canvases[j][i])
			}
		}
	}
	w, err := os.Create("update_of_belief.png")
	if err != nil {
		panic(err)
	}
	defer w.Close()
	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(w); err != nil {
		panic(err)
	}
}




func main() {
    // number of points
    N := 200
    //

    // generating the datapoints
    X := mat.NewDense(N, 2, nil)
    for pt:=0; pt<N; pt++ {
        X.Set(pt,0, -1.0 + float64(pt)/float64(N) * 2.0)
        X.Set(pt,1, 1.0)
    }
    //

    // generating y with noise
    beta := 1 / 0.3
    mu := mat.NewDense(2, 1, []float64{-1.3, 0.5})
    y := mat.NewDense(N, 1, nil)
    y.Mul(X, mu)
    normal := distuv.Normal{0.0, 1 / beta, randSrc}
    noise := mat.NewDense(N, 1, nil)
    noise.Apply(func (j, i int, v float64) float64 {return normal.Rand()}, noise)
    y.Add(y, noise)
    //

	const rows, cols = 4, 4
	plots := make([][]*plot.Plot, rows)

    // in each iteration we use one more point, and do calculations
    for row:=0; row<rows; row++ {
        plots[row] = make([]*plot.Plot, cols)
        for col:=0; col<cols; col++ {
            i := 1 + (col*rows + row) * 13
            X_few := mat.NewDense(i, 2, nil)
            y_few := mat.NewDense(i, 1, nil)
            for idx:=0; idx<i; idx++ {
                X_few.Set(idx,0, X.At(idx,0))
                X_few.Set(idx,1, 1)
                y_few.Set(idx,0, y.At(idx, 0))
            }
            fmt.Println("Distribution after seeing " + strconv.Itoa(i) + " data points")
            w, sigma := Posterior(X_few, y_few)
            plots[row][col] = PlotDistribution(flatten(w), sigma, "out" + strconv.Itoa(i) + ".png")

            predMu, predSigma := PredictivePosterior(mat.NewDense(2,1, []float64{0.5, 1}), X_few, y_few)
            fmt.Println("The real value at 0.5 is " + fmt.Sprintf("%.3f", -1.3*0.5 + 0.5))
            fmt.Println("The estimated value is " + fmt.Sprintf("%.3f", predMu))
            fmt.Println("The variance is " + fmt.Sprintf("%.3f", predSigma) + "\n")

        }
    }
    Subplots(rows, cols, plots)
    //









}