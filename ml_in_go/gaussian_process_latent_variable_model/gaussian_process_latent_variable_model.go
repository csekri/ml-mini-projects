package main

import (
    "fmt"
    "math"
    "image/color"
    "gonum.org/v1/gonum/mat"
    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/stat/distmv"
    "ml_playground/plt"
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/vg/draw"
    "gonum.org/v1/plot/palette"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"

)

var randSeed = 10
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

// x1 and x2 are both column vectors
func RadialBasisFunctionKernel(x1, x2 *mat.Dense, varSigma, lengthScale float64) *mat.Dense {
    const EUCLIDEAN_DISTANCE = 2
    NX1, _ := x1.Dims()
    NX2, _ := x2.Dims()
    kernel := mat.NewDense(NX1, NX2, nil)
    kernel.Apply(func (j,i int, v float64) float64 {
                    dist := x1.At(j,0) - x2.At(i,0)
                    return varSigma*math.Exp(-dist * dist / lengthScale)
                 }, kernel)
    // adding epsilon*(unit matrix) for numerical stability
    kernel.Apply(func (j,i int, v float64) float64 {
                    if i == j {
                        return v + 0.0001
                    }
                    return v
                 }, kernel)
    return kernel
}

func Sym2Dense(in *mat.SymDense) *mat.Dense {
    N, _ := in.Dims()
    sym := mat.NewDense(N, N, nil)
    for y:=0; y<N; y++ {
        for x:=0; x<N; x++ {
            sym.Set(y,x, in.At(y,x))
        }
    }
    return sym
}

func Dense2Sym(in *mat.Dense) *mat.SymDense {
    N, _ := in.Dims()
    ordinary := mat.NewSymDense(N, nil)
    for y:=0; y<N; y++ {
        for x:=0; x<N; x++ {
            ordinary.SetSym(y,x, in.At(y,x))
        }
    }
    return ordinary
}

func GaussianProcessPrediction(X, Y, XStar *mat.Dense, lengthScale, varSigma, betaNoise float64) (*mat.Dense, *mat.SymDense) {
    KStarX := RadialBasisFunctionKernel(XStar, X, lengthScale, varSigma)
    KXX := RadialBasisFunctionKernel(X, X, lengthScale, varSigma)
    KXX.Apply(func (j,i int, v float64) float64 {
                    if i == j { return v + 1 / betaNoise }
                    return v
                }, KXX)
    KStarStar := RadialBasisFunctionKernel(XStar, XStar, lengthScale, varSigma)
    KXX.Apply(func (j,i int, v float64) float64 {
                    if i == j {
                        return v + 0.0001
                    }
                    return v
                }, KXX)
    KXX.Inverse(KXX)
    KXXSize, _ := KXX.Dims()
    KStarXSize, _ := KStarX.Dims()
    tmp := mat.NewDense(KStarXSize, KXXSize, nil)
    tmp.Mul(KStarX, KXX)
    mu := mat.NewDense(KStarXSize, 1, nil)
    mu.Mul(tmp, Y)

    sigma := mat.NewDense(KStarXSize, KStarXSize, nil)
    sigma.Mul(tmp, KStarX.T())
    sigma.Sub(KStarStar, sigma)
    return mu, Dense2Sym(sigma)
}

func VisualizeGaussianProcess(X, Y, XStar *mat.Dense, lengthScale, varSigma, betaNoise float64) {
    numSamples := 100
    mu, sigma := GaussianProcessPrediction(X, Y, XStar, lengthScale, varSigma, betaNoise)
    multiNormal, _ := distmv.NewNormal(flatten(mu), sigma, randSrc)
    MuDim, _ := mu.Dims()
    samples := mat.NewDense(MuDim, numSamples, nil)
    for i:=0; i<numSamples; i++ {
        samples.SetCol(i, multiNormal.Rand(nil))
    }
    plt.FunctionMultiPlot(XStar, samples, "Samples from a Gaussian Process", "10cm", "7cm", "gp_pred.svg")
}

func VisualiseRBFKernel() {
    linSpaceRes := 200
    linSpace := make([]float64, linSpaceRes)
    for i := range linSpace { linSpace[i] = -6.0 +  12.0 * float64(i) / float64(linSpaceRes) }
    linSpaceVec := mat.NewDense(linSpaceRes, 1, linSpace)
    _K := RadialBasisFunctionKernel(linSpaceVec, linSpaceVec, 1.0, 2.0)
    K := Dense2Sym(_K)
    mu := make([]float64, linSpaceRes)
    multiNormal, _ := distmv.NewNormal(mu, K, randSrc)
    numSamples := 20
    samples := mat.NewDense(linSpaceRes, numSamples, nil)
        for i:=0; i<numSamples; i++ {
            samples.SetCol(i, multiNormal.Rand(nil))
        }
    plt.FunctionMultiPlot(linSpaceVec, samples, "Radial Basis Function Samples", "10cm", "7cm", "rbf.svg")
}

func VisualiseGaussianProcessBelief(Wid, Hei int, X, Y, XStar *mat.Dense, lengthScale, varSigma, betaNoise float64) {
    mu, sigma := GaussianProcessPrediction(X, Y, XStar, lengthScale, varSigma, betaNoise)
    XRang := plt.Range{-6.0, 6.0}
    YRang := plt.Range{-6.0, 6.0}
    m := plt.FuncHeatMap{
                Function: func (x,y float64) float64 {
                    MuDim, _ := mu.Dims()
//                     j := int(y - YRang.Min * float64(Hei) / (YRang.Max - YRang.Min))
                    i := int((x - XRang.Min) * float64(MuDim) / (XRang.Max - XRang.Min))
                    normal := distuv.Normal{mu.At(i,0), sigma.At(i,i), randSrc}
                    return normal.Prob(y)
                },
                Height: Wid,
                Width: Hei,
                XRange: XRang,
                YRange: YRang,
    }
    pal := palette.Heat(100, 1)
//     heatmap := plotter.NewHeatMap(&m, pal)
    heights := make([]float64, 50)
    for i := range heights { heights[i] = 0.01 * float64(i+1) }
    contour := plotter.NewContour(&m, heights, pal)
    fonts := font.NewCache(liberation.Collection())
	plot.DefaultTextHandler = text.Latex{
		Fonts: fonts,
	}
    p := plot.New()
    p.Title.Text = `Heatmap of our belief`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    img := plt.FillImage(&m, pal)
    pImg := plotter.NewImage(img, 0, 0, float64(m.Width), float64(m.Height))
    p.Add(pImg)
    p.Save(6*vg.Inch, 4*vg.Inch, "belief_heatmap.png")
//     p.Add(heatmap)
//     p.Save(6*vg.Inch, 4*vg.Inch, "belief_heatmap.png")

    XSize, _ := X.Dims()
    ScatterData := make(plotter.XYs, XSize)
    for i := range ScatterData {
        ScatterData[i].X = X.At(i,0)
        ScatterData[i].Y = Y.At(i,0)
	}
    sc, err := plotter.NewScatter(ScatterData)
	if err != nil {
		panic(err)
	}

    p = plot.New()
    p.Title.Text = `Contour map of our belief`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    p.Add(contour)
    sc.GlyphStyleFunc = func(i int) draw.GlyphStyle { return draw.GlyphStyle{
                                                        Color: color.RGBA{A:255},
                                                        Radius: 5, Shape: draw.CircleGlyph{},
                                                     }
	}
    p.Add(sc)
    p.Save(6*vg.Inch, 4*vg.Inch, "belief_contourmap.svg")
}


func main() {
    fmt.Println("")
    VisualiseRBFKernel()

    N := 5
    linSpace := make([]float64, N)
    for i := range linSpace { linSpace[i] = -4.0 +  8.0 * float64(i) / float64(N) }
    linSpaceVec := mat.NewDense(N, 1, linSpace)

    Y := mat.NewDense(N, 1, nil)
    normal := distuv.Normal{0.0, 1.0, randSrc}
    Y.Apply(func (j, i int, v float64) float64 {return 2*math.Sin(float64(j)) + 0.3*normal.Rand()}, Y)

    XStarRes := 300
    XStar := make([]float64, XStarRes)
    for i := range XStar { XStar[i] = -6.0 +  12.0 * float64(i) / float64(XStarRes) }
    XStarVec := mat.NewDense(XStarRes, 1, XStar)

    VisualizeGaussianProcess(linSpaceVec, Y, XStarVec, 2.0, 1.0, 1.5)
    VisualiseGaussianProcessBelief(200, 200, linSpaceVec, Y, XStarVec, 2.0, 1.0, 1.5)



}