package main

import (
    "fmt"
    "math"
    "gonum.org/v1/gonum/mat"
    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/stat/distmv"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"

    "ml_playground/utils"
    "ml_playground/plt"
    "ml_playground/gaussian_processes"
)

// random number seed and source
var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


/*
SUMMARY
    Draws and visualises samples from a Gaussian process.
PARAMETERS
    X *mat.Dense: column vector with the x coordinates
    Y *mat.Dense: column vector which is f(X)+noise
    lengthScale float64: parameter of RBF
    varSigma float64: parameter of the RBF
    betaNoise float64: the precision of the noise
RETURN
    N/A
*/
func VisualizeGaussianProcessSamples(X, Y, XStar *mat.Dense, lengthScale, varSigma, betaNoise float64) {
    numSamples := 50
    mu, sigma := gaussian_processes.GaussianProcessPrediction(X, Y, XStar, lengthScale, varSigma, betaNoise)
    multiNormal, _ := distmv.NewNormal(utils.Flatten(mu, true), sigma, randSrc)
    MuDim, _ := mu.Dims()
    samples := mat.NewDense(MuDim, numSamples, nil)
    for i:=0; i<numSamples; i++ {
        samples.SetCol(i, multiNormal.Rand(nil))
    }
    p := plot.New()
    lines := plt.MakeMultiLineUnicorn(mat.Col(nil, 0, XStar), samples, 1.2, plt.DesignedPalette{Type: plt.RANDOM_PALETTE, Num: numSamples, Extra: 6}, nil)
    for _, line := range lines { p.Add(line) }
//     p.Add(lines...) // this apparently doesn't work
    p.Title.Text, p.X.Label.Text, p.Y.Label.Text = "Samples from a Gaussian Process", "x", "y"
    p.Save(300, 200, "gp_pred.svg")

}


/*
SUMMARY
    Visualises in a heatmap and contour plot where we believe the function runs.
PARAMETERS
    Wid int: width of the heatmap in pixels
    Hei int: height of the heatmap in pixels
    X *mat.Dense: column vector with the x coordinates
    Y *mat.Dense: column vector which is f(X)+noise
    XStar *mat.Dense: a column vector where each row is an unknown locations where we would like to find Ystar=f(Xstar)
    lengthScale float64: parameter of RBF
    varSigma float64: parameter of the RBF
    betaNoise float64: the precision of the noise
RETURN
    N/A
*/
func VisualiseGaussianProcessBelief(Wid, Hei int, X, Y, XStar *mat.Dense, lengthScale, varSigma, betaNoise float64) {
    mu, sigma := gaussian_processes.GaussianProcessPrediction(X, Y, XStar, lengthScale, varSigma, betaNoise)
    XRang := plt.Range{-4.0, 4.0}
    YRang := plt.Range{-4.0, 4.0}
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
    pal := plt.DesignedPalette{Type: plt.BLACK_BODY_PALETTE, Num: 100}
    heights := make([]float64, 30)
    for i := range heights { heights[i] = 0.02 * float64(i+1) }
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

    xSize, _ := X.Dims()
    scatter := plt.MakeScatterUnicorn(mat.Col(nil, 0, X), mat.Col(nil, 0, Y), plt.CIRCLE_POINT_MARKER, 4.0, plt.DesignedPalette{Type: plt.UNI_PALETTE, Extra: 0x000000ff, Num: xSize})
    p = plot.New()
    p.Title.Text = `Contour map of our belief`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    p.Add(contour)
    p.Add(scatter)
    p.Legend.Add("data points", scatter)
    p.Save(6*vg.Inch, 4*vg.Inch, "belief_contourmap.svg")
}


/*
We generate ten points from a sine curve and some noise.
Then we compute the posterior of the gaussian process and visualise samples from the GP.
We compute the marginal distribution at each point in XStar, and plot it in a heatmap and contour plot.
*/
func main() {
    fmt.Println("")

    N := 10
    linSpace := utils.Linspace(-3.0, 3.0, N)
    linSpaceVec := mat.NewDense(N, 1, linSpace)

    Y := mat.NewDense(N, 1, nil)
    normal := distuv.Normal{0.0, 1.0, randSrc}
    Y.Apply(func (j, i int, v float64) float64 {return 2*math.Sin(v) + 0.3*normal.Rand()}, linSpaceVec)

    XStarRes := 300
    XStar := utils.Linspace(-4.0, 4.0, XStarRes)
    XStarVec := mat.NewDense(XStarRes, 1, XStar)

    varSigma, lengthScale, beta := 5.8, 7.5, 0.4
    VisualizeGaussianProcessSamples(linSpaceVec, Y, XStarVec, varSigma, lengthScale, beta)
    VisualiseGaussianProcessBelief(200, 200, linSpaceVec, Y, XStarVec, varSigma, lengthScale, beta)
}