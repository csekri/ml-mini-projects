package main

import (
    "fmt"
    "math"
//     "image/color"
    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/stat/distmv"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"

    "ml_playground/plt"
)

// random number seed and source
var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


/*
SUMMARY
    Infers the linear map that resulted in the observed data Y using
    eigenvalue/eigenvector decomposition
PARAMETERS
    Y *mat.Dense: the observed data
    LatentSpaceDimensions int: the number of dimensions in the latent space
RETURN
    *mat.Dense: the learned linear map
*/
func MaximumLikelihoodWeights(Y *mat.Dense, LatentSpaceDimensions int) *mat.Dense {
    _, YW := Y.Dims()
    CovarianceMatrix := mat.NewSymDense(YW, nil)
    stat.CovarianceMatrix(CovarianceMatrix, Y, nil)
    var SymEigenTool mat.EigenSym
    ok := SymEigenTool.Factorize(CovarianceMatrix, true)
    if !ok {
        fmt.Println("Eigenvalue decomposition has failed")
        panic("")
    }
    EigenVectors := mat.NewDense(YW, YW, nil)
    SymEigenTool.VectorsTo(EigenVectors)
    EigenValues := SymEigenTool.Values(nil)
    Weights := mat.NewDense(YW, LatentSpaceDimensions, nil)

    Indices := make([]int, len(EigenValues))
    floats.Argsort(EigenValues, Indices)
    for x:=0; x<LatentSpaceDimensions; x++ {
        Weights.SetCol(x, mat.Col(nil, Indices[len(Indices)-1-x], EigenVectors))
    }
    return Weights
}

/*
SUMMARY
    For any one point in the observed space determines its mean and covariance
    in the latent space
PARAMETERS
    W *mat.Dense: the linear map
    y *mat.Dense: the one point we query its location in the latent space (column vector)
    Mu *mat.Dense: the mean of each dimension of the points in observed space
        (has the same dimensions as y)
    Beta float64: precision (inverse of variance)
RETURN
    *mat.Dense: mean of the posterior distribution
    *mat.Dense: covariance of the posterior distribution
*/
func Posterior(W, y, Mu *mat.Dense, Beta float64) (*mat.Dense, *mat.Dense) {
    WWidth, WHeight := W.Dims()
    tmp := mat.NewDense(WWidth, WWidth, nil)
    tmp.Mul(W, W.T())
    tmp.Apply(func (j,i int, v float64) float64 { if i == j { return v+1/Beta }; return v }, tmp)
    err := tmp.Inverse(tmp)
    if err != nil { panic(err) }

    tmp2 := mat.NewDense(WHeight, WWidth, nil)
    tmp2.Mul(W.T(), tmp)

    YHeight, YWidth := y.Dims()
    YMinusMu := mat.NewDense(YHeight, YWidth, nil)
    YMinusMu.Sub(y, Mu)

    PosteriorMu := mat.NewDense(WHeight, YWidth, nil)
    PosteriorMu.Mul(tmp2, YMinusMu)

    PosteriorSigma := mat.NewDense(WHeight, WHeight, nil)
    PosteriorSigma.Mul(tmp2, W)
    PosteriorSigma.Apply(func (i,j int, v float64) float64 {  if i == j { return 1-v }; return -v }, PosteriorSigma)

    return PosteriorMu, PosteriorSigma
}


/*
SUMMARY
    Generates our sample data for the demonstration. The data forms a spiral.
PARAMETERS
    Num int: the number of points
RETURN
    *mat.Dense: matrix where each row is a point
*/
func GenerateSpiral(Num int) *mat.Dense {
    X := mat.NewDense(Num, 2, nil)
    Range := func (j int) float64 { return float64(j) / float64(Num) * 3.0 * 3.1416 }
    for y:=0; y<Num; y++ {
        t := Range(y)
        X.Set(y, 0, t * math.Sin(t))
        X.Set(y, 1, t * math.Cos(t))
    }
    return X
}


/*
SUMMARY
    Populates a slice with random numbers drawn from normal distribution.
PARAMETERS
    Num int: the length of the slice
    mu float64: mean of the normal distribution
    sigma float64: standard deviation of the normal distribution
*/
func RandomSlice(Num int, mu, sigma float64) []float64 {
    normal := distuv.Normal{mu, sigma, randSrc}
    slice := make([]float64, Num)
    for i := range slice { slice[i] = normal.Rand() }
    return slice
}

// // this type is used to define the BW colour palette
// type BWPalette string
//
//
// /*
// SUMMARY
//     Defines the palette interface for BWPalette
// PARAMETERS
//     N/A
// RETURN
//     []color.Color: the set of colours forming the palette (order does matter)
// */
// func (pal BWPalette) Colors() []color.Color {
//     colours := make([]color.Color, 256)
//     for i := range colours {
//         colours[i] = color.RGBA{R:uint8(i), G:uint8(i), B:uint8(i), A:255}
//     }
//     return colours
// }


/*
We generate a spiral in 2d.
We apply a random linear map on the spiral.
We have the spiral embedded in 10d.
Add some noise.
We infer the linear map.
Compute the posterior for each point in the observed points.
Plot the result.
*/
func main() {
    fmt.Println("")

    X := GenerateSpiral(100)
    W := mat.NewDense(10, 2, RandomSlice(10 * 2, 0, 1))
    Y := mat.NewDense(100, 10, nil)
    Y.Mul(X, W.T())
    Normal := distuv.Normal{0, 1, randSrc}
    Y.Apply(func (j, i int, v float64) float64 { return v + Normal.Rand() }, Y)
    Mu := mat.NewDense(10, 1, nil)
    for y:=0; y<10; y++ {
        Mu.Set(y, 0, stat.Mean(mat.Col(nil, y, Y), nil))
    }

    W = MaximumLikelihoodWeights(Y, 2)

    XPred := mat.NewDense(100, 2, nil)
    VarSigma := mat.NewSymDense(2, nil)

    for i:=0; i<100; i++ {
        column := mat.NewDense(10, 1, mat.Row(nil, i, Y))
        mu, sigma := Posterior(W, column, Mu, 0.5)
        XPred.SetRow(i, mat.Col(nil, 0, mu))
        for y:=0; y<2; y++ {
            for x:=0; x<2; x++ {
                VarSigma.SetSym(y, x, sigma.At(y, x))
            }
        }
    }

    density := mat.NewDense(300, 300, nil)

    XMin := floats.Min(mat.Col(nil, 0, XPred))
    XMax := floats.Max(mat.Col(nil, 0, XPred))
    YMin := floats.Min(mat.Col(nil, 1, XPred))
    YMax := floats.Max(mat.Col(nil, 1, XPred))
    fmt.Println(XMin, XMax, YMin, YMax)
    for i:=0; i<100; i++ {
        pdf, _ := distmv.NewNormal(mat.Row(nil, i, XPred), VarSigma, randSrc)
        for j:=0; j<300; j++ {
            for k:=0; k<300; k++ {
                y := YMin+float64(k)/300.0*(YMax-YMin)
                x := XMin+float64(j)/300.0*(XMax-XMin)
                density.Set(j, k, density.At(j, k) + pdf.Prob([]float64{x, y}))
            }
        }
    }

    m := plt.MatrixHeatMap{
        Matrix: density,
        XRange: plt.Range{XMin, XMax},
        YRange: plt.Range{YMin, YMax},
    }

    pal := plt.DesignedPalette{Type: plt.BLACK_BODY_PALETTE, Num: 256}
    heights := make([]float64, 50)
    for i := range heights { heights[i] = 0.1 * float64(i+1) }
    fonts := font.NewCache(liberation.Collection())
	plot.DefaultTextHandler = text.Latex{
		Fonts: fonts,
	}
    p := plot.New()
    p.Title.Text = `Density Plot of the Latent Space (PCA)`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    img := plt.FillImage(&m, pal)
    Height, Width := m.Matrix.Dims()
    pImg := plotter.NewImage(img, 0, 0, float64(Width), float64(Height))
    p.Add(pImg)
    p.Save(4*vg.Inch, 4*vg.Inch, "density_plot.png")

    p = plot.New()
    p.Title.Text = "Scatter Plot of the Latent Space (PCA)"
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    pal = plt.DesignedPalette{Type: plt.KINDLMANN_PALETTE, Num: len(mat.Col(nil, 0, XPred))}
    sc := plt.MakeScatterUnicorn(mat.Col(nil, 0, XPred), mat.Col(nil, 1, XPred), plt.CIRCLE_POINT_MARKER, 3.0, pal)
    p.Add(sc)
    p.Save(300, 300, "prediction_scatter_plot.svg")
}



