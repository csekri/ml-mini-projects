package main

import (
    "fmt"
    "math"
    "golang.org/x/exp/rand"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/stat"
    "gonum.org/v1/gonum/stat/distmv"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
    "gonum.org/v1/plot/text"
    "gonum.org/v1/plot/font"
    "gonum.org/v1/plot/font/liberation"

    "ml_playground/optimisers"
    "ml_playground/plt"
    "ml_playground/utils"
)

var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


// x1 and x2 are both column vectors
func RadialBasisFunctionKernel(x1, x2 *mat.Dense, varSigma, lengthScale float64) *mat.Dense {
    const EUCLIDEAN_DISTANCE = 2
    NX1, _ := x1.Dims()
    NX2, _ := x2.Dims()
    kernel := mat.NewDense(NX1, NX2, nil)
    kernel.Apply(func (j,i int, v float64) float64 {
                    dist := floats.Distance(mat.Row(nil, j, x1), mat.Row(nil, i, x2), EUCLIDEAN_DISTANCE)
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

func PartialDerivativeOfKernel(X, Kernel *mat.Dense, j, d int, Sigma, Lengthscale float64) *mat.Dense {
    N, _ := Kernel.Dims()
    kernel := mat.NewDense(N, N, nil)
    kernel.Apply(
        func (row, col int, v float64) float64 {
            if row == j {
                return 1.0 / Lengthscale * v * 2.0 * (X.At(col, d) - X.At(j, d))
            }
            if col == j {
                return 1.0 / Lengthscale * v * 2.0 * (X.At(row, d) - X.At(j, d))
            }
            return 0.0
        }, Kernel,
    )
    return kernel
}


func Gradient(X, Y *mat.Dense, varSigma, lengthScale float64) []float64 {
    N, D := X.Dims()
    grad := make([]float64, N*D)
    Kernel := RadialBasisFunctionKernel(X, X, varSigma, lengthScale)
    // add noise to the kernel, this also improves numerical stability
    Kernel.Apply(
        func (j, i int, v float64) float64 {
            if i == j {
                return v + 2.0
            }
            return v
        }, Kernel,
    )
    KernelInv := mat.NewDense(N, N, nil)
    tmp := mat.NewDense(N, N, nil)
    tmp2 := mat.NewDense(N, N, nil)
    err := KernelInv.Inverse(Kernel)
    if err != nil { panic(err) }
    for j:=0; j<N; j++ {
        for d:=0; d<D; d++ {
            partialDerivativeOfK := PartialDerivativeOfKernel(X, Kernel, j, d, varSigma, lengthScale)
            tmp.Mul(KernelInv, partialDerivativeOfK)
            grad[j*D + d] = float64(N) * mat.Trace(tmp)
            tmp2.Mul(Y, Y.T())
            tmp2.Mul(tmp2, tmp)
            tmp2.Mul(tmp2, KernelInv)
            grad[j*D + d] -= mat.Trace(tmp2)
        }
    }
    return grad
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


func OptimisableGrad(Y *mat.Dense, Sigma, LengthScale float64) func ([]float64) []float64 {
    return func (X []float64) []float64 {
        N, _ := Y.Dims()
        XMat := mat.NewDense(N, 2, X)
        return Gradient(XMat, Y, Sigma, LengthScale)
    }
}


/*
SUMMARY
    Draws and visualises samples from an RBF kernel.
PARAMETERS
    N/A
RETURN
    N/A
*/
func VisualiseRBFKernel() {
    linSpaceRes := 200
    linSpace := make([]float64, linSpaceRes)
    for i := range linSpace { linSpace[i] = -6.0 +  12.0 * float64(i) / float64(linSpaceRes) }
    linSpaceVec := mat.NewDense(linSpaceRes, 1, linSpace)
    _K := RadialBasisFunctionKernel(linSpaceVec, linSpaceVec, 1.0, 2.0)
    K := utils.Dense2Sym(_K)
    mu := make([]float64, linSpaceRes)
    multiNormal, _ := distmv.NewNormal(mu, K, randSrc)
    numSamples := 20
    samples := mat.NewDense(linSpaceRes, numSamples, nil)
        for i:=0; i<numSamples; i++ {
            samples.SetCol(i, multiNormal.Rand(nil))
        }
    plt.FunctionMultiPlot(linSpaceVec, samples, "Radial Basis Function Samples", "10cm", "7cm", "rbf.svg")
}

func F(X []float64, Y *mat.Dense, Sigma, LengthScale float64) float64 {
    N, _ := Y.Dims()
    D := len(X) / N
    XMat := mat.NewDense(N, D, X)
    Kernel := RadialBasisFunctionKernel(XMat, XMat, Sigma, LengthScale)
    Kernel.Apply(
        func (j, i int, v float64) float64 {
            if i == j {
                return v + 2.0
            }
            return v
        }, Kernel,
    )
    KernelInv := mat.NewDense(N, N, nil)
    N, YDim := Y.Dims()
    tmp := mat.NewDense(YDim, N, nil)
    tmp2 := mat.NewDense(YDim, YDim, nil)
    err := KernelInv.Inverse(Kernel)
    if err != nil { panic(err) }
    tmp.Mul(Y.T(), KernelInv)
    tmp2.Mul(tmp, Y)
    return float64(N) * math.Log(mat.Det(Kernel)) + mat.Trace(tmp2)
}

func ObjectiveFunc(Y *mat.Dense, Sigma, LengthScale float64) func ([]float64) float64 {
    return func (X []float64) float64 {
        return F(X, Y, Sigma, LengthScale)
    }
}


func main() {
    fmt.Println("")
    NumPoints := 80

    X := GenerateSpiral(NumPoints)
    W := mat.NewDense(10, 2, RandomSlice(10 * 2, 0, 1))
    Y := mat.NewDense(NumPoints, 10, nil)
    Y.Mul(X, W.T())
    Normal := distuv.Normal{0, 1, randSrc}
    Y.Apply(func (j, i int, v float64) float64 { return v + Normal.Rand() }, Y)
    Mu := mat.NewDense(10, 1, nil)
    for y:=0; y<10; y++ {
        Mu.Set(y, 0, stat.Mean(mat.Col(nil, y, Y), nil))
    }

    optimiser := optimisers.Adam(0.3, 0.90, 0.999, 1e-8, 0.5e-1)

    gradientFunc := OptimisableGrad(Y, 1.0, 1.0/2.0)

    Finished := false
    At := make([]float64, 2*NumPoints)
    At = RandomSlice(2*NumPoints, 0, 1)

    for i:=0; i<1000 && !Finished; i++ {
        At, Finished, _ = optimiser(gradientFunc, At)
        fmt.Println("Step", i, "Converged?", Finished, "Loss", F(At, Y, 1.0, 1.0/2.0))
    }
    XPred := mat.NewDense(NumPoints, 2, nil)
    for y:=0; y<NumPoints; y++ {
        for x:=0; x<2; x++ {
            XPred.Set(y, x, At[y*2 + x])
        }
    }

    density := mat.NewDense(300, 300, nil)

    XMin := -0.5 //floats.Min(mat.Col(nil, 0, XPred))
    XMax := 1.5 //floats.Max(mat.Col(nil, 0, XPred))
    YMin := -1.5 //floats.Min(mat.Col(nil, 1, XPred))
    YMax := 0.5 //floats.Max(mat.Col(nil, 1, XPred))
    fmt.Println(XMin, XMax, YMin, YMax)
    for i:=0; i<NumPoints; i++ {
        pdf, _ := distmv.NewNormal(mat.Row(nil, i, XPred), mat.NewSymDense(2, []float64{0.01,0.0,0.0,0.01}), randSrc)
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
    p.Title.Text = `Density Plot of the Latent Space (GPLVM)`
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    img := plt.FillImage(&m, pal)
    Height, Width := m.Matrix.Dims()
    pImg := plotter.NewImage(img, 0, 0, float64(Width), float64(Height))
    p.Add(pImg)
    p.Save(4*vg.Inch, 4*vg.Inch, "density_plot.png")

    p = plot.New()
    p.Title.Text = "Scatter Plot of the Latent Space (GPLVM)"
    p.X.Label.Text = `$x$`
    p.Y.Label.Text = `$y$`
    pal = plt.DesignedPalette{Type: plt.KINDLMANN_PALETTE, Num: len(mat.Col(nil, 0, XPred))}
    sc := plt.MakeScatterUnicorn(mat.Col(nil, 0, XPred), mat.Col(nil, 1, XPred), plt.CIRCLE_POINT_MARKER, 3.0, pal)
    p.Add(sc)
    p.Save(300, 300, "prediction_scatter_plot.svg")



}