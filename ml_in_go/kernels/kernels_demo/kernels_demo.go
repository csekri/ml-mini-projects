package main

import (
    "gonum.org/v1/gonum/mat"
    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/stat/distmv"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"

    "ml_playground/utils"
    "ml_playground/plt"
    "ml_playground/kernels"
)

// random number seed and source
var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


/*
SUMMARY
    Visualises the kernel with the given parameters.
PARAMETERS
    Params kernels.Parameters: the type and parameters of the kernel
    NumSamples int: the number of samples drawn from the distribution
RETURN
    *plot.Plot: a plot containing the lines of the samples
*/
func VisualiseKernel(Params kernels.Parameters, NumSamples int) *plot.Plot {
    linSpaceRes := 200
    linSpace := utils.Linspace(-6.0, 6.0, linSpaceRes)
    linSpaceVec := mat.NewDense(linSpaceRes, 1, linSpace)
    _K := kernels.Kernel(linSpaceVec, linSpaceVec, Params)
    K := utils.Dense2Sym(_K)
    mu := make([]float64, linSpaceRes)
    multiNormal, _ := distmv.NewNormal(mu, K, randSrc)
    samples := mat.NewDense(linSpaceRes, NumSamples, nil)
        for i:=0; i<NumSamples; i++ {
            samples.SetCol(i, multiNormal.Rand(nil))
        }
    p := plot.New()
    lines := plt.MakeMultiLineUnicorn(linSpace, samples, 1.2, plt.DesignedPalette{Type: plt.RANDOM_PALETTE, Num: NumSamples}, nil)
    for _, line := range lines { p.Add(line) }
//     p.Add(lines...) // this apparently doesn't work
    p.Title.Text, p.X.Label.Text, p.Y.Label.Text = "Radial Basis Function Samples", "x", "y"
    return p
}


/*
SUMMARY
    Visualises the kernel covariance matrix with the given parameters.
PARAMETERS
    Params kernels.Parameters: the type and parameters of the kernel
RETURN
    *plot.Plot: a plot containing the density plot
*/
func VisualiseKernelMatrix(Params kernels.Parameters) *plot.Plot {
    linSpaceRes := 200
    linSpace := utils.Linspace(-6.0, 6.0, linSpaceRes)
    linSpaceVec := mat.NewDense(linSpaceRes, 1, linSpace)
    K := kernels.Kernel(linSpaceVec, linSpaceVec, Params)

    p := plot.New()
    m := plt.MatrixHeatMap{
        Matrix: K,
        XRange: plt.Range{-5.0, 5.0},
        YRange: plt.Range{-5.0, 5.0},
    }
    pal := plt.DesignedPalette{Type: plt.KINDLMANN_PALETTE, Num: 100}
    img := plt.FillImage(&m, pal)
    Height, Width := m.Matrix.Dims()
    pImg := plotter.NewImage(img, 0, 0, float64(Width), float64(Height))
    p.Add(pImg)
    return p
}


// We plot RBF, linear, periodic matrix and samples figures, 6 figures altogether.
func main() {
    RBFParams := kernels.Parameters{Type: kernels.RBF, VarSigma: 2.0, LengthScale: 1.0}
    LinearParams := kernels.Parameters{Type: kernels.LINEAR, VarSigma: 2.0}
    PeriodicParams := kernels.Parameters{Type: kernels.PERIODIC, VarSigma: 2.0, LengthScale: 1.0, Period: 3.5}

    p := VisualiseKernel(RBFParams, 10)
    p.Title.Text, p.X.Label.Text, p.Y.Label.Text = "Radial Basis Function Kernel Samples", "x", "y"
    p.Save(300, 200, "rbf.svg")
    p = VisualiseKernel(LinearParams, 10)
    p.Title.Text, p.X.Label.Text, p.Y.Label.Text = "Linear Kernel Samples", "x", "y"
    p.Save(300, 200, "linear.svg")
    p = VisualiseKernel(PeriodicParams, 10)
    p.Title.Text, p.X.Label.Text, p.Y.Label.Text = "Periodic Kernel Samples", "x", "y"
    p.Save(300, 200, "periodic.svg")

    p = VisualiseKernelMatrix(RBFParams)
    p.Title.Text, p.Title.TextStyle.Font.Size = "RBF Kernel Covariance", 20
    p.Save(300, 300, "rbf_matrix.png")
    p = VisualiseKernelMatrix(LinearParams)
    p.Title.Text, p.Title.TextStyle.Font.Size = "Linear Kernel Covariance", 20
    p.Save(300, 300, "linear_matrix.png")
    p = VisualiseKernelMatrix(PeriodicParams)
    p.Title.Text, p.Title.TextStyle.Font.Size = "Periodic Kernel Covariance", 20
    p.Save(300, 300, "periodic_matrix.png")
}