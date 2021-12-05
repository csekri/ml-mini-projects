package main

import (
    "fmt"
    "math"
    "golang.org/x/exp/rand"

    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"

    "gonum.org/v1/plot"

    "ml_playground/utils"
    "ml_playground/plt"
    "ml_playground/pic"
)

// random number seed and source
var randSeed = 9
var randSrc = rand.NewSource(uint64(randSeed))


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
    return mu, utils.Dense2Sym(sigma)
}


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

func AcquisitionMeanCov(X, Y []float64, AllX []float64) (*mat.Dense, *mat.SymDense) {
    return GaussianProcessPrediction(mat.NewDense(len(X), 1, X),
                                           mat.NewDense(len(Y), 1, Y),
                                           mat.NewDense(len(AllX), 1, AllX),
                                           1.0,
                                           1.0,
                                           math.Inf(1),
                                           )
}

func ExpectedImprovement(FStar float64, Mu *mat.Dense, Sigma *mat.SymDense) []float64 {
    N, _ := Mu.Dims()
    expectation := make([]float64, N)
    for i := range expectation {
        mu := Mu.At(i, 0)
        sigma := Sigma.At(i, i)
        normal := distuv.Normal{mu, sigma, randSrc}
        expectation[i] = (FStar - mu) * normal.CDF(FStar) + sigma * normal.Prob(FStar)
    }
    return expectation
}

func IsElem(Item int, X []int) bool {
    if len(X) == 0 {
        return false
    } else {
        return (Item == X[0]) || IsElem(Item, X[1:])
    }
}

func GetNRandomIndices(N, UpperBound int) []int {
    randSrc := rand.NewSource(uint64(randSeed))
    randGen := rand.New(randSrc)
    var indices []int
    for ; len(indices) < N; {
        ind := randGen.Intn(UpperBound)
        if !IsElem(ind, indices) {
            indices = append(indices, ind)
        }
    }
    return indices
}

func Argmin(X []float64) int {
    Min := math.Inf(1)
    var XMin int
    for i := range X {
        if X[i] < Min {
            Min = X[i]
            XMin = i
        }
    }
    return XMin
}

func Argmax(X []float64) int {
    Max := math.Inf(-1)
    var XMax int
    for i := range X {
        if X[i] > Max {
            Max = X[i]
            XMax = i
        }
    }
    return XMax
}

func BOPlot(Alpha, XCoord []float64, Mu *mat.Dense, Sigma *mat.SymDense, X []float64, F func (float64) float64) (*plot.Plot, *plot.Plot) {

    p := plot.New()

    objectiveFunction := make([]float64, len(XCoord))
    mu := mat.Col(nil, 0, Mu)
    upperSigma := make([]float64, len(XCoord))
    lowerSigma := make([]float64, len(XCoord))
    FPrimes := make([]float64, len(X))

    for i := range FPrimes {
        FPrimes[i] = F(X[i])
    }
    xStar := Argmin(FPrimes)
    for i := range XCoord {
        objectiveFunction[i] = F(XCoord[i])
        upperSigma[i] = mu[i] + 9*Sigma.At(i,i)
        lowerSigma[i] = mu[i] - 9*Sigma.At(i,i)
    }

    zigZagX := make([]float64, 2*len(XCoord))
    zigZagY := make([]float64, 2*len(XCoord))
    for i := range zigZagX {
        zigZagX[i] = XCoord[i/2]
        if i % 2 == 0 {
            zigZagY[i] = lowerSigma[i/2]
        } else {
            zigZagY[i] = upperSigma[i/2]
        }
    }

    lZigZag := plt.MakeLineUnicorn(zigZagX, zigZagY, 2.0, 0x0000eeff, nil)
    lUpperSigma := plt.MakeLineUnicorn(XCoord, upperSigma, 2.0, 0x0000eeff, nil)
    lLowerSigma := plt.MakeLineUnicorn(XCoord, lowerSigma, 2.0, 0x0000eeff, nil)
    lAlpha := plt.MakeLineUnicorn(XCoord, Alpha, 2.0, 0x0000eeff, nil)
    lMu := plt.MakeLineUnicorn(XCoord, mu, 3.0, 0x00ee00ff, nil)
    lObjectiveFunction := plt.MakeLineUnicorn(XCoord, objectiveFunction, 2.0, 0xee0000ff, nil)
    sFPrimes := plt.MakeScatterUnicorn(X, FPrimes, plt.CIRCLE_POINT_MARKER, 4.0, plt.DesignedPalette{Type: plt.UNI_PALETTE, Extra: 0x000000ff, Num: len(X)})
    sXStar := plt.MakeScatterUnicorn([]float64{X[xStar]}, []float64{F(X[xStar])}, plt.PYRAMID_POINT_MARKER, 6.0, plt.DesignedPalette{Type: plt.UNI_PALETTE, Extra: 0xff0000ff, Num: len(X)})

    p.Add(lZigZag, lUpperSigma, lLowerSigma, lMu, lObjectiveFunction, sFPrimes, sXStar)

    p.Legend.Add("uncertainty boundaries", lUpperSigma)
    p.Legend.Add("predicted mean", lMu)
    p.Legend.Add("objective function", lObjectiveFunction)
    p.Legend.Add("queried points", sFPrimes)
    p.Legend.Add("current minimum", sXStar)

    p2 := plot.New()
    p2.Add(lAlpha)
    p2.Legend.Add("expected improvement", lAlpha)

    p.X.Min = -3.0
    p.X.Max = 3.0
    p.Y.Min = -5.0
    p.Y.Max = 10.0
    p.X.Label.Text = "x"
    p.Y.Label.Text = "y"
    p.Title.Text = "Steps of Bayesian Optimisation"

    p2.X.Label.Text = "x"
    p2.Y.Label.Text = "y"
    p2.Title.Text = "Expected Improvement"
    return p, p2
}


func BO(F func (float64) float64, AllX []float64, NumRandom, NumIter int) (float64, float64) {
    if NumRandom >= len(AllX) + NumIter { panic("Error: the number of initial random locations is no less than the number of iterations") }

    XStartInds := GetNRandomIndices(NumRandom, len(AllX))
    var X []float64
    var XComp []float64
    for i := range AllX {
        if IsElem(i, XStartInds) {
            X = append(X, AllX[i])
        } else {
            XComp = append(XComp, AllX[i])
        }
    }

    FPrimes := make([]float64, NumRandom)
    for i := range FPrimes {
        FPrimes[i] = F(X[i])
    }
    FStar := floats.Min(FPrimes)
    XStar := X[Argmin(FPrimes)]

        fmt.Println("s", XStar, "\t", FStar)

    gm1 := pic.GifMaker{Width: 500, Height: 500, Delay:150}
    gm2 := pic.GifMaker{Width: 500, Height: 200, Delay:150}
    for i:=0; i<NumIter; i++ {
        mu, sigma := AcquisitionMeanCov(X, FPrimes, AllX)
        alpha := ExpectedImprovement(FStar, mu, sigma)
        p1, p2 := BOPlot(alpha, AllX, mu, sigma, X, F)
        gm1.CollectFrames(p1)
        gm2.CollectFrames(p2)
        XPrime := AllX[Argmax(alpha)]
        NewFValue := F(XPrime)
        fmt.Println(XPrime, "\t", NewFValue)
        FPrimes = append(FPrimes, NewFValue)
        X = append(X, XPrime)
        if NewFValue < FStar {
            FStar = NewFValue
            XStar = XPrime
        }
    }
    gm1.RenderFrames("BO_curves.gif")
    gm2.RenderFrames("BO_expected_improvement.gif")
    return XStar, FStar
}





func main() {
    f := func (x float64) float64 {
        return math.Sin(3.0*x) - x + x*x
    }

    NumPoints := 100
    X := make([]float64, NumPoints)
    for i:=0; i<NumPoints; i++ {
        X[i] = -3.0 + float64(i) / float64(NumPoints) * 6.0
    }

    x, y := BO(f, X, 3, 7)

    fmt.Println("x", x, "y", y)
}



