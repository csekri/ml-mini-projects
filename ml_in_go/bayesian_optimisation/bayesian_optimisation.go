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
    "ml_playground/gaussian_processes"
)

// random number seed and source
var randSeed = 9
var randSrc = rand.NewSource(uint64(randSeed))


/*
SUMMARY
    Computes the mean and covariance of the acquisition function.
PARAMETERS
    X []float64: column vector of seen locations
    Y []float64: column vector of seen labels
    AllX []float64: contains all locations of interest
RETURN
    *mat.Dense: column vector of the mean
    *mat.SymDense: square symmetric matrix of the covariance
*/
func AcquisitionMeanCov(X, Y []float64, AllX []float64) (*mat.Dense, *mat.SymDense) {
    return gaussian_processes.GaussianProcessPrediction(mat.NewDense(len(X), 1, X),
                                           mat.NewDense(len(Y), 1, Y),
                                           mat.NewDense(len(AllX), 1, AllX),
                                           1.0,
                                           1.0,
                                           math.Inf(1),
                                           )
}


/*
SUMMARY
    Computes the mean and covariance of the acquisition function.
PARAMETERS
    FStar float64: the current minimum
    Mu *mat.Dense: column vector of the mean
    Sigma *mat.SymDense: square symmetric matrix of the covariance
RETURN
    []float64: the expected improvement for all locations
    *mat.SymDense: square symmetric matrix of the covariance
*/
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


/*
SUMMARY
    Generates several positive random integers.
PARAMETERS
    N int: the number of random numbers
    UpperBound int: the upper limit for each random number
RETURN
    []int: a slice of these integers
*/
func GetNRandomIndices(N, UpperBound int) []int {
    randSrc := rand.NewSource(uint64(randSeed))
    randGen := rand.New(randSrc)
    var indices []int
    for ; len(indices) < N; {
        ind := randGen.Intn(UpperBound)
        if !utils.IsElem(ind, indices) {
            indices = append(indices, ind)
        }
    }
    return indices
}


/*
SUMMARY
    Generates two plots: one for the state of BO, the other for the expected improvement.
PARAMETERS
    Alpha []float64: expected improvement
    XCoord []float64: the set of x coordinates we search the minimum in
    Mu *mat.Dense: the expectation for the objective function
    X []float64: the queried locations
    F func (float64) float64: the objective function
RETURN
    *plot.Plot: the state of the optimisation
    *plot.Plot: the expected improvement corresponding to the state
*/
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
    xStar := utils.Argmin(FPrimes)
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


/*
SUMMARY
    Performs the Bayesian Optimisation. The objective function is called
    exactly `NumRandom+NumIter` number of times. In the meantime it generates
    an animation (2 .gif files) of the optimisation.
PARAMETERS
    F func (float64) float64: objective function
    AllX []float64: all the x coordinates
    NumRandom int: the number of randomly chosen locations BO starts with
    NumIter int: the number of iterations
RETURN
    float64: the minimum location
    float64: the function value at the minimum
*/
func BO(F func (float64) float64, AllX []float64, NumRandom, NumIter int) (float64, float64) {
    if NumRandom >= len(AllX) + NumIter { panic("Error: the number of initial random locations is no less than the number of iterations") }
    XStartInds := GetNRandomIndices(NumRandom, len(AllX))
    var X []float64
    for i := range AllX {
        if utils.IsElem(i, XStartInds) {
            X = append(X, AllX[i])
        }
    }
    FPrimes := make([]float64, NumRandom)
    for i := range FPrimes {
        FPrimes[i] = F(X[i])
    }
    FStar := floats.Min(FPrimes)
    XStar := X[utils.Argmin(FPrimes)]
    gm1 := pic.GifMaker{Width: 500, Height: 500, Delay:150}
    gm2 := pic.GifMaker{Width: 500, Height: 200, Delay:150}
    for i:=0; i<NumIter; i++ {
        mu, sigma := AcquisitionMeanCov(X, FPrimes, AllX)
        alpha := ExpectedImprovement(FStar, mu, sigma)
        p1, p2 := BOPlot(alpha, AllX, mu, sigma, X, F)
        gm1.CollectFrames(p1)
        gm2.CollectFrames(p2)
        XPrime := AllX[utils.Argmax(alpha)]
        NewFValue := F(XPrime)
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


// We BO optimise F(x)=sin(3x)-x+x^2 and animate the process.
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



