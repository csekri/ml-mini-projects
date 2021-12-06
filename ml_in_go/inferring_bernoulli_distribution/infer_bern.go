package main

import (
    "gonum.org/v1/gonum/stat/distuv"
    "gonum.org/v1/gonum/floats"
    "golang.org/x/exp/rand"

    "image/color"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// random number seed and source
var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


/*
SUMMARY
    Computes the posterior distribution. We assume the prior has beta distribution
    as well as the posterior.
PARAMETERS
    a float64: parameter of the beta distribution in the prior
    b float64: parameter of the beta distribution in the prior
    X []float64: the data (coin flips of 1s and 0s) we have observed
    linSpace []float64: the x values where we want to plot the probability
        density function of the posterior
RETURN
    []float64: the probability density function of the probability the coin biased towards head
*/
func Posterior(a, b float64, X, linSpace []float64) []float64 {
    XSum := floats.Sum(X)
    a_n := a + XSum
    b_n := b + float64(len(X)) - XSum
    beta := distuv.Beta{a_n, b_n, randSrc}
    pdf := make([]float64, len(linSpace))
    for i, v := range linSpace {
        pdf[i] = beta.Prob(v)
    }
    return pdf
}


/*
SUMMARY
    Converts points to plotter.XYs, that can be used in gonum/plot.
PARAMETERS
    linSpace []float64: x coordinates
    data []float64: y coordinates
RETURN
    plotter.XYs: the points in the type that's useful for gonum/plot
*/
func plotPoints(linSpace, data []float64) plotter.XYs {
    pts := make(plotter.XYs, len(linSpace))
    for i := range pts {
        pts[i].X = linSpace[i]
        pts[i].Y = data[i]
    }
    return pts
}


/*
SUMMARY
    Visualises the probability density function as we observe more and more points.
PARAMETERS
    a float64: parameter of the beta distribution in the prior
    b float64: parameter of the beta distribution in the prior
    priorMu []float64: a slice containing the prior distribution
    linSpace []float64: the x values where we want to plot the probability
        density functions
RETURN
    N/A
*/
func PlotCurveEvolution(a,b float64, priorMu, linSpace, X []float64) {
    p := plot.New()

    p.Title.Text = "Evolution of our model"
    p.Title.TextStyle.Font.Size = 20
    p.X.Label.Text = "x"
    p.Y.Label.Text = "y"
    // Draw a grid behind the data
    p.Add(plotter.NewGrid())

    // Make a line plotter and set its style.
    pts := plotPoints(linSpace, priorMu)
    l, err := plotter.NewLine(pts); if err != nil { panic(err) }
    l.LineStyle.Width = vg.Points(2)
    l.LineStyle.Color = color.RGBA{G: 255, A: 255}
    p.Add(l)
    p.Legend.Add("prior distribution", l)
    for i := range X {
        y := Posterior(a, b, X[:i], linSpace)
        pts := plotPoints(linSpace, y)
        l, err := plotter.NewLine(pts); if err != nil { panic(err) }
        l.LineStyle.Width = vg.Points(0.2)
        l.LineStyle.Color = color.RGBA{R: 255, A: 255}
        p.Add(l)
    }
    // Make a line plotter and set its style.
    y := Posterior(a, b, X, linSpace)
    pts = plotPoints(linSpace, y)
    l, err = plotter.NewLine(pts); if err != nil { panic(err) }
    l.LineStyle.Width = vg.Points(2)
    l.LineStyle.Color = color.RGBA{B: 255, A: 255}
    p.Add(l)
    p.Legend.Add("posterior distribution after seeing all coin flips", l)
    p.Legend.Top = true

    if err := p.Save(6*vg.Inch, 4*vg.Inch, "points.svg"); err != nil {
		panic(err)
	}
}


/*
We generate the coin flips and the x values.
We define the prior and finally plot the updated beliefs.
*/
func main() {
    // parameters to generate data (Bern(x|mu) in N trials) = Binomial(n, mu)
    mu := 0.2
    N := 100

    bernoulli := distuv.Bernoulli{mu, randSrc}

    // generate random data
    X := make([]float64, N)
    for i := range X { X[i] = bernoulli.Rand() }
    linSpaceRes := 300
    linSpace := make([]float64, linSpaceRes)
    for i := range linSpace { linSpace[i] = float64(i) / float64(linSpaceRes) }

    // now let's define the prior
    a := 10.0
    b := 10.0

    beta := distuv.Beta{a, b, randSrc}
    priorMu := make([]float64, len(linSpace))
    for i := range priorMu { priorMu[i] = beta.Prob(linSpace[i]) }
    PlotCurveEvolution(a, b, priorMu, linSpace, X)
}