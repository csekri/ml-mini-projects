package main

import (
    "fmt"
    "math"
    "golang.org/x/exp/rand"

    "gonum.org/v1/gonum/mat"

    "ml_playground/pic"
    "ml_playground/utils"
)


// random number seed and source
var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


// this type stores a pair of coordinate
type Coord struct {
    Y int
    X int
}


/*
SUMMARY
    Returns the set of neighbours of a pixel in all 8 directions.
PARAMETERS
    j int: row of the pixel
    i int: col of the pixel
    Height int: the height of the image
    Width int: the width of the image
RETURN
    []Coord: set of coordinates (pair of indices) of all neighbours
*/
func Neighbours(j, i int, Height, Width int) []Coord {
    if j == 0 && i == 0 {
        return []Coord{Coord{1,0},
                       Coord{1,1},
                       Coord{0,1}}
    } else if j == Height-1 && i == Width-1 {
        return []Coord{Coord{Height-2, Width-1},
                       Coord{Height-2, Width-2},
                       Coord{Height-1, Width-2}}
    } else if j == Height-1 && i == 0 {
        return []Coord{Coord{Height-2, 0},
                       Coord{Height-2, 1},
                       Coord{Height-1, 1}}
    } else if j == 0 && i == Width - 1 {
        return []Coord{Coord{0, Width-2},
                       Coord{1, Width-2},
                       Coord{1, Width-1}}
    } else if j == Height-1 {
        return []Coord{Coord{Height-1, i+1},
                       Coord{Height-1, i-1},
                       Coord{Height-2, i},
                       Coord{Height-2, i-1},
                       Coord{Height-2, i+1}}
    } else if j == 0 {
        return []Coord{Coord{0, i+1},
                       Coord{0, i-1},
                       Coord{1, i},
                       Coord{1, i-1},
                       Coord{1, i+1}}
    } else if i == Width-1 {
        return []Coord{Coord{j-1, Width-1},
                       Coord{j+1, Width-1},
                       Coord{j, Width-2},
                       Coord{j-1, Width-2},
                       Coord{j+1, Width-2}}
    } else if i == 0 {
        return []Coord{Coord{j-1, 0},
                       Coord{j+1, 0},
                       Coord{j, 1},
                       Coord{j-1, 1},
                       Coord{j+1, 1}}
    } else {
        return []Coord{Coord{j, i+1},
                       Coord{j, i-1},
                       Coord{j-1, i},
                       Coord{j-1, i-1},
                       Coord{j-1, i+1},
                       Coord{j+1, i},
                       Coord{j+1, i-1},
                       Coord{j+1, i+1}}
    }
}


/*
SUMMARY
    Computes the sum: `\sum_{j\in\mathcal{N}(I)} Value\cdot X_j` where `\mathcal{N}(I)` is the set
    of neighbours of I in X.
PARAMETERS
    j int: row of the pixel
    i int: col of the pixel
    Width int: the width of the image
RETURN
    []Coord: set of coordinates (pair of indices) of all neighbours
*/
func NeighbourSum(I int, Value float64, X []float64, Width int) float64 {
    Height := len(X) / Width
    coord2Index := func (y, x int) int {
        return y*Width + x
    }
    index2Coord := func (index int) (int, int) {
        return index / Width, index % Width
    }
    miniSum := 0.0
    row, col := index2Coord(I)
    for _, neighbour := range Neighbours(row , col, Height, Width) {
        index := coord2Index(neighbour.Y, neighbour.X)
        miniSum += X[index] * Value
    }
    return miniSum
}


/*
SUMMARY
    Computes the ICM c*log(p(X|Y)) for the a given configuration. This is used only in the initial setup as a trick
    because it is very costly to compute this for every pixel.
PARAMETERS
    X []float64: the latent space of -1s and +1s
    Y []float64: the corrupted image
RETURN
    float64: the likelihood
*/
func ICMComputeInitialProd(X, Y []float64) float64 {
    prod := 0.0
    for i := range X {
        prod += math.Exp(-100.0*math.Pow(2*Y[i]-1 - X[i], 2.0))
    }
    return prod
}


/*
SUMMARY
    Computes the ICM c*log(p(X)) for the a given configuration. This is used only in the initial setup as a trick
    because it is very costly to compute this for every pixel.
PARAMETERS
    X []float64: the latent space of -1s and +1s
    Width int: the width of the image
RETURN
    float64: the likelihood
*/
func ICMComputeInitialNeighbourWeight(X []float64, Width int) float64 {
    sum := 0.0
    for i := range X {
        sum += NeighbourSum(i, X[i], X, Width)
    }
    return sum
}


/*
SUMMARY
    Computes the ICM p(X_I=OnOff, X, Y). With a simple trick we don't recompute p(X) and p(X|Y) but only
    alter the Ith factor.
PARAMETERS
    I int: the index of the pixel
    OnOff: +1 or -1 we test at the Ith index
    X []float64: the latent space of -1s and +1s
    Y []float64: the corrupted image
    Width int: the width of the image
    reUseProd float64: p(X|Y)
    reUseNeighbour float64: p(X)
RETURN
    float64: the likelihood
*/
func IsingJointProbability(I int, OnOff float64, X, Y []float64, Width int, reUseProd, reUseNeighbour float64) float64 {
    prod := reUseProd - math.Exp(-100.0*math.Pow(2*Y[I]-1 - X[I], 2.0)) + math.Exp(-100.0*math.Pow(2*Y[I]-1 - OnOff, 2.0))
    neighbourFactor := reUseNeighbour - NeighbourSum(I, X[I], X, Width) + NeighbourSum(I, OnOff, X, Width)
    return prod + 0.001*(neighbourFactor)
}


/*
SUMMARY
    Computes the Iterative Conditional Modes for Ising Model on binary images.
PARAMETERS
    Img *mat.Dense: one channel from an image (in BW images all channels are the same)
    Periods int: the number of iterations
RETURN
    *mat.Dense: one channel of the ICM restored image
*/
func ICM(Img *mat.Dense, Periods int) *mat.Dense {
    Height, Width := Img.Dims()
    y := utils.Flatten(Img, true)
    x := utils.Linspace(1.0, 1.0, len(y))
    prod := ICMComputeInitialProd(x, y)
    neighbourFactor := ICMComputeInitialNeighbourWeight(x, Width)
    for tau:=0; tau<Periods; tau++ {
        for i := range x {
            left := IsingJointProbability(i, 1.0, x, y, Width, prod, neighbourFactor)
            right := IsingJointProbability(i, -1.0, x, y, Width, prod, neighbourFactor)
            oldXValue := x[i]
            if left > right {
                x[i] = 1.0
            } else {
                x[i] = -1.0
            }
            prod = prod - math.Exp(-100.0*math.Pow(2*y[i]-1 - oldXValue, 2.0)) + math.Exp(-100.0*math.Pow(2*y[i]-1 - x[i], 2.0))
            neighbourFactor = neighbourFactor - NeighbourSum(i, oldXValue, x, Width) + NeighbourSum(i, x[i], x, Width)
        }
    }
    return mat.NewDense(Height, Width, x)
}


/*
SUMMARY
    Computes the Gibbs Sampling posterior p(Y|X).
PARAMETERS
    I int: the Index
    X []float64: the latent space of -1s and +1s
    Y []float64: the corrupted image
    Width int: the width of the image
RETURN
    *mat.Dense: one channel of the ICM restored image
*/
func GibbsPosterior(I int, X, Y []float64, Width int) float64 {
    numerator := math.Exp(-10.0*math.Pow(2*Y[I]-1 - 1.0, 2.0)) * math.Exp(NeighbourSum(I, 1.0, X, Width))
    denominator := numerator + math.Exp(-10.0*math.Pow(2*Y[I]-1 - (-1.0), 2.0)) * math.Exp(NeighbourSum(I, -1.0, X, Width))
    prob := numerator / denominator
    return prob
}


/*
SUMMARY
    Computes the Gibbs Sampling for Ising Model on binary images.
PARAMETERS
    Img *mat.Dense: one channel from an image (in BW images all channels are the same)
    Periods int: the number of iterations
RETURN
    *mat.Dense: one channel of the ICM restored image
*/
func GibbsSampling(Img *mat.Dense, Periods int) *mat.Dense {
    Height, Width := Img.Dims()
    y := utils.Flatten(Img, true)
    x := utils.Linspace(1.0, 1.0, len(y))
    for tau:=0; tau<Periods; tau++ {
        for i := range x {
            prob := GibbsPosterior(i, x, y, Width)
            t := rand.Float64()
            if prob > t {
                x[i] = 1.0
            } else {
                x[i] = -1.0
            }
        }
    }
    return mat.NewDense(Height, Width, x)
}


/*
Next we try to restore a noisy image with ICM on Ising Model and animate the effort.
Next we try to restore it with Gibbs Sampling on Ising Model and animate the effort.
*/
func main() {
    var img pic.RGBImg = make([]mat.Dense, 3)
    err := img.LoadPixels("noisy_scottie.jpg")
    if err != nil { panic(err) }

    Height, Width := img[0].Dims()
    img.Apply(func (j, i int, v float64) float64 { return v/255.0 })
    matrix := mat.NewDense(Height, Width, nil)
    matrix.Copy(&img[0])
    gm := pic.GifMaker{ Delay: 100 }
    fmt.Println("Processing Gibbs animation")
    for i:=1; i<6; i++ {
        rand.Seed(6)
        channel := GibbsSampling(matrix, i)
        channel.Apply(func (j, i int, v float64) float64 { return (v+1)*127.5 }, channel)
        img[0] = *mat.NewDense(Height, Width, nil)
        img[1] = *mat.NewDense(Height, Width, nil)
        img[2] = *mat.NewDense(Height, Width, nil)
        img[0].Copy(channel)
        img[1].Copy(channel)
        img[2].Copy(channel)
        gm.CollectImages(img.ToImage())
    }
    gm.RenderFrames("scottie_gibbs.gif")
    fmt.Println("Gibbs animation concluded\nProcessing ICM animation")
    gm = pic.GifMaker{ Delay: 100 }
    for i:=1; i<6; i++ {
        channel := ICM(matrix, i)
        channel.Apply(func (j, i int, v float64) float64 { return (v+1)*127.5 }, channel)
        img[0] = *mat.NewDense(Height, Width, nil)
        img[1] = *mat.NewDense(Height, Width, nil)
        img[2] = *mat.NewDense(Height, Width, nil)
        img[0].Copy(channel)
        img[1].Copy(channel)
        img[2].Copy(channel)
        gm.CollectImages(img.ToImage())
    }
    gm.RenderFrames("scottie_icm.gif")
    fmt.Println("ICM animation concluded")
}
