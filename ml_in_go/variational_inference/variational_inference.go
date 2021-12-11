package main

import (
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
    of neighbours of I in X. When there are less than eight neighbours the sum is normalised.
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
    neighbours := Neighbours(row , col, Height, Width)
    for _, neighbour := range neighbours {
        index := coord2Index(neighbour.Y, neighbour.X)
        miniSum += X[index] * Value
    }
    return miniSum / float64(len(neighbours)) * 8.0
}


/*
SUMMARY
    Returns a large value if X is likely to have generated Y.
PARAMETERS
    X float64: x value
    Y float64: y value
RETURN
    float64: large if |2*Y-1 - X|<epsilon, else small

*/
func L(X, Y float64) float64 {
    return math.Exp(-0.5*math.Pow(2*Y-1 - X, 2))
}

/*
SUMMARY
    Computes the Variational Bayes for Ising Model on binary images.
PARAMETERS
    Img *mat.Dense: one channel from an image (in BW images all channels are the same)
    Periods int: the number of iterations
RETURN
    *mat.Dense: one channel of the ICM restored image
*/
func VariationalBayes(Img *mat.Dense, Periods int) *mat.Dense {
    Height, Width := Img.Dims()
    y := utils.Flatten(Img, true)
    x := utils.Linspace(1.0, 1.0, len(y))
    mu := utils.Linspace(0.0, 0.0, len(y))
    m := 0.0
    for tau:=0; tau<Periods; tau++ {
        for i := range x {
            m = NeighbourSum(i, 1.0, mu, Width)
            mu[i] = math.Tanh(m + 7.0 * (L(1.0, y[i]) - L(-1.0, y[i])))
        }
    }
    img := mat.NewDense(Height, Width, mu)
    return img
}


/*
Next we try to restore a noisy image with ICM on Ising Model and animate the effort.
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
    for i:=1; i<6; i++ {
        rand.Seed(6)
        channel := VariationalBayes(matrix, i)
        channel.Apply(func (j, i int, v float64) float64 { return (v+1)*127.5 }, channel)
        img[0] = *mat.NewDense(Height, Width, nil)
        img[1] = *mat.NewDense(Height, Width, nil)
        img[2] = *mat.NewDense(Height, Width, nil)
        img[0].Copy(channel)
        img[1].Copy(channel)
        img[2].Copy(channel)
        gm.CollectImages(img.ToImage())
    }
    gm.RenderFrames("scottie_vb.gif")
}
