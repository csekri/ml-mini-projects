package main


import (
    "fmt"
    "image/color"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/stat/distuv"

    "gonum.org/v1/plot"

    "ml_playground/pic"
    "ml_playground/plt"
    "ml_playground/utils"
)

/*
SUMMARY
    Compares two slices of slices. Return true is they are equal,
    otherwise returns false.
PARAMETERS
    a [][]float64: the first slice of slices
    b [][]float64: the second slice of slices
RETURN
    bool: true if a==b else false
*/
func Equal2dSlice(a,b [][]float64) bool {
    for i := range a {
        if !floats.Equal(a[i], b[i]) {
            return false
        }
    }
    return true
}


/*
SUMMARY
    Creates and populates an M by N matrix with uniform random numbers
PARAMETERS
    N int: # of rows
    M int: # of columns
    seed int: seed for the random number generator
    rangeLow float64: the lower bound of the random number range
    rangeHigh float64: the upper bound of the random number range
RETURN
    *mat.Dense: the random matrix
*/
func CreateRandomPoints(N, M, seed int, rangeLow, rangeHigh float64) *mat.Dense {
    uniform := distuv.Uniform{rangeLow, rangeHigh, rand.NewSource(uint64(seed))}
    pts := mat.NewDense(M, N, nil)
    for y:=0; y<M; y++ {
        for x:=0; x<N; x++ {
            pts.Set(y,x, uniform.Rand())
        }
    }
    return pts
}


/*
SUMMARY
    For each point in a set of points, computes which centre is the closest and labels
    the point with the index of the centre.
PARAMETERS
    points *mat.Dense: the points we would like to find a centre
    centres *mat.Dense: the centres in the KMeans algorithm
RETURN
    []int: the labels for each point
*/
func LabelPairwiseDistances(points *mat.Dense, centres [][]float64) []int {
    dims, n := points.Dims()
    var labels []int = make([]int, n)
    var distances []float64 = make([]float64, len(centres))
    var distance_idx []int = make([]int, len(centres))
    var point []float64 = make([]float64, dims)
    for i:=0; i<n; i++ {
        point = mat.Col(nil, i, points)
        for j, centre := range centres {
            distances[j] = floats.Distance(centre, point, 2)
        }
        floats.Argsort(distances, distance_idx)
        labels[i] = distance_idx[0]
    }
    return labels
}


/*
SUMMARY
    Implements the KMeans unsupervised algorithm.
PARAMETERS
    points *mat.Dense: the points we would like to find a cluster
    numClasses int: the number of clusters we would like to find
RETURN
    []int: the labels for each point
    [][]float64: the centres KMeans converged into
*/
func KMeansClassify(points *mat.Dense, numClasses int) ([]int, [][]float64) {
    dims, n := points.Dims()
    uniform := distuv.Uniform{0, float64(n), rand.NewSource(uint64(69))}
    var centres, oldCentres [][]float64
    for i:=0; i<numClasses; i++ {
        centres = append(centres, mat.Col(nil, int(uniform.Rand()), points))
        oldCentres = append(oldCentres, make([]float64, dims))
    }
    labels := make([]int, n)
    steps := 0
    for !Equal2dSlice(centres, oldCentres) {
        steps++
        fmt.Println(steps)
        for idxToCopy := range centres {
            copy(oldCentres[idxToCopy], centres[idxToCopy])
        }
        labels = LabelPairwiseDistances(points, centres)
        for i:=0; i<numClasses; i++ {
            classSize := 0
            for j:=0; j<n; j++ {
                if labels[j] == i {
                    floats.Add(centres[i], mat.Col(nil, j, points))
                    classSize++
                }
            }
            if classSize > 0 {
                for k, c := range centres[i] {
                    centres[i][k] = c / float64(classSize)
                }
            }
        }
    }
    return labels, centres
}


/*
SUMMARY
    Applies KMeans for image segmentation. Note that this algorithm is quite slow.
PARAMETERS
    img pic.RGBImg: the input image we would like to segment;
        the output is saved into this variable
    numClasses int: the number of colours after segmentation
RETURN
    N/A
*/
func SegmentImage(img pic.RGBImg, numClasses int) {
    height, width := img[0].Dims()
    r_flat := utils.Flatten(&img[0], true)
    g_flat := utils.Flatten(&img[1], true)
    b_flat := utils.Flatten(&img[2], true)

    points := mat.NewDense(3, width*height, nil)
    points.SetRow(0, r_flat)
    points.SetRow(1, g_flat)
    points.SetRow(2, b_flat)
    labels, centres := KMeansClassify(points, numClasses)

    for i, label := range labels {
        img[0].Set(i/width, i%width, centres[label][0])
        img[1].Set(i/width, i%width, centres[label][1])
        img[2].Set(i/width, i%width, centres[label][2])
    }
}


/*
SUMMARY
    Computes the maximum in an integer slice.
PARAMETERS
    X []int: input slice
RETURN
    int: the minimum value
*/
func MaxInt(X []int) int {
    max := -1
    for i := range X {
        if X[i] > max { max = X[i] }
    }
    return max
}


/*
SUMMARY
    Creates a plot for the KMeans result in 2d. Each cluster is assigned a random colour.
PARAMETERS
    xs []float64: x coordinates
    ys []float64: y coordinates
    cs []int: labels
RETURN
    *plot.Plot: the resulting plot
*/
func KMeansPlot(xs, ys []float64, cs []int) *plot.Plot {
    numColours := MaxInt(cs) + 1
    classColours := plt.DesignedPalette{Type: plt.RANDOM_PALETTE, Num: numColours, Extra: 5}.Colors()
    customColours := make([]color.Color, len(cs))
    for i, label := range cs {
        customColours[i] = classColours[label]
    }
    pal := plt.CustomPalette{customColours}
    scatter := plt.MakeScatterUnicorn(xs, ys, plt.CIRCLE_POINT_MARKER, 4.0, pal)
    p := plot.New()
    p.Add(scatter)
    return p
}


/*
We create random points, apply KMeans and visualise it.
We also segment an image with KMeans and save the result.
*/
func main() {
    points := CreateRandomPoints(300, 2, 6, 0, 255)
    cs, _ := KMeansClassify(points, 10)
    xs := mat.Row(nil, 0, points)
    ys := mat.Row(nil, 1, points)
    p := KMeansPlot(xs, ys, cs)
    p.Title.Text, p.X.Label.Text, p.Y.Label.Text = "Kmeans Scatter Plot", "x", "Y"
    p.Save(300, 200, "kmeans.svg")

    var img pic.RGBImg = make([]mat.Dense, 3)
    img.LoadPixels("image.jpg")
    SegmentImage(img, 10)
    img.SaveImage("image_segmented.jpg")
}
