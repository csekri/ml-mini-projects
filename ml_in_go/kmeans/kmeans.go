package main


import (
    "fmt"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
    "golang.org/x/exp/rand"
    "gonum.org/v1/gonum/stat/distuv"

    "ml_playground/pic"
    "ml_playground/plt"
    "ml_playground/utils"
)

func Equal2dSlice(a,b [][]float64) bool {
    for i := range a {
        if !floats.Equal(a[i], b[i]) {
            return false
        }
    }
    return true
}

// n: number of points, dims: number of dimensions the points are embedded in
// rangeLow, rangeHigh : random number limits
func CreateRandomPoints(n, dims, seed int, rangeLow, rangeHigh float64) *mat.Dense {
    uniform := distuv.Uniform{rangeLow, rangeHigh, rand.NewSource(uint64(seed))}
    pts := mat.NewDense(dims, n, nil)
    for y:=0; y<dims; y++ {
        for x:=0; x<n; x++ {
            pts.Set(y,x, uniform.Rand())
        }
    }
    return pts
}

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

func SegmentImage(img pic.RGBImg, numClasses int) {
    height, width := img[0].Dims()
    r_flat := utils.Flatten(&img[0])
    g_flat := utils.Flatten(&img[1])
    b_flat := utils.Flatten(&img[2])

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


func main() {
//     uniform := distuv.Uniform{0, 10, rand.NewSource(10)}
//     var xs, ys, cs []float64
//     for i:=0; i<50; i++ {
//         xs = append(xs, uniform.Rand())
//         ys = append(ys, uniform.Rand())
//         cs = append(cs, uniform.Rand())
//     }
//     points := mat.NewDense(2, 1000, nil)
//     dims, n := points.Dims()
//     for y:=0; y<dims; y++ {
//         for x:=0; x<n; x++ {
//             points.Set(y,x, uniform.Rand())
//         }
//     }
    points := CreateRandomPoints(300, 2, 6, 0, 255)
    cs, _ := KMeansClassify(points, 10)
    xs := mat.Row(nil, 0, points)
    ys := mat.Row(nil, 1, points)
    plt.ScatterPlotWithLabels(xs, ys, cs, "10cm", "7cm", "Kmeans Scatter Plot", "kmeans.svg")

    var img pic.RGBImg = make([]mat.Dense, 3)
    img.LoadPixels("image.jpg")
    SegmentImage(img, 15)
    img.SaveImage("image_segmented.jpg")



    





}
