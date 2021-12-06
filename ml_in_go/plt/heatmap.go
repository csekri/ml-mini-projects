package plt

import (
    "image"
    "gonum.org/v1/plot/palette"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/gonum/mat"
)


// this type will be useful for implementing the GridXYZ interface
type Range struct {
    Min float64
    Max float64
}

/*
this type will implement the GridXYZ interface,
when a function and range passed it will create a heatmap/contour
plot of that function within the domain defined by XRange and YRange
*/
type FuncHeatMap struct {
    Function func (x, y float64) float64
    Height int
    Width int
    XRange Range
    YRange Range
}

/*
defines the type for creating heatmap/contour plot for a matrix
*/
type MatrixHeatMap struct {
    Matrix *mat.Dense
    XRange Range
    YRange Range
}


/*
SUMMARY
    Interface function for MatrixHeatMap
PARAMETERS
    N/A
RETURN
    int: height in pixels
    int width in pixels
*/
func (f *MatrixHeatMap) Dims() (int, int) {
    H, W := f.Matrix.Dims()
    return H, W
}


/*
SUMMARY
    Interface function for MatrixHeatMap
PARAMETERS
    c int: column
RETURN
    float64: x coordinate for the c column in the matrix
*/
func (f *MatrixHeatMap) X(c int) float64 {
    _, W := f.Matrix.Dims()
    return f.XRange.Min + float64(c) / float64(W) * (f.XRange.Max-f.XRange.Min)
}


/*
SUMMARY
    Interface function for MatrixHeatMap
PARAMETERS
    r int: row
RETURN
    float64: y coordinate for the r row in the matrix
*/
func (f *MatrixHeatMap) Y(r int) float64 {
    H, _ := f.Matrix.Dims()
    return f.YRange.Min + float64(r) / float64(H) * ((f.YRange.Max-f.YRange.Min))
}


/*
SUMMARY
    Interface function for MatrixHeatMap
PARAMETERS
    c int: column
    r int: row
RETURN
    float64: z coordinate r row and c column in the matrix
*/
func (f *MatrixHeatMap) Z(c, r int) float64 {
    _, M := f.Matrix.Dims()
    return f.Matrix.At(c, M-r-1)
}


/*
SUMMARY
    Interface function for FuncHeatMap
PARAMETERS
    N/A
RETURN
    int: height in pixels
    int width in pixels
*/
func (f *FuncHeatMap) Dims() (int, int) {
    return f.Height, f.Width
}


/*
SUMMARY
    Interface function for FuncHeatMap
PARAMETERS
    c int: column
RETURN
    float64: x coordinate for the c column in the figure
*/
func (f *FuncHeatMap) X(c int) float64 {
    return f.XRange.Min + float64(c) / float64(f.Width) * (f.XRange.Max-f.XRange.Min)
}


/*
SUMMARY
    Interface function for FuncHeatMap
PARAMETERS
    r int: row
RETURN
    float64: y coordinate for the r row in the figure
*/
func (f *FuncHeatMap) Y(r int) float64 {
    return f.YRange.Min + float64(r) / float64(f.Height) * ((f.YRange.Max-f.YRange.Min))
}


/*
SUMMARY
    Interface function for MatrixHeatMap
PARAMETERS
    c int: column
    r int: row
RETURN
    float64: z coordinate y coordinate at r row and x coordinate at c column in the matrix
*/
func (f *FuncHeatMap) Z(c, r int) float64 {
//     _, M := f.Dims()
    return f.Function(f.X(c), f.Y(r))
}


/*
SUMMARY
    Rasterises the heatmap plot, there are issues (gridlines) when this is not done.
PARAMETERS
    data plotter.GridXYZ: contains the data about the heatmap
    pal palette.Palette: contains the colormap the paint the heatmap
RETURN
    *image.RGBA64: the resulting raster image with the heatmap
*/
func FillImage (data plotter.GridXYZ, pal palette.Palette) *image.RGBA64 {
    n, m := data.Dims()
    img := image.NewRGBA64(image.Rectangle{
        Min: image.Point{X: 0, Y: 0},
        Max: image.Point{X: n, Y: m},
    })
    colors := pal.Colors()

    max := data.Z(0, 0)
    min := data.Z(0, 0)
    for i := 0; i < n; i++ {
        for j := 0; j < m; j++ {
            if data.Z(i, j) > max {
            max = data.Z(i, j)
            }

            if data.Z(i, j) < min {
                min = data.Z(i, j)
            }
        }
    }

    for i := 0; i < n; i++ {
        for j := 0; j < m; j++ {
            v := data.Z(i, j)
            colorIdx := int((v - min) * float64(len(colors)-1) / (max - min))
            img.Set(i, n-j, colors[colorIdx])
        }
    }
    return img
}
