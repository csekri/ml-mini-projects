/*
This library contains a few functions and methods
which helps with image IO and manipulation.
*/
package pic

import (
    "math"
    "image"
    "image/jpeg"
    "image/color"
    "os"
    "golang.org/x/exp/rand"

    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat/distuv"
)

// This type is used to manipulate the pixels of an image, no alpha channel is present
type RGBImg []mat.Dense


// random number seed and source
var randSeed = 10
var randSrc = rand.NewSource(uint64(randSeed))


/*
SUMMARY
    Loads an image from the disk into a matrix of pixels.
PARAMETERS
    Filename string: path of the image
RETURN
    error: error whether the method successfully terminates
*/
func (out RGBImg) LoadPixels(Filename string) error {
    image.RegisterFormat("jpeg", "\xff\xd8", jpeg.Decode, jpeg.DecodeConfig)
    file, err := os.Open(Filename)
    if err != nil { panic(err) }
    defer file.Close()
    img, _, err := image.Decode(file)
    if err != nil { panic(err) }

    bounds := img.Bounds()
    width, height := bounds.Max.X, bounds.Max.Y
    rgbImg := make([][]float64, 3)
    rgbImg[0] = make([]float64, width*height)
    rgbImg[1] = make([]float64, width*height)
    rgbImg[2] = make([]float64, width*height)
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            r, g, b, _ := img.At(x, y).RGBA()
            rgbImg[0][x + y*width] = float64(r / 257)
            rgbImg[1][x + y*width] = float64(g / 257)
            rgbImg[2][x + y*width] = float64(b / 257)
        }
    }
    out[0] = *mat.NewDense(height, width, rgbImg[0])
    out[1] = *mat.NewDense(height, width, rgbImg[1])
    out[2] = *mat.NewDense(height, width, rgbImg[2])
    return err
}


/*
SUMMARY
    Converts an image into Image.Image in the Go standard library.
PARAMETERS
    N/A
RETURN
    image.Image: result of the conversion
*/
func (in RGBImg) ToImage() image.Image {
    height, width := in[0].Dims()
    pic := image.NewRGBA(image.Rect(0, 0, width, height))
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            pic.Set(x, y, color.RGBA{uint8(in[0].At(y,x)), uint8(in[1].At(y,x)), uint8(in[2].At(y,x)), 255})
        }
    }
    return pic
}


/*
SUMMARY
    Save the image in a file.
PARAMETERS
    filename string: the name of the file we create to save the image into
RETURN
    N/A
*/
func (in RGBImg) SaveImage(Filename string) {
    pic := in.ToImage()
    f, err := os.Create(Filename)
    if err != nil { panic(err) }
    defer f.Close()
    // Encode to `PNG` with level 98 then save to file
    err = jpeg.Encode(f, pic, &jpeg.Options{98})
    if err != nil { panic(err) }
}


/*
SUMMARY
    Let's say we'd like to compresses the color space (255 colour) down to a specified number.
    Given a colour value this function returns the value in the new colour space
PARAMETERS
    ColourValue float64: number between 0.0 and 255.0
    NumColours int: number between 2 and 255
RETURN
    float64: the colour value in the new color space
*/
func DitherColorMap(ColourValue float64, NumColours int) float64 {
    divideFactor := 256 / NumColours
    multiplyFactor := 255 / (NumColours-1)
    return float64(int(ColourValue) / divideFactor) * float64(multiplyFactor)
}

/*
SUMMARY
    Converts an image to grayscale.
PARAMETERS
    N/A
RETURN
    N/A
*/
func (out RGBImg) GrayScale() {
    height, width := out[0].Dims()
    tmpImg := mat.NewDense(height, width, nil)
    tmpImg.Copy(&out[0])
    tmpImg.Add(tmpImg, &out[1])
    tmpImg.Add(tmpImg, &out[2])

    tmpImg.Apply(func (i, j int, v float64) float64 {return v / 3}, tmpImg)
    out[0] = *mat.NewDense(height, width, nil)
    out[1] = *mat.NewDense(height, width, nil)
    out[2] = *mat.NewDense(height, width, nil)
    out[0].Copy(tmpImg)
    out[1].Copy(tmpImg)
    out[2].Copy(tmpImg)
}


/*
SUMMARY
    Converts an image to binary by thresholding.
PARAMETERS
    Threshold float64: percent i.e. if colour brightness is above this the colour will be white o.w. black
RETURN
    N/A
*/
func (out RGBImg) BinaryThreshold(Threshold float64) {
    height, width := out[0].Dims()
    tmpImg := mat.NewDense(height, width, nil)
    tmpImg.Copy(&out[0])
    tmpImg.Add(tmpImg, &out[1])
    tmpImg.Add(tmpImg, &out[2])

    tmpImg.Apply(
        func (i, j int, v float64) float64 {
            if v / 3.0 > 255.0 * Threshold / 100.0 {
                return 255.0
            }
            return 0.0
        }, tmpImg)
    out[0] = *mat.NewDense(height, width, nil)
    out[1] = *mat.NewDense(height, width, nil)
    out[2] = *mat.NewDense(height, width, nil)
    out[0].Copy(tmpImg)
    out[1].Copy(tmpImg)
    out[2].Copy(tmpImg)
}


/*
SUMMARY
    Applies convolution to a matrix with a given kernel.
PARAMETERS
    matrix *mat.Dense: the matrix, the result is written in this variable
    kernel *mat.Dense: an N by N matrix with where N is odd
RETURN
    N/A
*/
func Convolution(matrix, kernel *mat.Dense) {
    H, W := matrix.Dims()
    HK, WK := kernel.Dims()
    HK, WK = HK / 2, WK / 2
    out := mat.NewDense(H + 2*HK, W + 2*WK, nil)
    for y_:=0; y_ < H+2*HK; y_++ {  // this commented part may be useful for custom padding
        for x_:=0; x_ < W+2*WK; x_++ {
            y, x := y_ - HK, x_ - WK
            if (y < 0 || y >= H) || (x < 0 || x >= W) {
                out.Set(y_, x_, 0.0)
            } else {
                out.Set(y_, x_, matrix.At(y, x))
                matrix.Set(y, x, 0.0)
            }
        }
    }
    for y:=0; y < H; y++ {
        for x:=0; x < W; x++ {
            for j:=-HK; j<=HK; j++ {
                for i:=-WK; i<=WK; i++ {
                    k := kernel.At(j+HK, i+WK)
                    matrix.Set(y, x, matrix.At(y, x) + out.At(HK+y+j, WK+x+i)*k)
                }
            }
        }
    }
}


func SobelKernelX() *mat.Dense {
    return mat.NewDense(3, 3, []float64{1.0,2.0,1.0, 0.0,0.0,0.0, -1.0,-2.0,-1.0})
}

func SobelKernelY() *mat.Dense {
    return mat.NewDense(3, 3, []float64{1.0,0.0,-1.0, 2.0,0.0,-2.0, 1.0,0.0,-1.0})
}


/*
SUMMARY
    Converts an image to binary by thresholding.
PARAMETERS
    Threshold float64: percent i.e. if colour brightness is above this the colour will be white o.w. black
RETURN
    N/A
*/
func (out RGBImg) SobelEdgeDetection() {
    height, width := out[0].Dims()
    out.GrayScale()
    imgX := mat.NewDense(height, width, nil)
    imgY := mat.NewDense(height, width, nil)
    imgX.Copy(&out[0])
    imgY.Copy(&out[1])
    Convolution(imgX, SobelKernelX())
    Convolution(imgY, SobelKernelY())
    img := mat.NewDense(height, width, nil)
    for y:=0; y < height; y++ {
        for x:=0; x < width; x++ {
            img.Set(y, x, math.Sqrt(math.Pow(imgX.At(y, x), 2.0) + math.Pow(imgY.At(y, x), 2.0)))
        }
    }
    minX := mat.Min(img)
    maxX := mat.Max(img)

    img.Apply(
        func (j,i int, v float64) float64 {
            return (v - minX) / (maxX - minX) * 255.0
        },
        img,
    )

    out[0] = *mat.NewDense(height, width, nil)
    out[1] = *mat.NewDense(height, width, nil)
    out[2] = *mat.NewDense(height, width, nil)
    out[0].Copy(img)
    out[1].Copy(img)
    out[2].Copy(img)
}


/*
SUMMARY
    Inverts the colour of an image.
PARAMETERS
    N/A
RETURN
    N/A
*/
func (out RGBImg) InvertColour() {
    height, width := out[0].Dims()
    for ch:=0; ch<3; ch++ {
        for y:=0; y < height; y++ {
            for x:=0; x < width; x++ {
                out[ch].Set(y, x, 255.0 - out[ch].At(y, x))
            }
        }
    }
}


/*
SUMMARY
    Applies function a function to all values in all channels of the image.
PARAMETERS
    F func(j, i int, v float64: the function to apply
RETURN
    N/A
*/
func (out RGBImg) Apply(F func(j, i int, v float64) float64) {
    for ch:=0; ch<3; ch++ {
        out[ch].Apply(F, &out[ch])
    }
}


func (img RGBImg) AddNoise(Sigma, Proportion float64) {
    H, W := img[0].Dims()
    normal := distuv.Normal{Mu: 0.0, Sigma: Sigma, Src: randSrc}
    uniform := distuv.Uniform{Min: 0.0, Max: 1.0, Src: randSrc}
    for ch:=0; ch<3; ch++ {
        for y:=0; y<H; y++ {
            for x:=0; x<W; x++ {
                noise := normal.Rand()
                if uniform.Rand() < Proportion {
                    img[ch].Set(y, x, img[0].At(y, x) + noise)
                }
                if img[ch].At(y, x) > 255.0 {
                    img[ch].Set(y, x, 255.0)
                }
                if img[ch].At(y, x) < 0.0 {
                    img[ch].Set(y, x, 0.0)
                }
            }
        }
    }
}


/*
SUMMARY
    Dithers an image (all three channels) using Jarvis-Judice-Ninke dithering,
    this is visually better than the Floyd–Steinberg dithering.
PARAMETERS
    NumColours int: number of colours to dither into, should be in the range of 2-255
RETURN
    N/A
*/
func (out RGBImg) Dither(NumColours int) {
    height, width := out[0].Dims()
    maskVector := []float64{0,0,0,7,5, 3,5,7,5,3, 1,3,5,3,1}
    mask := mat.NewDense(3, 5, maskVector)
    mask.Apply(func (i, j int, v float64) float64 {return v / 48}, mask)
    errorNumber := 0.0

    for y:=0; y<height; y++ {
        for x:=0; x<width; x++ {
            for c:=0; c<3; c++ {
                oldValue := out[c].At(y,x)
                newValue := DitherColorMap(oldValue, NumColours)
                out[c].Set(y,x, newValue)
                errorNumber = oldValue - newValue
                if -1<x-2 && x+2<width && y+2<height {
                    tmpMask := mat.NewDense(3, 5, nil)
                    tmpMask.Copy(mask)
                    tmpMask.Apply(func (i, j int, v float64) float64 {return v * errorNumber}, tmpMask)
                    slice := out[c].Slice(y,y+3, x-2,x+3)
                    tmpMask.Add(slice, tmpMask)
                    for j:=0; j < 3; j++ {
                        for i:=0; i < 5; i++ {
                            out[c].Set(y+j, x-2+i, tmpMask.At(j,i))
                        }
                    }
                }
            }
        }
    }
}