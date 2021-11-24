/*
This library contains a few functions and methods
which helps with image IO and manipulation.
*/
package pic

import (
    "image"
    "image/jpeg"
    "image/color"
    "os"
    "gonum.org/v1/gonum/mat"
)

// This type is used to manipulate the pixels of an image, no alpha channel is present
type RGBImg []mat.Dense


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