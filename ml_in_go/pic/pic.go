package pic

import (
  "fmt"
  "image"
  "image/jpeg"
  "image/color"
  "os"
  "gonum.org/v1/gonum/mat"
)

type RGBImg []mat.Dense

func main() {
    var img RGBImg = make([]mat.Dense, 3)

    err := img.LoadPixels("human5.jpg")
    if err != nil {
        fmt.Println("Error: File could not be opened")
        os.Exit(1)
    }
    img.GrayScale()
    img.Dither(4)

    img.SaveImage("outimage.jpg")

   
  
  
}
func (out RGBImg) LoadPixels(str string) error {
    image.RegisterFormat("jpeg", "\xff\xd8", jpeg.Decode, jpeg.DecodeConfig)
    file, err := os.Open(str)

    if err != nil {
        fmt.Println("Error: File could not be opened")
        os.Exit(1)
    }

    defer file.Close()

    img, _, err := image.Decode(file)

    if err != nil {
        return err
    }

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

func (in RGBImg) SaveImage(filename string) {
    height, width := in[0].Dims()
    pic := image.NewRGBA(image.Rect(0, 0, width, height))
    for y := 0; y < height; y++ {
        for x := 0; x < width; x++ {
            pic.Set(x, y, color.RGBA{uint8(in[0].At(y,x)), uint8(in[1].At(y,x)), uint8(in[2].At(y,x)), 255})
        }
    }

    f, err := os.Create(filename)
    if err != nil {
        fmt.Println("Error: File could not be written")
        os.Exit(1)
    }
    defer f.Close()

    // Encode to `PNG` with `DefaultCompression` level
    // then save to file
    err = jpeg.Encode(f, pic, &jpeg.Options{98})
    if err != nil {
        // Handle error
    }
}


func DitherColorMap(colour_value float64, num_colours int) float64 {
    divide_factor := 256 / num_colours
    multiply_factor := 255 / (num_colours-1)
    return float64(int(colour_value) / divide_factor) * float64(multiply_factor)
}

func (out RGBImg) GrayScale() {
    height, width := out[0].Dims()
    tmp_img := mat.NewDense(height, width, nil)
    tmp_img.Copy(&out[0])
    tmp_img.Add(tmp_img, &out[1])
    tmp_img.Add(tmp_img, &out[2])

    tmp_img.Apply(func (i, j int, v float64) float64 {return v / 3}, tmp_img)
    out[0] = *mat.NewDense(height, width, nil)
    out[1] = *mat.NewDense(height, width, nil)
    out[2] = *mat.NewDense(height, width, nil)
    out[0].Copy(tmp_img)
    out[1].Copy(tmp_img)
    out[2].Copy(tmp_img)
}


func (out RGBImg) Dither(num_colours int) {
    height, width := out[0].Dims()
    mask_vector := []float64{0,0,0,7,5, 3,5,7,5,3, 1,3,5,3,1}
    mask := mat.NewDense(3, 5, mask_vector)
    mask.Apply(func (i, j int, v float64) float64 {return v / 48}, mask)

    error_number := 0.0
    for y:=0; y<height; y++ {
        for x:=0; x<width; x++ {
            for c:=0; c<3; c++ {
                old_value := out[c].At(y,x)
                new_value := DitherColorMap(old_value, num_colours)
                out[c].Set(y,x, new_value)
                error_number = old_value - new_value
                if -1<x-2 && x+2<width && y+2<height {
                    tmp_mask := mat.NewDense(3, 5, nil)
                    tmp_mask.Copy(mask)
                    tmp_mask.Apply(func (i, j int, v float64) float64 {return v * error_number}, tmp_mask)
                    slice := out[c].Slice(y,y+3, x-2,x+3)
                    tmp_mask.Add(slice, tmp_mask)
                    for j:=0; j < 3; j++ {
                        for i:=0; i < 5; i++ {
                            out[c].Set(y+j, x-2+i, tmp_mask.At(j,i))
                        }
                    }
                }
            }
        }
    }
}