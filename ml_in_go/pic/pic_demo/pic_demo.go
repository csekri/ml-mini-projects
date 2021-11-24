/*
This demo show some functions in the library pic in action.
We load the image, convert it to grayscale, then turn it into a
binary image using dithering with 2 colours.
*/
package main

import (
    "ml_playground/pic"
    "gonum.org/v1/gonum/mat"
)

func main() {
    var img pic.RGBImg = make([]mat.Dense, 3)
    err := img.LoadPixels("image.jpg")
    if err != nil { panic(err) }

    img.GrayScale()
    img.Dither(2)
    img.SaveImage("outimage.jpg")
}
