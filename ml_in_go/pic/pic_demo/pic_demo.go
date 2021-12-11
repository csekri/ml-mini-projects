package main

import (
    "ml_playground/pic"
    "gonum.org/v1/gonum/mat"
)


/*
This demo show some functions in the library pic in action.
We load the image, convert it to grayscale, then turn it into a
binary image using dithering with 2 colours.
*/
func main() {
    var img pic.RGBImg = make([]mat.Dense, 3)
    err := img.LoadPixels("image.jpg")
    if err != nil { panic(err) }

    img.GrayScale()
    img.Dither(2)
    img.SaveImage("outimage.jpg")

    err = img.LoadPixels("dog.jpg")
    if err != nil { panic(err) }

    img.BinaryThreshold(70.0)
    img.AddNoise(150.0, 0.6)
    img.GrayScale()
    img.SaveImage("noisy_scottie.jpg")

}
