package pic

import (
  "os"
  "image"
  "image/gif"

  "gonum.org/v1/plot"
  "gonum.org/v1/plot/vg"
)

// this type is used to create gif animations
type GifMaker struct {
    Images []*image.Paletted
    Width int
    Height int
    Delay int
    Delays []int
}


/*
SUMMARY
    In the gif encoder it is important to use paletted image. After all a gif contains
    a reduced number of colours.
PARAMETERS
    img image.Image: the input image
RETURN
    *image.Paletted: the paletted image
*/
func ImageToPaletted(img image.Image) *image.Paletted {
	pm, ok := img.(*image.Paletted)
	if !ok {
		b := img.Bounds()
		pm = image.NewPaletted(b, nil)
		q := &MedianCutQuantizer{NumColor: 256}
		q.Quantize(pm, b, img, image.ZP)
	}
	return pm
}


/*
SUMMARY
    Adds a new plot frame to the animation.
PARAMETERS
    p *plot.Plot: the plot in the new frame
RETURN
    N/A
*/
func (gm *GifMaker) CollectFrames(p *plot.Plot)  {
    p.Save(vg.Points(float64(gm.Width/4*3)), vg.Points(float64(gm.Height/4*3)), "tmp.png")
	infile, err := os.Open("tmp.png")
    if err != nil {
        // replace this with real error handling
        panic(err)
    }
    defer infile.Close()

    // Decode will figure out what type of image is in the file on its own.
    // We just have to be sure all the image packages we want are imported.
    src, _, err := image.Decode(infile)
    if err != nil {
        // replace this with real error handling
        panic(err)
    }

    imgPal := ImageToPaletted(src)
    gm.Images = append(gm.Images, imgPal)
    gm.Delays = append(gm.Delays, gm.Delay)
}


/*
SUMMARY
    Adds a new image frame to the animation.
PARAMETERS
    Img image.Image: the image in the new frame
RETURN
    N/A
*/
func (gm *GifMaker) CollectImages(Img image.Image)  {
    imgPal := ImageToPaletted(Img)
    gm.Images = append(gm.Images, imgPal)
    gm.Delays = append(gm.Delays, gm.Delay)
}


/*
SUMMARY
    Renders and saves all the collected frames.
PARAMETERS
    filename string: the filename/path of the gif file we save
RETURN
    N/A
*/
func (gm *GifMaker) RenderFrames(filename string) {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		panic(err)
		return
	}
	defer f.Close()
	gif.EncodeAll(f, &gif.GIF{
		Image: gm.Images,
		Delay: gm.Delays,
	})
}
