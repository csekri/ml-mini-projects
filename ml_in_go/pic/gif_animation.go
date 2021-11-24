package pic

import (
  "os"
  "image"
  "image/gif"

  "gonum.org/v1/plot"
  "gonum.org/v1/plot/vg"
)

type GifMaker struct {
    Images []*image.Paletted
    Width int
    Height int
    Delay int
    Delays []int
}

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
