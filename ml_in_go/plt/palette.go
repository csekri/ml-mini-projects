package plt

import (
    "image/color"
    "golang.org/x/exp/rand"
    "gonum.org/v1/plot/palette/moreland"
)

/*
A customisable type that implements gonum/plot/palette interface.
    Colours []color.Color: defines all the colours in the palette
*/
type CustomPalette struct {
    Colours []color.Color
}

/*
SUMMARY
    The interface method of gonum/plot/palette defined on CustomPalette.
PARAMETERS
    N/A
RETURN
    []color.Color: the colours of the palette
*/
func (cp CustomPalette) Colors() []color.Color {
    return cp.Colours
}

// types defined for DesignedPalette
const BLACK_TRANSITION_PALETTE = 0
const RANDOM_PALETTE = 1
const KINDLMANN_PALETTE = 2
const EXTENDED_KINDLMANN_PALETTE = 3
const BLACK_BODY_PALETTE = 4
const EXTENDED_BLACK_BODY_PALETTE = 5
const UNI_PALETTE = 6

/*
This type can conveniently create many different kinds of palettes conforming with gonum/plot/palette.
    Type int: the type of the palette, these are defined above
    Num int: the number of colours in the palette
    Extra int: information carrying extra information about the palette (not every type requires it)
    Reverse bool: if false the colour mapping is in regular order, if true the colour mapping order is reversed
*/
type DesignedPalette struct {
    Type int
    Num int
    Extra int
    Reverse bool
}

/*
SUMMARY
    The gonum/plot/palette interface method for DesignedPalette.
PARAMETERS
    N/A
RETURN
    []color.Color: the colours of the palette
*/
func (dp DesignedPalette) Colors() []color.Color {
    colours := make([]color.Color, dp.Num)
    switch dp.Type {
        case BLACK_TRANSITION_PALETTE:
            // in this case extra represents a colour as A + 255*B + 255*255*G + 255*255*255*R, hexadecimal notation is more intuitive
            r := (dp.Extra >> 24) & 255
            g := (dp.Extra >> 16) & 255
            b := (dp.Extra >> 8) & 255
            a := dp.Extra & 255
            for i := range colours {
                colours[i] = color.RGBA{R: uint8(r*i/dp.Num), G: uint8(g*i/dp.Num), B: uint8(b*i/dp.Num), A: uint8(a)}
            }
        case RANDOM_PALETTE:
            // in this case extra represents a random seed
            randSrc := rand.NewSource(uint64(dp.Extra))
            randGen := rand.New(randSrc)
            for i := range colours {
                colours[i] = color.RGBA{R: uint8(randGen.Int()%256), G: uint8(randGen.Int()%256), B: uint8(randGen.Int()%256), A: 255}
            }
        case KINDLMANN_PALETTE:
            cMap := moreland.Kindlmann()
            cMap.SetMax(1.0)
            cMap.SetMin(0.0)
            for i := range colours {
                value := cMap.Min() + float64(i) / float64(dp.Num) * (cMap.Max() - cMap.Min())
                colours[i], _ = cMap.At(value)
            }
        case EXTENDED_KINDLMANN_PALETTE:
            cMap := moreland.ExtendedKindlmann()
            cMap.SetMax(1.0)
            cMap.SetMin(0.0)
            for i := range colours {
                value := cMap.Min() + float64(i) / float64(dp.Num) * (cMap.Max() - cMap.Min())
                colours[i], _ = cMap.At(value)
            }
        case BLACK_BODY_PALETTE:
            cMap := moreland.BlackBody()
            cMap.SetMax(1.0)
            cMap.SetMin(0.0)
            for i := range colours {
                value := cMap.Min() + float64(i) / float64(dp.Num) * (cMap.Max() - cMap.Min())
                colours[i], _ = cMap.At(value)
            }
        case EXTENDED_BLACK_BODY_PALETTE:
            cMap := moreland.ExtendedBlackBody()
            cMap.SetMax(1.0)
            cMap.SetMin(0.0)
            for i := range colours {
                value := cMap.Min() + float64(i) / float64(dp.Num) * (cMap.Max() - cMap.Min())
                colours[i], _ = cMap.At(value)
            }
        case UNI_PALETTE:
            r, g, b, a := HEX2RGBA(dp.Extra)
            for i := range colours {
                colours[i] = color.RGBA{r, g, b, a}
            }
    }
    if dp.Reverse {
        for i, j := 0, len(colours)-1; i < j; i, j = i+1, j-1 {
            colours[i], colours[j] = colours[j], colours[i]
        }
    }
    return colours
}
