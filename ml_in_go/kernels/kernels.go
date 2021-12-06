package kernels

import (
    "math"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
)

const RBF = 0
const LINEAR = 1
const PERIODIC = 2


type Parameters struct {
    Type int
    VarSigma float64
    LengthScale float64
    Period float64
}


/*
SUMMARY
    Given two column column vectors (not necessarily same length),
    computes the radial basis function (RBF) kernel
PARAMETERS
    x1 *mat.Dense: M by N matrix
    x2 *mat.Dense: M' by N matrix
    Params Parameters: struct containing the type of kernel and respective kernel parameters
RETURN
    *mat.Dense: a matrix containing the kernel
*/
func Kernel(x1, x2 *mat.Dense, Params Parameters) *mat.Dense {
    kernelType, varSigma, lengthScale, period := Params.Type, Params.VarSigma, Params.LengthScale, Params.Period
    const EUCLIDEAN_DISTANCE = 2
    NX1, _ := x1.Dims()
    NX2, _ := x2.Dims()
    kernel := mat.NewDense(NX1, NX2, nil)
    kernel.Apply(func (j,i int, v float64) float64 {
                    switch kernelType {
                        case RBF:
                            dist := floats.Distance(mat.Row(nil, j, x1), mat.Row(nil, i, x2), EUCLIDEAN_DISTANCE)
                            return varSigma*math.Exp(-dist * dist / lengthScale)
                        case LINEAR:
                            dot := floats.Dot(mat.Row(nil, j, x1), mat.Row(nil, i, x2))
                            return varSigma * dot
                        case PERIODIC:
                            dist := floats.Distance(mat.Row(nil, j, x1), mat.Row(nil, i, x2), EUCLIDEAN_DISTANCE)
                            return varSigma * math.Exp(-2.0*math.Pow(math.Sin((math.Pi/period)*dist), 2.0)/math.Pow(lengthScale, 2.0))
                    }
                    return 0.0
                 }, kernel)
    // adding epsilon*(unit matrix) for numerical stability
    kernel.Apply(func (j,i int, v float64) float64 {
                    if i == j {
                        return v + 0.0001
                    }
                    return v
                 }, kernel)
    return kernel
}




