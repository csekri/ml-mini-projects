package gaussian_processes

import (
    "gonum.org/v1/gonum/mat"

    "ml_playground/utils"
    "ml_playground/kernels"
)


/*
SUMMARY
    For an unknown location Xstar, computes the distribution of its image.
PARAMETERS
    X *mat.Dense: column vector with the x coordinates
    Y *mat.Dense: column vector which is f(X)+noise
    XStar *mat.Dense: column vector of unknown locations of interest
    lengthScale float64: parameter of RBF
    varSigma float64: parameter of the RBF
    betaNoise float64: the precision of the noise
RETURN
    *mat.Dense: mean of the normal distribution
    *mat.Dense: covariance of the normal distribution
*/
func GaussianProcessPrediction(X, Y, XStar *mat.Dense, lengthScale, varSigma, betaNoise float64) (*mat.Dense, *mat.SymDense) {
    params := kernels.Parameters{Type: kernels.RBF, VarSigma: varSigma, LengthScale: lengthScale}
    KStarX := kernels.Kernel(XStar, X, params)
    KXX := kernels.Kernel(X, X, params)
    KXX.Apply(func (j,i int, v float64) float64 {
                    if i == j { return v + 1 / betaNoise }
                    return v
                }, KXX)
    KStarStar := kernels.Kernel(XStar, XStar, params)
    KXX.Inverse(KXX)
    KXXSize, _ := KXX.Dims()
    KStarXSize, _ := KStarX.Dims()
    tmp := mat.NewDense(KStarXSize, KXXSize, nil)
    tmp.Mul(KStarX, KXX)
    mu := mat.NewDense(KStarXSize, 1, nil)
    mu.Mul(tmp, Y)

    sigma := mat.NewDense(KStarXSize, KStarXSize, nil)
    sigma.Mul(tmp, KStarX.T())
    sigma.Sub(KStarStar, sigma)
    return mu, utils.Dense2Sym(sigma)
}
