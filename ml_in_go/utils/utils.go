package utils

import (
    "gonum.org/v1/gonum/mat"
)


/*
SUMMARY
    Flattens a matrix into a slice.
PARAMETERS
    matrix *mat.Dense: M by N matrix to be flattened
RETURN
    []float64: slice with length M*N
*/
func Flatten(matrix *mat.Dense) []float64 {
    M, N := matrix.Dims()
    flattened := make([]float64, M*N)
    for y:=0; y<M; y++ {
        for x:=0; x<N; x++ {
            flattened[y*N + x] = matrix.At(y,x)
        }
    }
    return flattened
}


/*
SUMMARY
    Converts a dense symmetric matrix to a regular dense matrix.
PARAMETERS
    matrix *mat.SymDense: N by N symmetric matrix to be converted
RETURN
    *mat.Dense: the converted dense matrix
*/
func Sym2Dense(in *mat.SymDense) *mat.Dense {
    N, _ := in.Dims()
    sym := mat.NewDense(N, N, nil)
    for y:=0; y<N; y++ {
        for x:=0; x<N; x++ {
            sym.Set(y,x, in.At(y,x))
        }
    }
    return sym
}

/*
SUMMARY
    Converts a dense matrix to a symmetric dense matrix.
PARAMETERS
    matrix *mat.Dense: M by M symmetric matrix to be converted
RETURN
    *mat.SymDense: the converted symmetric dense matrix
*/
func Dense2Sym(in *mat.Dense) *mat.SymDense {
    N, _ := in.Dims()
    ordinary := mat.NewSymDense(N, nil)
    for y:=0; y<N; y++ {
        for x:=0; x<N; x++ {
            ordinary.SetSym(y,x, in.At(y,x))
        }
    }
    return ordinary
}
