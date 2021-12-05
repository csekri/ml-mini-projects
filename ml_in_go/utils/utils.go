package utils

import (
    "gonum.org/v1/gonum/mat"
)


/*
SUMMARY
    Flattens a matrix into a slice.
PARAMETERS
    matrix *mat.Dense: M by N matrix to be flattened
    RowWise bool: if true the flattening happens from left to right, row after row,
        if false the flattened happens from up to down, column after column
RETURN
    []float64: slice with length M*N
*/
func Flatten(matrix *mat.Dense, RowWise bool) []float64 {
    M, N := matrix.Dims()
    flattened := make([]float64, M*N)
    for y:=0; y<M; y++ {
        for x:=0; x<N; x++ {
            if RowWise {
                flattened[y*N + x] = matrix.At(y,x)
            } else {
                flattened[x*M + y] = matrix.At(y,x)
            }
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


/*
SUMMARY
    Creates equally spaced points in an interval.
PARAMETERS
    Min float64: lower bound of the interval
    Max float64: upper bound of the interval
    Num int: the number of the equally spaced points
RETURN
    []float64: slice of equally paced number.
*/
func Linspace(Min, Max float64, Num int) []float64 {
    slice := make([]float64, Num)
    for i := range slice {
        slice[i] = Min + float64(i) / float64(Num) * (Max - Min)
    }
    return slice
}

