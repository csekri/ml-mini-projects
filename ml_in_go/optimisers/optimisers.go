package optimisers

import (
    "math"
    "gonum.org/v1/gonum/floats"
)

func SGD(StepSize, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    Terminate := false

    if StepSize <= 0 { panic("Negative/0 step size/learning rate encountered") }
    if ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Gradient := Derivative(At)
        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            NewAt[i] = At[i] - StepSize * Gradient[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func SGDMomentum(StepSize, Friction, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    var Velocity []float64
    Terminate := false

    if StepSize <= 0 { panic("Negative/0 step size/learning rate encountered") }
    if Friction <= 0 { panic("Negative value encountered as initial value for the gradient accumulator") }
    if ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(Velocity) == 0 {
            Velocity = make([]float64, len(At))
        }
        Gradient := Derivative(At)
        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            Velocity[i] = Friction*Velocity[i] + StepSize*Gradient[i]
            NewAt[i] = At[i] - Velocity[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func NesterovAcceleratedGradient(StepSize, Friction, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    var Velocity []float64
    Terminate := false

    if StepSize <= 0 { panic("Negative/0 step size/learning rate encountered") }
    if Friction <= 0 { panic("Negative value encountered as initial value for the gradient accumulator") }
    if ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(Velocity) == 0 {
            Velocity = make([]float64, len(At))
        }
        ModifiedAt := make([]float64, len(At))
        for i := range ModifiedAt {
            ModifiedAt[i] = At[i] - Friction * Velocity[i]
        }
        Gradient := Derivative(ModifiedAt)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            Velocity[i] = Friction*Velocity[i] + StepSize*Gradient[i]
            NewAt[i] = At[i] - Velocity[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func BacktrackingLineSearch(Beta, ConvergeEpsilon float64) func(func ([]float64) float64, func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    Terminate := false

    if Beta <= 0 || Beta >= 1 { panic("Negative/0 beta parameter encountered in BacktrackingLineSearch") }
    if ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(F func ([]float64) float64, Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Gradient := Derivative(At)
        SearchAt := make([]float64, len(At))

        LeftHandSide := 1.0
        RightHandSide := 0.0
        NormSquare := math.Pow(floats.Norm(Gradient, 2), 2)
        t := 1.0
        for ; LeftHandSide > RightHandSide; {
            for i := range At {
                SearchAt[i] = At[i] - t * Gradient[i]
            }
            LeftHandSide = F(SearchAt)
            RightHandSide = F(At) - t / 2.0 * NormSquare
            t *= Beta
        }
        Steps++
        if floats.Distance(At, SearchAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return SearchAt, Terminate, Steps
    }
}

func Adagrad(StepSize, InitialAccumulatorValue, Epsilon, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    var G []float64
    Terminate := false

    if StepSize <= 0 { panic("Negative/0 step size/learning rate encountered") }
    if InitialAccumulatorValue < 0 { panic("Negative value encountered as initial value for the gradient accumulator") }
    if Epsilon <= 0 || ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(G) == 0 {
            G = make([]float64, len(At))
            floats.AddConst(InitialAccumulatorValue, G)
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            G[i] = G[i] + Gradient[i] * Gradient[i]
            NewAt[i] = At[i] - StepSize / (math.Sqrt(G[i] + Epsilon)) * Gradient[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

// coded following the original paper
// https://arxiv.org/pdf/1212.5701.pdf
// at page 3 Algorithm 1
func Adadelta(Decay, Epsilon, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    var G, Delta []float64
    Terminate := false

    if Decay < 0 || Decay > 1 { panic("Adadelta's decay is outside of range [0,1)") }
    if Epsilon <= 0 || ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Steps++
        if len(G) == 0 {
            G = make([]float64, len(At))
            Delta = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            G[i] = Decay * G[i] + (1-Decay) * Gradient[i] * Gradient[i]
            Update := - math.Sqrt((Delta[i]+Epsilon) / (G[i]+Epsilon)) * Gradient[i] // one sqrt is enough
            Delta[i] = Decay * Delta[i] + (1-Decay) * Update * Update
            NewAt[i] = At[i] + Update
        }
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

func RMSprop(StepSize, Decay, Epsilon, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    var G []float64
    Terminate := false

    if StepSize <= 0 { panic("Negative/0 step size/learning rate encountered") }
    if Decay < 0 || Decay > 1 { panic("RMSprop's decay is outside of range [0,1)") }
    if Epsilon <= 0 || ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        if len(G) == 0 {
            G = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            G[i] = Decay * G[i] + (1-Decay) * Gradient[i] * Gradient[i]
            NewAt[i] = At[i] - StepSize / (math.Sqrt(G[i] + Epsilon)) * Gradient[i]
        }
        Steps++
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}

// implemented following the original paper
// https://arxiv.org/pdf/1412.6980.pdf
// at page 2
func Adam(StepSize, Beta1, Beta2, Epsilon, ConvergeEpsilon float64) func(func ([]float64) []float64, []float64) ([]float64, bool, int) {
    Steps := 0
    var M, MHat, V, VHat []float64
    Terminate := false

    if StepSize <= 0 { panic("Negative/0 step size/learning rate encountered") }
    if Beta1 < 0 || Beta1 >= 1 { panic("Adam's beta1 is outside of range [0,1)") }
    if Beta2 < 0 || Beta2 >= 1 { panic("Adam's beta2 is outside of range [0,1)") }
    if Epsilon <= 0 || ConvergeEpsilon <= 0 { panic("Negative/0 epsilon encountered") }

    return func(Derivative func ([]float64) []float64, At []float64) ([]float64, bool, int) {
        Steps++
        if len(M) == 0 {
            M = make([]float64, len(At))
            MHat = make([]float64, len(At))
            V = make([]float64, len(At))
            VHat = make([]float64, len(At))
        }
        Gradient := Derivative(At)

        NewAt := make([]float64, len(Gradient))
        for i := range NewAt {
            M[i] = Beta1 * M[i] + (1-Beta1) * Gradient[i]
            V[i] = Beta2 * V[i] + (1-Beta2) * Gradient[i] * Gradient[i]
            MHat[i] = M[i] / (1 - math.Pow(Beta1, float64(Steps)))
            VHat[i] = V[i] / (1 - math.Pow(Beta2, float64(Steps)))
            NewAt[i] = At[i] - StepSize * MHat[i] / (math.Sqrt(VHat[i]) + Epsilon)
        }
        if floats.Distance(At, NewAt, 2) < ConvergeEpsilon {
            Terminate = true
        }
        return NewAt, Terminate, Steps
    }
}
