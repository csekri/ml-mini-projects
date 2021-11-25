package optimisers

import (
    "math"
    "gonum.org/v1/gonum/floats"
)

/*
Optimisation introduction

A scalar map is a function that maps one or more variables to just one variable. Let f: R^N -> R be a scalar map.
We want to find local extremums (maxima and minima) of f. The process of finding these is called optimisation.
We often focus on either just minima or maxima in a problem. Finding maxima in f is equivalent to finding minima in -f.
Thus if it is not mentioned otherwise, we always find the minima when optimisation. In optimisation we call
f the objective function.

The gradient

Because f is scalar, the derivative is a column vector of length N. This contains all the partial derivatives of f.
The derivative of the scalar map is also called gradient. The gradient defines a direction in the space. It points
towards the direction where the local increase of f is the greatest (this is not difficult to justify with maths).

Steps of optimisation

During optimisation we select a point in the domain of f. We compute the gradient there and at the opposite
direction we take a step and arrive at a new location. We repeat this until convergence.
*/





/*
SUMMARY
    Gradient descent with constant step size; or stochastic gradient descent (the name we know it from
    Computer Science).
PARAMETERS
    StepSize float64: step size or learning rate
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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


/*
SUMMARY
    SGD with momentum
PARAMETERS
    StepSize float64: step size or learning rate
    Friction float64: determines how much momentum is taken into account
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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


/*
SUMMARY
    Nesterov accelerated gradient (NAG)
PARAMETERS
    StepSize float64: step size or learning rate
    Friction float64: determines how much momentum is taken into account
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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


/*
SUMMARY
    Gradient descent with backtracking line search
PARAMETERS
    Beta float64: parameter that sets what step sizes the algorithm will choose from
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) float64, func ([]float64) []float64, []float64) ([]float64, bool, int):
        a function that takes the objective function a gradient function and a position and return
        the new position, boolean whether the convergence has happened and the number of steps already
        taken place
*/
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


/*
SUMMARY
    Adagrad
PARAMETERS
    StepSize float64: step size or learning rate
    InitialAccumulatorValue float64: the gradient accumulator starts with this value, helps the optimisation
        starting off faster
    Epsilon float64: ensures the denominator is not zero
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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

/*
SUMMARY
    Adadelta, coded following the original paper https://arxiv.org/pdf/1212.5701.pdf at page 3 Algorithm 1
PARAMETERS
    Decay float64: decreases the accumulator each time, mitigates the slowed learning rate in late stages
        of the optimisation of Adagrad (read the paper for full detail)
    Epsilon float64: ensures the denominator is not zero
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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


/*
SUMMARY
    RMSprop
PARAMETERS
    StepSize float64: step size or learning rate
    Decay float64: decreases the accumulator each time (same motivation as in Adadelta)
    Epsilon float64: ensures the denominator is not zero
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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

/*
SUMMARY
    Adam, implemented following the original paper https://arxiv.org/pdf/1412.6980.pdf at page 2
PARAMETERS
    StepSize float64: step size or learning rate
    Beta1 float64: decay factor for the first moment accumulator (read the paper for full details)
    Beta2 float64: decay factor for the second moment accumulator (read the paper for full details)
    Epsilon float64: ensures the denominator is not zero
    ConvergeEpsilon: epsilon in the convergence criterion
RETURN
    func(func ([]float64) []float64, []float64) ([]float64, bool, int): a function that takes
        a gradient function and a position and return the new position, boolean whether the convergence
        has happened and the number of steps already taken place
*/
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
