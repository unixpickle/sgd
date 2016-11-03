package sgd

import (
	"math"

	"github.com/unixpickle/autofunc"
)

// AdaGrad is a Gradienter and a Transformer which
// implements the AdaGrad SGD algorithm.
//
// When used as a Gradienter, this will use its wrapped
// Gradienter to acquire gradients and then pass said
// gradients to Transform.
type AdaGrad struct {
	Gradienter Gradienter
	Damping    float64

	squaredHistory autofunc.Gradient
}

func (a *AdaGrad) Gradient(s SampleSet) autofunc.Gradient {
	return a.Transform(a.Gradienter.Gradient(s))
}

func (a *AdaGrad) Transform(actualGrad autofunc.Gradient) autofunc.Gradient {
	if a.squaredHistory == nil {
		a.squaredHistory = actualGrad.Copy()
		for _, v := range a.squaredHistory {
			for i, x := range v {
				v[i] *= x
			}
		}
	} else {
		for variable, vec := range actualGrad {
			histVec := a.squaredHistory[variable]
			for i, x := range vec {
				histVec[i] += x * x
			}
		}
	}

	for variable, vec := range actualGrad {
		histVec := a.squaredHistory[variable]
		for i, x := range histVec {
			vec[i] /= math.Sqrt(x) + a.Damping
		}
	}

	return actualGrad
}
