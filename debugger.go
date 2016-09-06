package sgd

import (
	"log"
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Debugger is a Gradienter that echos gradients from a
// wrapped Gradienter and simultaneously logs various
// information about the gradients.
type Debugger struct {
	Gradienter Gradienter

	// Parameters stores the list of parameters in a
	// specific order (the order used to print out
	// variable-specific information).
	Parameters []*autofunc.Variable
}

func (d *Debugger) Gradient(s SampleSet) autofunc.Gradient {
	res := d.Gradienter.Gradient(s)

	logGradientStats(res)
	for i, p := range d.Parameters {
		logVariableStats(p, res[p], i)
	}

	return res
}

func logGradientStats(g autofunc.Gradient) {
	var mean, variance float64
	var count int
	for _, v := range g {
		count += len(v)
		for _, x := range v {
			mean += x
			variance += x * x
		}
	}
	mean /= float64(count)
	variance /= float64(count)
	variance -= mean * mean

	log.Printf("Overall mean=%f variance=%f", mean, variance)
}

func logVariableStats(v *autofunc.Variable, grad linalg.Vector, idx int) {
	var mean, variance, maxAbs, meanChange float64
	for i, x := range v.Vector {
		mean += x
		variance += x * x
		maxAbs = math.Max(maxAbs, math.Abs(x))
		meanChange += math.Abs(x / grad[i])
	}
	mean /= float64(len(v.Vector))
	variance /= float64(len(v.Vector))
	meanChange /= float64(len(v.Vector))
	variance -= mean * mean

	log.Printf("Variable %d: mean=%f variance=%f max=%f val/grad=%f", idx, mean, variance,
		maxAbs, meanChange)
}
