package sgd

import "github.com/unixpickle/autofunc"

// A Biaser scales some variables in a gradient, leaving
// the rest unchanged.
type Biaser struct {
	Gradienter Gradienter
	Scales     map[*autofunc.Variable]float64
}

// NewBiaserUniform creates a Biaser which biases all of
// the listed variables by a fixed scaler.
func NewBiaserUniform(g Gradienter, v []*autofunc.Variable, s float64) *Biaser {
	res := &Biaser{
		Gradienter: g,
		Scales:     map[*autofunc.Variable]float64{},
	}
	for _, variable := range v {
		res.Scales[variable] = s
	}
	return res
}

// Gradient obtains a gradient from the inner Gradienter,
// then scales the biased variables appropriately.
func (b *Biaser) Gradient(s SampleSet) autofunc.Gradient {
	res := b.Gradienter.Gradient(s)
	for v, s := range b.Scales {
		if vec, ok := res[v]; ok {
			vec.Scale(s)
		}
	}
	return res
}
