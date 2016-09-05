package sgd

import "github.com/unixpickle/autofunc"

// GradientCapper is a Gradienter which caps individual
// components of the gradient to a certain maximum
// magnitude.
type GradientCapper struct {
	Gradienter Gradienter
	Cap        float64
}

func (g *GradientCapper) Gradient(s SampleSet) autofunc.Gradient {
	res := g.Gradienter.Gradient(s)
	for _, vec := range res {
		for i, x := range vec {
			if x > g.Cap {
				vec[i] = g.Cap
			} else if x < -g.Cap {
				vec[i] = -g.Cap
			}
		}
	}
	return res
}
