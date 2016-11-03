package sgd

import "github.com/unixpickle/autofunc"

// GradientCapper is a Gradienter which caps individual
// components of the gradient to a certain maximum
// magnitude.
//
// When used as a Gradienter, this will use its wrapped
// Gradienter to acquire gradients and then pass said
// gradients to Transform.
type GradientCapper struct {
	Gradienter Gradienter
	Cap        float64
}

func (g *GradientCapper) Gradient(s SampleSet) autofunc.Gradient {
	return g.Transform(g.Gradienter.Gradient(s))
}

func (g *GradientCapper) Transform(res autofunc.Gradient) autofunc.Gradient {
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
