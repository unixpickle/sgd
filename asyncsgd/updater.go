package asyncsgd

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
)

// An Updater updates a set of parameters using gradients.
//
// The Update method must never be called more than once
// at a time, and should be given complete ownership of
// the parameters while it's running.
//
// The Update method may modify the gradient.
type Updater interface {
	Update(g autofunc.Gradient)
}

// A GradienterUpdater updates parameters by feeding a
// gradient through an sgd.Gradienter.
type GradienterUpdater struct {
	// StepSize is the SGD step size.
	StepSize float64

	// Gradienter is the gradienter to use.
	Gradienter sgd.Gradienter

	// RootGradienter is the internal field of Gradienter
	// that it uses to compute raw gradients.
	// For instance, if Gradienter is an *sgd.Adam, then
	// this would be &Gradienter.Gradienter.
	RootGradienter *sgd.Gradienter
}

// Update applies the gradienter to g and descends along
// the resulting gradient.
func (g *GradienterUpdater) Update(grad autofunc.Gradient) {
	*g.RootGradienter = &constantGradienter{G: grad}
	newGrad := g.Gradienter.Gradient(nil)
	newGrad.AddToVars(-g.StepSize)
}

type constantGradienter struct {
	G autofunc.Gradient
}

func (c *constantGradienter) Gradient(s sgd.SampleSet) autofunc.Gradient {
	return c.G
}
