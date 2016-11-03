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

// A TransformerUpdater updates parameters by feeding them
// into an sgd.Transformer and then doing an SGD step.
type GradienterUpdater struct {
	StepSize    float64
	Transformer sgd.Transformer
}

// Update applies the gradienter to g and descends along
// the resulting gradient.
func (g *GradienterUpdater) Update(grad autofunc.Gradient) {
	g.Transformer.Transform(grad).AddToVars(-g.StepSize)
}
