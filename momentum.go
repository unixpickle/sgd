package sgd

import "github.com/unixpickle/autofunc"

// Momentum implements basic momentum.
//
// When used as a Gradienter, this will use its wrapped
// Gradienter to acquire gradients and then pass said
// gradients to Transform.
type Momentum struct {
	Gradienter Gradienter
	Momentum   float64

	velocity autofunc.Gradient
}

func (m *Momentum) Gradient(s SampleSet) autofunc.Gradient {
	return m.Transform(m.Gradienter.Gradient(s))
}

func (m *Momentum) Transform(grad autofunc.Gradient) autofunc.Gradient {
	if m.velocity == nil {
		m.velocity = grad.Copy()
	} else {
		m.velocity.Scale(m.Momentum)
		m.velocity.Add(grad)
	}
	for variable, vec := range m.velocity {
		copy(grad[variable], vec)
	}
	return grad
}
