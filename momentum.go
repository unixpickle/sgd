package sgd

import "github.com/unixpickle/autofunc"

// Momentum implements basic momentum.
type Momentum struct {
	Gradienter Gradienter
	Momentum   float64

	velocity autofunc.Gradient
}

func (m *Momentum) Gradient(s SampleSet) autofunc.Gradient {
	grad := m.Gradienter.Gradient(s)
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
