package sgd

import (
	"math"

	"github.com/unixpickle/autofunc"
)

type GradientNorm int

const (
	L2Norm GradientNorm = iota
	InfNorm
)

// GradientClipper is a Gradienter which scales down
// gradients so that the norm of the gradient is less
// than a certain value.
//
// When used as a Gradienter, this will use its wrapped
// Gradienter to acquire gradients and then pass said
// gradients to Transform.
type GradientClipper struct {
	Gradienter Gradienter
	Threshold  float64
	Norm       GradientNorm
}

func (c *GradientClipper) Gradient(s SampleSet) autofunc.Gradient {
	return c.Transform(c.Gradienter.Gradient(s))
}

func (c *GradientClipper) Transform(res autofunc.Gradient) autofunc.Gradient {
	var norm float64

	switch c.Norm {
	case L2Norm:
		for _, vec := range res {
			norm += vec.Dot(vec)
		}
		norm = math.Sqrt(norm)
	case InfNorm:
		for _, vec := range res {
			norm = math.Max(norm, vec.MaxAbs())
		}
	}

	if norm > c.Threshold {
		res.Scale(c.Threshold / norm)
	}
	return res
}
