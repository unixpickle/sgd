package sgd

import "github.com/unixpickle/autofunc"

// A Learner is anything with some parameters.
type Learner interface {
	Parameters() []*autofunc.Variable
}

// A Gradienter is anything which can compute a
// gradient for a set of samples.
//
// In general, it is not safe to call a Gradienter's
// methods from multiple Goroutines at once.
type Gradienter interface {
	// Gradient returns the total error gradient for
	// all the samples in a set.
	// The returned result is only valid until the
	// next call to Gradient (or to RGradient, if
	// this is also an RGradienter)
	Gradient(SampleSet) autofunc.Gradient
}

// An RGradienter is anything which can compute
// a gradient and r-gradient for a set of samples.
//
// Just like for Gradienter, it is not safe to call
// an RGradienter's methods concurrently.
type RGradienter interface {
	Gradienter

	// RGradient returns the total error gradient and
	// r-gradient all the samples in a set.
	// The returned result is only valid until the
	// next call to Gradient or RGradient.
	RGradient(autofunc.RVector, SampleSet) (autofunc.Gradient, autofunc.RGradient)
}

// A Transformer transforms gradients in some way, such as
// preconditioning the gradients or applying momentum.
//
// The Transform method may modify its argument, and its
// return value is not guaranteed to be related to its
// argument in any way.
type Transformer interface {
	Transform(autofunc.Gradient) autofunc.Gradient
}
