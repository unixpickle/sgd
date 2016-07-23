// +build js

package sgd

// SGDInteractive is like SGD, but it calls a
// function before every epoch and stops when
// said function returns false, or when the
// user sends a kill signal.
//
// This is not supported in the js platform.
func SGDInteractive(g Gradienter, s SampleSet, stepSize float64, batchSize int, sf func() bool) {
	panic("not supported on this platform")
}
