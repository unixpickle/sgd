// +build js

package sgd

// SGDInteractive is like SGD, but it calls a
// function before every epoch and stops when
// said function returns false, or when the
// user sends a kill signal.
// The sf function may modify the SampleSet,
// and the new sample set will be used on the
// next iteration.
//
// This is not supported in the js platform.
func SGDInteractive(g Gradienter, s SampleSet, stepSize float64, batchSize int, sf func() bool) {
	panic("not supported on this platform")
}
