// +build !js

package sgd

// SGDInteractive is like SGD, but it calls sf before
// each epoch and stops when sf returns false.
//
// On platforms which support interrupts, it also stops
// for an os.Interrupt signal.
//
// Calling sf may modify the SampleSet for the next epoch,
// allowing for a dynamic set of samples (provided that
// the SampleSet can be modified in place).
func SGDInteractive(g Gradienter, s SampleSet, stepSize float64, batchSize int, sf func() bool) {
	loopUntilKilled(sf, func() {
		SGD(g, s, stepSize, 1, batchSize)
	})
}
