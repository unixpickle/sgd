// +build !js

package sgd

import "github.com/unixpickle/sgd"

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

// SGDMini is like SGDInteractive, but the interactive
// function is called for each mini-batch rather than
// for each complete epoch.
// Since each mini-batch is different, sf is passed the
// next mini-batch so it can perform mini-batch-specific
// tasks.
func SGDMini(g Gradienter, s SampleSet, stepSize float64, batchSize int,
	sf func(batch SampleSet) bool) {
	shuffledSet := s.Copy()
	sampleIdx := shuffledSet.Len()
	var subset sgd.SampleSet
	loopUntilKilled(func() bool {
		sampleIdx += batchSize
		if sampleIdx >= shuffledSet.Len() {
			sampleIdx = 0
			sgd.ShuffleSampleSet(shuffledSet)
		}
		bs := batchSize
		if bs > shuffledSet.Len()-sampleIdx {
			bs = shuffledSet.Len() - sampleIdx
		}
		subset = shuffledSet.Subset(sampleIdx, bs)
		return sf(subset)
	}, func() {
		grad := g.Gradient(subset)
		grad.AddToVars(-stepSize)
	})
}
