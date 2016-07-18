// Package sgd implements stochastic gradient descent
// for training neural networks and solving other
// kinds of numerical optimization problems.
//
// It includes many modified variants of SGD, such as
// AdaGrad and RMSProp.
package sgd

// SGD performs stochastic gradient descent using the
// specified Gradienter.
// It runs until a certain number of epochs (full sweeps
// over the sample set) have elapsed.
func SGD(g Gradienter, samples SampleSet, stepSize float64, epochs, batchSize int) {
	s := samples.Copy()
	for i := 0; i < epochs; i++ {
		ShuffleSampleSet(s)
		for j := 0; j < s.Len(); j += batchSize {
			count := batchSize
			if count > s.Len()-j {
				count = s.Len() - j
			}
			subset := s.Subset(j, j+count)
			grad := g.Gradient(subset)
			grad.AddToVars(-stepSize)
		}
	}
}
