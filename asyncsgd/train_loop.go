package asyncsgd

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
)

// TrainLoop runs an SGD training node.
//
// The batchSize argument specifies the number of samples
// to process at once.
//
// The syncDelay argument specifies how many batches to
// run between syncing with the parameter server.
//
// If non-nil, the logFunc callback is called after every
// synchronization to log the network's progress.
func TrainLoop(g sgd.Gradienter, s sgd.SampleSet, batchSize, syncDelay int,
	c *ParamClient, p []*autofunc.Variable, logFunc func()) error {
	shuffledSet := s.Copy()
	sampleIdx := shuffledSet.Len()

	var unsyncedBatches int
	var totalGrad autofunc.Gradient
	for {
		sampleIdx += batchSize
		if sampleIdx >= shuffledSet.Len() {
			sampleIdx = 0
			sgd.ShuffleSampleSet(shuffledSet)
		}
		bs := batchSize
		if bs > shuffledSet.Len()-sampleIdx {
			bs = shuffledSet.Len() - sampleIdx
		}
		subset := shuffledSet.Subset(sampleIdx, sampleIdx+bs)
		batchGrad := g.Gradient(subset)
		if totalGrad == nil {
			totalGrad = batchGrad.Copy()
		} else {
			totalGrad.Add(batchGrad)
		}
		unsyncedBatches++
		if unsyncedBatches >= syncDelay {
			if err := c.WriteParams(totalGrad, p); err != nil {
				return errors.New("write params: " + err.Error())
			}
			if err := c.ReadParams(p); err != nil {
				return errors.New("read params: " + err.Error())
			}
			unsyncedBatches = 0
			totalGrad = nil
			if logFunc != nil {
				logFunc()
			}
		}
	}
}
