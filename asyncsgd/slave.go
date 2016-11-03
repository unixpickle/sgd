package asyncsgd

import (
	"errors"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
)

// A Slave operates an SGD training node.
type Slave struct {
	batchSize  int
	client     *ParamClient
	gradienter sgd.Gradienter
	params     []*autofunc.Variable

	miniBatch sgd.SampleSet
	samples   sgd.SampleSet
	sampleIdx int

	accumGrad autofunc.Gradient
}

// NewSlave creates a slave for a training scenario.
func NewSlave(g sgd.Gradienter, s sgd.SampleSet, batchSize int, c *ParamClient,
	p []*autofunc.Variable) *Slave {
	res := &Slave{
		batchSize:  batchSize,
		client:     c,
		gradienter: g,
		params:     p,

		samples:   s,
		sampleIdx: s.Len(),
	}
	res.cycleMinibatch()
	return res
}

// Batch returns a copy of the current minibatch.
func (s *Slave) Batch() sgd.SampleSet {
	return s.miniBatch.Copy()
}

// Step runs SGD on the next mini-batch and advances to
// the next mini-batch.
func (s *Slave) Step() {
	batchGrad := s.gradienter.Gradient(s.Batch())
	if s.accumGrad == nil {
		s.accumGrad = batchGrad.Copy()
	} else {
		s.accumGrad.Add(batchGrad)
	}
	s.cycleMinibatch()
}

// Sync syncs with the parameter server.
func (s *Slave) Sync() error {
	if s.accumGrad != nil {
		if err := s.client.WriteParams(s.accumGrad, s.params); err != nil {
			return errors.New("write params: " + err.Error())
		}
		s.accumGrad = nil
	}
	if err := s.client.ReadParams(s.params); err != nil {
		return errors.New("read params: " + err.Error())
	}
	return nil
}

// Loop runs the slave until an error occurs.
//
// The syncInterval is the number of mini-batches between
// parameter synchronizations.
//
// If non-nil, the logFunc callback is called with the
// next and last mini-batches.
// The first call will have a nil last mini-batch.
func (s *Slave) Loop(syncInterval int, logFunc func(next, last sgd.SampleSet)) error {
	var unsyncCount int
	var last sgd.SampleSet
	for {
		next := s.Batch()
		if logFunc != nil {
			logFunc(next, last)
		}
		last = next
		s.Step()
		unsyncCount++
		if unsyncCount >= syncInterval {
			unsyncCount = 0
			if err := s.Sync(); err != nil {
				return err
			}
		}
	}
}

func (s *Slave) cycleMinibatch() {
	if s.miniBatch == nil || s.sampleIdx+s.miniBatch.Len() >= s.samples.Len() {
		s.sampleIdx = 0
		sgd.ShuffleSampleSet(s.samples)
	} else {
		s.sampleIdx += s.miniBatch.Len()
	}
	bs := s.batchSize
	if bs > s.samples.Len()-s.sampleIdx {
		bs = s.samples.Len() - s.sampleIdx
	}
	s.miniBatch = s.samples.Subset(s.sampleIdx, s.sampleIdx+bs)
}
