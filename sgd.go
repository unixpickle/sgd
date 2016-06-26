// Package sgd implements stochastic gradient descent
// for training neural networks and solving other
// kinds of numerical optimization problems.
//
// It includes many modified variants of SGD, such as
// AdaGrad and RMSProp.
package sgd

import (
	"fmt"
	"os"
	"os/signal"
	"sync/atomic"
)

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

// SGDInteractive is like SGD, but it calls a
// function before every epoch and stops when
// said function returns false, or when the
// user sends a kill signal.
func SGDInteractive(g Gradienter, s SampleSet, stepSize float64, batchSize int, sf func() bool) {
	var killed uint32

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	defer func() {
		select {
		case <-c:
		default:
			signal.Stop(c)
			close(c)
		}
	}()

	go func() {
		_, ok := <-c
		if !ok {
			return
		}
		signal.Stop(c)
		close(c)
		atomic.StoreUint32(&killed, 1)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
	}()

	for atomic.LoadUint32(&killed) == 0 {
		if sf != nil {
			if !sf() {
				return
			}
		}
		if atomic.LoadUint32(&killed) != 0 {
			return
		}
		SGD(g, s, stepSize, 1, batchSize)
	}
}
