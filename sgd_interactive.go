// +build !js

package sgd

import (
	"fmt"
	"os"
	"os/signal"
	"sync/atomic"
)

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
