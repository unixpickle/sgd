// +build js

package sgd

func loopUntilKilled(sf func() bool, tf func()) {
	for {
		if !sf() {
			return
		}
		tf()
	}
}
