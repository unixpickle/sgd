package sgd

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"io"
	"math"
	"sort"

	"github.com/unixpickle/num-analysis/linalg"
)

// A Hasher is a SampleSet which can compute hashes of its
// constituent samples.
type Hasher interface {
	SampleSet
	Hash(i int) []byte
}

// HashVectors is a helper for hashing vector-valued
// training samples.
// It takes vectors and hashes them in a
// randomly-distributed manner.
func HashVectors(vecs ...linalg.Vector) []byte {
	var buf bytes.Buffer
	tempBuf := make([]byte, 8)

	var lastValue float64
	var valueCount byte
	for _, vec := range vecs {
		for _, x := range vec {
			if x == lastValue && valueCount < 0xff {
				valueCount++
			} else {
				if valueCount > 0 {
					buf.WriteByte(valueCount)
					writeFloatBits(&buf, tempBuf, lastValue)
				}
				lastValue = x
				valueCount = 1
			}
		}
		if valueCount > 0 {
			buf.WriteByte(valueCount)
			writeFloatBits(&buf, tempBuf, lastValue)
			valueCount = 0
		}
		// Separator between vectors.
		buf.WriteByte(0)
	}
	res := md5.Sum(buf.Bytes())
	return res[:]
}

// HashSplit partitions a Hasher.
//
// The given Hasder may be reordered, and the returned
// sample sets will be subsets of it.
//
// The leftRatio argument specifies the expected fraction
// of samples that should end up on the left partition.
func HashSplit(h Hasher, leftRatio float64) (left, right SampleSet) {
	if leftRatio == 0 {
		return h.Subset(0, 0), h
	} else if leftRatio == 1 {
		return h, h.Subset(0, 0)
	}
	cutoff := hashCutoff(leftRatio)
	insertIdx := 0
	for i := 0; i < h.Len(); i++ {
		hash := h.Hash(i)
		if compareHashes(hash, cutoff) < 0 {
			h.Swap(insertIdx, i)
			insertIdx++
		}
	}
	splitIdx := sort.Search(h.Len(), func(i int) bool {
		return compareHashes(h.Hash(i), cutoff) >= 0
	})
	return h.Subset(0, splitIdx), h.Subset(splitIdx, h.Len())
}

func hashCutoff(ratio float64) []byte {
	res := make([]byte, 8)
	for i := range res {
		ratio *= 256
		value := int(ratio)
		ratio -= float64(value)
		if value == 256 {
			value = 255
		}
		res[i] = byte(value)
	}
	return res
}

func compareHashes(h1, h2 []byte) int {
	max := len(h1)
	if len(h2) > max {
		max = len(h2)
	}
	for i := 0; i < max; i++ {
		var h1Val, h2Val byte
		if i < len(h1) {
			h1Val = h1[i]
		}
		if i < len(h2) {
			h2Val = h2[i]
		}
		if h1Val < h2Val {
			return -1
		} else if h1Val > h2Val {
			return 1
		}
	}
	return 0
}

func writeFloatBits(w io.Writer, temp []byte, val float64) {
	binary.BigEndian.PutUint64(temp, math.Float64bits(val))
	w.Write(temp)
}
