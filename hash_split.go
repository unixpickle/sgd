package sgd

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"math"
	"sort"

	"github.com/unixpickle/num-analysis/linalg"
)

// A Hasher can generate a binary hash of its contents.
// Hashes should be randomly distributed.
type Hasher interface {
	Hash() []byte
}

// HashVectors is a helper for hashing vector-valued
// training samples.
// It takes vectors and hashes them in a
// randomly-distributed manner.
func HashVectors(vecs ...linalg.Vector) []byte {
	var buf bytes.Buffer
	tempBuf := make([]byte, 8)
	for _, vec := range vecs {
		for _, x := range vec {
			binary.BigEndian.PutUint64(tempBuf, math.Float64bits(x))
			buf.Write(tempBuf)
		}
		buf.WriteByte(0)
	}
	res := md5.Sum(buf.Bytes())
	return res[:]
}

// HashSplit partitions a SampleSet whose samples all
// implement Hasher.
//
// The given sample set may be reordered, and the returned
// sample sets will be subsets of it.
//
// The leftRatio argument specifies the expected fraction
// of samples that should end up on the left partition.
func HashSplit(s SampleSet, leftRatio float64) (left, right SampleSet) {
	if leftRatio == 0 {
		return s.Subset(0, 0), s
	} else if leftRatio == 1 {
		return s, s.Subset(0, 0)
	}

	cutoff := hashCutoff(leftRatio)
	insertIdx := 0
	for i := 0; i < s.Len(); i++ {
		hash := s.GetSample(i).(Hasher).Hash()
		if compareHashes(hash, cutoff) < 0 {
			s.Swap(insertIdx, i)
			insertIdx++
		}
	}
	splitIdx := sort.Search(s.Len(), func(i int) bool {
		return compareHashes(s.GetSample(i).(Hasher).Hash(), cutoff) >= 0
	})
	return s.Subset(0, splitIdx), s.Subset(splitIdx, s.Len())
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
