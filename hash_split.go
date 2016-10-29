package sgd

import (
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
	sum := md5.New()

	destBytes := make([]byte, 8)
	for _, vec := range vecs {
		for _, x := range vec {
			bits := math.Float64bits(x)
			binary.BigEndian.PutUint64(destBytes, bits)
			sum.Write(destBytes)
		}
		sum.Write([]byte{0})
	}

	return sum.Sum(nil)
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

	sorter := &hashSorter{set: s}
	sort.Sort(sorter)

	cutoff := hashCutoff(leftRatio)
	splitIdx := sort.Search(s.Len(), func(i int) bool {
		return compareHashes(s.GetSample(i).(Hasher).Hash(), cutoff) >= 0
	})
	return s.Subset(0, splitIdx), s.Subset(splitIdx, s.Len())
}

type hashSorter struct {
	set SampleSet
}

func (h *hashSorter) Len() int {
	return h.set.Len()
}

func (h *hashSorter) Swap(i, j int) {
	h.set.Swap(i, j)
}

func (h *hashSorter) Less(i, j int) bool {
	hash1 := h.set.GetSample(i).(Hasher).Hash()
	hash2 := h.set.GetSample(j).(Hasher).Hash()
	return compareHashes(hash1, hash2) < 0
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
