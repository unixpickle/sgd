package sgd

import (
	"bytes"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

type hashTestSample struct {
	hash []byte
}

func (h *hashTestSample) Hash() []byte {
	return h.hash
}

func TestHashSplit(t *testing.T) {
	sampleSet := SliceSampleSet{
		&hashTestSample{hash: []byte{0xac}},
		&hashTestSample{hash: []byte{0x0}},
		&hashTestSample{hash: []byte{0x50}},
		&hashTestSample{hash: []byte{0xdc}},
		&hashTestSample{hash: []byte{0x99}},
	}
	left, right := HashSplit(sampleSet, 0.5)
	if left.Len() != 2 || right.Len() != 3 {
		t.Fatalf("invalid lengths %d,%d (expected 2,3)", left.Len(), right.Len())
	}
	leftValues := map[string]bool{"\x00": true, "\x50": true}
	rightValues := map[string]bool{"\x99": true, "\xac": true, "\xdc": true}
	for _, v := range left.(SliceSampleSet) {
		h := string(v.(Hasher).Hash())
		if !leftValues[h] {
			t.Errorf("unexpected left hash: %v", []byte(h))
		} else {
			leftValues[h] = false
		}
	}
	for _, v := range right.(SliceSampleSet) {
		h := string(v.(Hasher).Hash())
		if !rightValues[h] {
			t.Errorf("unexpected right hash: %v", []byte(h))
		} else {
			rightValues[h] = false
		}
	}
}

func TestHashVectors(t *testing.T) {
	vecs1 := []linalg.Vector{{1, 2, 3}, {4, 5, 6}}
	vecs2 := []linalg.Vector{{1, 2}, {3, 4, 5, 6}}
	vecs3 := []linalg.Vector{{1, 2, 2}, {3, 4, 5, 6}}
	hash1 := HashVectors(vecs1...)
	hash2 := HashVectors(vecs2...)
	hash3 := HashVectors(vecs3...)
	hash4 := HashVectors(vecs1...)
	if !bytes.Equal(hash1, hash4) {
		t.Error("inconsistent hashes")
	}
	if bytes.Equal(hash1, hash2) {
		t.Error("hash collision")
	}
	if bytes.Equal(hash2, hash3) {
		t.Error("hash collision")
	}
}
