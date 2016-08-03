package sgd

import "math/rand"

// A SampleSet is an abstract list of abstract training
// samples.
type SampleSet interface {
	// Len returns the length of the sample set.
	Len() int

	// Copy creates a shallow copy of the sample set.
	// The training samples themselves needn't be
	// copied.
	Copy() SampleSet

	// Swap swaps the samples at two indices.
	Swap(i, j int)

	// Get gets the sample at the given index.
	GetSample(idx int) interface{}

	// Subset creates a SampleSet which represents the
	// subset of this sample set from the start index
	// (inclusive) to the end index (exclusive).
	//
	// The resulting sample set is only valid as long
	// as elements of the receiver are not manipulated
	// via Swap(). This is similar to the regular Go
	// slice behavior, wherein a sub-slice of a slice
	// references the same backing array.
	Subset(start, end int) SampleSet
}

func ShuffleSampleSet(s SampleSet) {
	for i := 0; i < s.Len(); i++ {
		j := i + rand.Intn(s.Len()-i)
		s.Swap(i, j)
	}
}

// SliceSampleSet is a SampleSet which is backed by a
// slice of training samples.
type SliceSampleSet []interface{}

func (s SliceSampleSet) Len() int {
	return len(s)
}

func (s SliceSampleSet) Copy() SampleSet {
	res := make(SliceSampleSet, len(s))
	copy(res, s)
	return s
}

func (s SliceSampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SliceSampleSet) GetSample(idx int) interface{} {
	return s[idx]
}

func (s SliceSampleSet) Subset(start, end int) SampleSet {
	return s[start:end]
}
