package asyncsgd

import (
	"bytes"
	"encoding/binary"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

var byteOrder = binary.BigEndian

// SerializeVectors serializes the vectors as binary data.
func SerializeVectors(vecs []linalg.Vector) []byte {
	var res bytes.Buffer
	binary.Write(&res, byteOrder, uint64(len(vecs)))
	for _, x := range vecs {
		binary.Write(&res, byteOrder, uint64(len(x)))
		for _, val := range x {
			binary.Write(&res, byteOrder, val)
		}
	}
	return res.Bytes()
}

// DeserializeVectors decodes serialized vectors.
func DeserializeVectors(d []byte) ([]linalg.Vector, error) {
	r := bytes.NewReader(d)
	var vecCount uint64
	if err := binary.Read(r, byteOrder, &vecCount); err != nil {
		return nil, serializer.ErrBufferUnderflow
	}
	if int(vecCount)*8 > r.Len() {
		return nil, serializer.ErrBufferUnderflow
	}
	res := make([]linalg.Vector, int(vecCount))
	for i := range res {
		var valCount uint64
		if err := binary.Read(r, byteOrder, &valCount); err != nil {
			return nil, serializer.ErrBufferUnderflow
		}
		if int(valCount) > r.Len()*8 {
			return nil, serializer.ErrBufferUnderflow
		}
		res[i] = make(linalg.Vector, int(valCount))
		for j := range res[i] {
			if err := binary.Read(r, byteOrder, &res[i][j]); err != nil {
				return nil, serializer.ErrBufferUnderflow
			}
		}
	}
	return res, nil
}
