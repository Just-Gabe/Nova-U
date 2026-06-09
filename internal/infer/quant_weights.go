package infer

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

type QuantizedEntry struct {
	Name      string
	Shape     []int
	GroupSize int
	NumBits   int
	Packed    []uint8
	Scales    []float32
	Zeros     []float32
}

func SaveWeightsQuantized(path string, entries []QuantizedEntry) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write([]byte("QNOV")); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(1)); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(len(entries))); err != nil {
		return err
	}

	for _, e := range entries {
		nameBytes := []byte(e.Name)
		if len(nameBytes) > 0xFFFF {
			return fmt.Errorf("name too long: %s", e.Name)
		}
		if err := binary.Write(f, binary.LittleEndian, uint16(len(nameBytes))); err != nil {
			return err
		}
		if _, err := f.Write(nameBytes); err != nil {
			return err
		}

		if len(e.Shape) > 255 {
			return fmt.Errorf("too many dims for %s", e.Name)
		}
		if err := binary.Write(f, binary.LittleEndian, uint8(len(e.Shape))); err != nil {
			return err
		}
		for _, d := range e.Shape {
			if err := binary.Write(f, binary.LittleEndian, int32(d)); err != nil {
				return err
			}
		}

		if err := binary.Write(f, binary.LittleEndian, uint16(e.GroupSize)); err != nil {
			return err
		}
		if err := binary.Write(f, binary.LittleEndian, uint8(e.NumBits)); err != nil {
			return err
		}

		totalElems := 1
		for _, d := range e.Shape {
			totalElems *= d
		}
		if err := binary.Write(f, binary.LittleEndian, int32(totalElems)); err != nil {
			return err
		}

		packedSize := tensor.QuantizedSize(totalElems, e.NumBits)
		if err := binary.Write(f, binary.LittleEndian, int32(packedSize)); err != nil {
			return err
		}

		if _, err := f.Write(e.Packed); err != nil {
			return err
		}

		nGroups := tensor.NumGroups(totalElems, e.GroupSize)
		for i := 0; i < nGroups; i++ {
			if err := binary.Write(f, binary.LittleEndian, e.Scales[i]); err != nil {
				return err
			}
		}
		for i := 0; i < nGroups; i++ {
			if err := binary.Write(f, binary.LittleEndian, e.Zeros[i]); err != nil {
				return err
			}
		}
	}

	return nil
}

func LoadWeightsQuantized(path string) ([]QuantizedEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, err
	}
	if string(magic) != "QNOV" {
		return nil, fmt.Errorf("bad magic: %q (want QNOV)", magic)
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != 1 {
		return nil, fmt.Errorf("unsupported QNOVA version: %d", version)
	}

	var nEntries uint32
	if err := binary.Read(f, binary.LittleEndian, &nEntries); err != nil {
		return nil, err
	}

	entries := make([]QuantizedEntry, nEntries)
	for i := range entries {
		var nameLen uint16
		if err := binary.Read(f, binary.LittleEndian, &nameLen); err != nil {
			return nil, err
		}
		nameBytes := make([]byte, nameLen)
		if _, err := io.ReadFull(f, nameBytes); err != nil {
			return nil, err
		}
		entries[i].Name = string(nameBytes)

		var nDims uint8
		if err := binary.Read(f, binary.LittleEndian, &nDims); err != nil {
			return nil, err
		}
		entries[i].Shape = make([]int, nDims)
		for j := range entries[i].Shape {
			var d int32
			if err := binary.Read(f, binary.LittleEndian, &d); err != nil {
				return nil, err
			}
			entries[i].Shape[j] = int(d)
		}

		var groupSize uint16
		if err := binary.Read(f, binary.LittleEndian, &groupSize); err != nil {
			return nil, err
		}
		entries[i].GroupSize = int(groupSize)

		var numBits uint8
		if err := binary.Read(f, binary.LittleEndian, &numBits); err != nil {
			return nil, err
		}
		entries[i].NumBits = int(numBits)

		var totalElems int32
		if err := binary.Read(f, binary.LittleEndian, &totalElems); err != nil {
			return nil, err
		}

		var packedSize int32
		if err := binary.Read(f, binary.LittleEndian, &packedSize); err != nil {
			return nil, err
		}

		entries[i].Packed = make([]uint8, packedSize)
		if _, err := io.ReadFull(f, entries[i].Packed); err != nil {
			return nil, err
		}

		nGroups := tensor.NumGroups(int(totalElems), entries[i].GroupSize)
		entries[i].Scales = make([]float32, nGroups)
		entries[i].Zeros = make([]float32, nGroups)

		for j := 0; j < nGroups; j++ {
			if err := binary.Read(f, binary.LittleEndian, &entries[i].Scales[j]); err != nil {
				return nil, err
			}
		}
		for j := 0; j < nGroups; j++ {
			if err := binary.Read(f, binary.LittleEndian, &entries[i].Zeros[j]); err != nil {
				return nil, err
			}
		}
	}

	return entries, nil
}

func DequantizeEntry(e *QuantizedEntry) *tensor.Tensor {
	totalElems := 1
	for _, d := range e.Shape {
		totalElems *= d
	}

	var indices []uint8
	switch e.NumBits {
	case 4:
		indices = tensor.Unpack4bit(e.Packed, totalElems)
	case 2:
		indices = tensor.Unpack2bit(e.Packed, totalElems)
	default:
		indices = e.Packed
	}

	data := tensor.GroupDequantize(indices, e.Scales, e.Zeros, e.GroupSize, e.NumBits)
	return tensor.New(data, e.Shape)
}
