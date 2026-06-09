package tensor

import "math"

type PackedWeights struct {
	Indices  []uint8
	Scales   []float32
	Zeros    []float32
	Shape    []int
	GroupSize int
	NumBits   int
}

func NumGroups(n, groupSize int) int {
	return (n + groupSize - 1) / groupSize
}

func GroupQuantize(data []float32, groupSize, numBits int) (indices []uint8, scales, zeros []float32) {
	n := len(data)
	nGroups := NumGroups(n, groupSize)
	levels := 1 << numBits

	indices = make([]uint8, n)
	scales = make([]float32, nGroups)
	zeros = make([]float32, nGroups)

	for g := 0; g < nGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > n {
			end = n
		}

		minVal := float32(math.Inf(1))
		maxVal := float32(math.Inf(-1))
		for i := start; i < end; i++ {
			if data[i] < minVal {
				minVal = data[i]
			}
			if data[i] > maxVal {
				maxVal = data[i]
			}
		}

		scale := (maxVal - minVal) / float32(levels-1)
		if scale == 0 {
			scale = 1.0
		}

		scales[g] = scale
		zeros[g] = minVal

		for i := start; i < end; i++ {
			q := (data[i] - minVal) / scale
			idx := int(math.Round(float64(q)))
			if idx < 0 {
				idx = 0
			}
			if idx >= levels {
				idx = levels - 1
			}
			indices[i] = uint8(idx)
		}
	}

	return indices, scales, zeros
}

func GroupDequantize(indices []uint8, scales, zeros []float32, groupSize, numBits int) []float32 {
	n := len(indices)
	out := make([]float32, n)
	nGroups := NumGroups(n, groupSize)

	for g := 0; g < nGroups; g++ {
		start := g * groupSize
		end := start + groupSize
		if end > n {
			end = n
		}
		scale := scales[g]
		zero := zeros[g]

		for i := start; i < end; i++ {
			out[i] = float32(indices[i])*scale + zero
		}
	}

	return out
}

func Pack4bit(indices []uint8) []uint8 {
	n := len(indices)
	padded := (n + 1) / 2
	packed := make([]uint8, padded)
	for i := 0; i < n; i += 2 {
		b := indices[i] & 0x0F
		if i+1 < n {
			b |= (indices[i+1] & 0x0F) << 4
		}
		packed[i/2] = b
	}
	return packed
}

func Unpack4bit(packed []uint8, n int) []uint8 {
	indices := make([]uint8, n)
	for i := 0; i < n; i += 2 {
		b := packed[i/2]
		indices[i] = b & 0x0F
		if i+1 < n {
			indices[i+1] = (b >> 4) & 0x0F
		}
	}
	return indices
}

func Pack2bit(indices []uint8) []uint8 {
	n := len(indices)
	padded := (n + 3) / 4
	packed := make([]uint8, padded)
	for i := 0; i < n; i += 4 {
		b := uint8(0)
		for j := 0; j < 4 && i+j < n; j++ {
			b |= (indices[i+j] & 0x03) << (2 * j)
		}
		packed[i/4] = b
	}
	return packed
}

func Unpack2bit(packed []uint8, n int) []uint8 {
	indices := make([]uint8, n)
	for i := 0; i < n; i += 4 {
		b := packed[i/4]
		for j := 0; j < 4 && i+j < n; j++ {
			indices[i+j] = (b >> (2 * j)) & 0x03
		}
	}
	return indices
}

func QuantizedSize(n int, numBits int) int {
	switch numBits {
	case 2:
		return (n + 3) / 4
	case 4:
		return (n + 1) / 2
	default:
		return n
	}
}
