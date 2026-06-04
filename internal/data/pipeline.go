package data

import (
	"math/rand"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

type Batch struct {
	Input  *tensor.Tensor
	Target *tensor.Tensor
}

type DataLoader struct {
	Dataset    Dataset
	BatchSize  int
	Shuffle    bool
	epoch      int
	indices    []int
	position   int
}

func NewDataLoader(d Dataset, batchSize int, shuffle bool) *DataLoader {
	n := d.Len()
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	return &DataLoader{
		Dataset:   d,
		BatchSize: batchSize,
		Shuffle:   shuffle,
		indices:   indices,
	}
}

func (dl *DataLoader) Reset() {
	dl.position = 0
	if dl.Shuffle {
		for i := len(dl.indices) - 1; i > 0; i-- {
			j := rand.Intn(i + 1)
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		}
	}
}

func (dl *DataLoader) NextBatch() (*Batch, bool) {
	if dl.position >= len(dl.indices) {
		return nil, false
	}

	end := dl.position + dl.BatchSize
	if end > len(dl.indices) {
		end = len(dl.indices)
	}
	batchSize := end - dl.position

	var firstSample Sample
	for i := dl.position; i < end; i++ {
		s := dl.Dataset.Get(dl.indices[i])
		if i == dl.position {
			firstSample = s
		}
	}

	seqLen := len(firstSample.Input)
	inputData := make([]float32, batchSize*seqLen)
	targetData := make([]float32, batchSize*seqLen)

	idx := 0
	for i := dl.position; i < end; i++ {
		s := dl.Dataset.Get(dl.indices[i])
		for j, v := range s.Input {
			inputData[idx*seqLen+j] = float32(v)
		}
		for j, v := range s.Target {
			targetData[idx*seqLen+j] = float32(v)
		}
		idx++
	}

	dl.position = end

	input := tensor.New(inputData, []int{batchSize, seqLen})
	target := tensor.New(targetData, []int{batchSize, seqLen})

	return &Batch{Input: input, Target: target}, true
}

func (dl *DataLoader) Iter() chan *Batch {
	ch := make(chan *Batch)
	go func() {
		dl.Reset()
		for {
			batch, ok := dl.NextBatch()
			if !ok {
				break
			}
			ch <- batch
		}
		close(ch)
	}()
	return ch
}
