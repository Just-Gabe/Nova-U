package data

import "math/rand"

type Sample struct {
	Input  []int
	Target []int
}

type Dataset interface {
	Len() int
	Get(idx int) Sample
}

type SequenceDataset struct {
	Tokens   []int
	SeqLen   int
	Stride   int
}

func NewSequenceDataset(tokens []int, seqLen, stride int) *SequenceDataset {
	return &SequenceDataset{
		Tokens: tokens,
		SeqLen: seqLen,
		Stride: stride,
	}
}

func (d *SequenceDataset) Len() int {
	n := (len(d.Tokens) - d.SeqLen - 1) / d.Stride
	if n < 0 {
		return 0
	}
	return n
}

func (d *SequenceDataset) Get(idx int) Sample {
	start := idx * d.Stride
	input := make([]int, d.SeqLen)
	target := make([]int, d.SeqLen)
	for i := 0; i < d.SeqLen; i++ {
		input[i] = d.Tokens[start+i]
		target[i] = d.Tokens[start+i+1]
	}
	return Sample{Input: input, Target: target}
}

type TextFileDataset struct {
	Tokens   []int
	SeqLen   int
}

func NewTextFileDataset(tokens []int, seqLen int) *TextFileDataset {
	return &TextFileDataset{
		Tokens: tokens,
		SeqLen: seqLen,
	}
}

func (d *TextFileDataset) Len() int {
	return len(d.Tokens) / d.SeqLen
}

func (d *TextFileDataset) Get(idx int) Sample {
	start := idx * d.SeqLen
	end := start + d.SeqLen
	if end >= len(d.Tokens)-1 {
		end = len(d.Tokens) - 1
	}
	input := make([]int, d.SeqLen)
	target := make([]int, d.SeqLen)
	for i := 0; i < d.SeqLen; i++ {
		input[i] = d.Tokens[start+i]
		if start+i+1 < len(d.Tokens) {
			target[i] = d.Tokens[start+i+1]
		} else {
			target[i] = 0
		}
	}
	return Sample{Input: input, Target: target}
}

type ShuffleDataset struct {
	Dataset Dataset
	Indices []int
}

func NewShuffleDataset(d Dataset) *ShuffleDataset {
	n := d.Len()
	indices := rand.Perm(n)
	return &ShuffleDataset{Dataset: d, Indices: indices}
}

func (s *ShuffleDataset) Len() int { return s.Dataset.Len() }

func (s *ShuffleDataset) Get(idx int) Sample {
	return s.Dataset.Get(s.Indices[idx])
}

type MapDataset struct {
	Dataset Dataset
	Fn      func(Sample) Sample
}

func (m *MapDataset) Len() int { return m.Dataset.Len() }

func (m *MapDataset) Get(idx int) Sample {
	return m.Fn(m.Dataset.Get(idx))
}
