package data

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sort"
)

type Transform func(Sample) Sample

type Compose struct {
	Transforms []Transform
}

func (c Compose) Apply(s Sample) Sample {
	for _, t := range c.Transforms {
		s = t(s)
	}
	return s
}

func ToFloat32(in []int) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	VocabSize() int
}

type SimpleTokenizer struct {
	CharToID map[rune]int
	IDToChar map[int]rune
}

// NewSimpleTokenizer builds a char-level tokenizer over the union of all
// characters seen in `texts`. Reserved IDs 0/1/2 are PAD/BOS/EOS.
// Characters are assigned in sorted order so the vocab is deterministic.
func NewSimpleTokenizer(texts []string) *SimpleTokenizer {
	seen := make(map[rune]struct{})
	for _, t := range texts {
		for _, r := range t {
			seen[r] = struct{}{}
		}
	}
	chars := make([]rune, 0, len(seen))
	for r := range seen {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	charToID := map[rune]int{}
	idToChar := map[int]rune{}
	nextID := 3 // reserve 0/1/2
	for _, r := range chars {
		charToID[r] = nextID
		idToChar[nextID] = r
		nextID++
	}
	return &SimpleTokenizer{CharToID: charToID, IDToChar: idToChar}
}

// Encode returns one int per rune (not per byte). Unknown characters map to PAD (0).
func (t *SimpleTokenizer) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		if id, ok := t.CharToID[r]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, 0)
		}
	}
	return ids
}

// Decode skips reserved control IDs (0/1/2 = PAD/BOS/EOS) and unknown IDs.
func (t *SimpleTokenizer) Decode(tokens []int) string {
	runes := make([]rune, 0, len(tokens))
	for _, id := range tokens {
		if id < 3 {
			continue
		}
		if r, ok := t.IDToChar[id]; ok {
			runes = append(runes, r)
		}
	}
	return string(runes)
}

func (t *SimpleTokenizer) VocabSize() int {
	// +3 for reserved PAD/BOS/EOS slots that aren't in CharToID.
	return len(t.CharToID) + 3
}

// --- persistence ---

type vocabFile struct {
	Version int            `json:"version"`
	Chars   map[string]int `json:"chars"`
}

func (t *SimpleTokenizer) Save(path string) error {
	chars := make(map[string]int, len(t.CharToID))
	for r, id := range t.CharToID {
		chars[string(r)] = id
	}
	data, err := json.MarshalIndent(vocabFile{Version: 1, Chars: chars}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func LoadSimpleTokenizer(path string) (*SimpleTokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	var v vocabFile
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("parse vocab: %w", err)
	}
	if v.Version != 1 {
		return nil, fmt.Errorf("unsupported vocab version: %d", v.Version)
	}
	t := &SimpleTokenizer{
		CharToID: make(map[rune]int, len(v.Chars)),
		IDToChar: make(map[int]rune, len(v.Chars)),
	}
	for s, id := range v.Chars {
		for _, r := range s {
			t.CharToID[r] = id
			t.IDToChar[id] = r
			break
		}
	}
	return t, nil
}

type BPETokenizer struct {
	Vocab    map[string]int
	Reverse  map[int]string
	VocabSiz int
}

func NewBPETokenizer(vocabSize int) *BPETokenizer {
	return &BPETokenizer{
		Vocab:    make(map[string]int),
		Reverse:  make(map[int]string),
		VocabSiz: vocabSize,
	}
}

func RandomCrop(sample Sample, cropLen int) Sample {
	if len(sample.Input) <= cropLen {
		return sample
	}
	start := rand.Intn(len(sample.Input) - cropLen)
	return Sample{
		Input:  sample.Input[start : start+cropLen],
		Target: sample.Target[start : start+cropLen],
	}
}
