package model

import (
	"container/heap"
	"math"
	"math/rand"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

type SamplingMode int

const (
	Greedy  SamplingMode = 0
	TopK    SamplingMode = 1
	Sampling SamplingMode = 2
)

type ARConfig struct {
	UNetConfig    UNetConfig
	MaxGenLen     int
	ContextLen    int
	SamplingMode  SamplingMode
	Temperature   float32
	TopK          int
	BOSID         int
	EOSID         int
	PadID         int
}

func DefaultARConfig() ARConfig {
	return ARConfig{
		UNetConfig:   DefaultUNetConfig(),
		MaxGenLen:    256,
		SamplingMode: TopK,
		Temperature:  0.8,
		TopK:         40,
		BOSID:        1,
		EOSID:        2,
		PadID:        0,
	}
}

type ARModel struct {
	Config ARConfig
	UNet   *UNet1D
}

func NewARModel(cfg ARConfig) *ARModel {
	return &ARModel{
		Config: cfg,
		UNet:   NewUNet1D(cfg.UNetConfig),
	}
}

func (m *ARModel) Forward(x *tensor.Tensor) *tensor.Tensor {
	return m.UNet.Forward(x)
}

func (m *ARModel) Generate(context []int, maxSteps int) []int {
	if maxSteps <= 0 {
		maxSteps = m.Config.MaxGenLen
	}

	seq := make([]int, len(context)+1)
	copy(seq, context)
	seq[len(context)] = m.Config.BOSID

	gran := 1 << uint(m.Config.UNetConfig.NumLevels) // 2^levels
	ctxLen := m.Config.ContextLen
	if ctxLen <= 0 {
		ctxLen = m.Config.UNetConfig.MaxSeqLen
	}
	maxLen := m.Config.UNetConfig.MaxSeqLen

	for step := 0; step < maxSteps; step++ {
		if len(seq) > ctxLen {
			copy(seq, seq[len(seq)-ctxLen:])
			seq = seq[:ctxLen]
		}

		blockSize := len(seq)
		if blockSize > maxLen {
			blockSize = maxLen
		}
		inp := seq[len(seq)-blockSize:]

		// Pad on the LEFT with PadID so the most recent token is at the last
		// position (where we read logits from). This avoids letting the model
		// learn anything from a non-existing right context.
		padded := padLeft(inp, gran, m.Config.PadID, maxLen)

		inpTensor := tensor.New(intsToFloats(padded), []int{1, len(padded)})
		logits := m.Forward(inpTensor)

		lastPosLogits := extractLastPos(logits)
		nextTok := sample(lastPosLogits, m.Config)

		seq = append(seq, nextTok)
		if nextTok == m.Config.EOSID {
			break
		}
	}
	return seq
}

// padLeft returns inp padded on the left with padID up to the smallest multiple
// of granularity that is >= len(inp). Clamped to maxLen.
func padLeft(inp []int, granularity, padID, maxLen int) []int {
	target := ((len(inp) + granularity - 1) / granularity) * granularity
	if target > maxLen {
		target = maxLen
	}
	if target == len(inp) {
		return inp
	}
	if target < len(inp) {
		// shouldn't happen given maxLen >= granularity, but clamp
		return inp[len(inp)-target:]
	}
	out := make([]int, target)
	pad := target - len(inp)
	for i := 0; i < pad; i++ {
		out[i] = padID
	}
	copy(out[pad:], inp)
	return out
}

func (m *ARModel) Params() map[string]*tensor.Tensor {
	return m.UNet.Params()
}

func (m *ARModel) SetParams(params map[string]*tensor.Tensor) {
	m.UNet.SetParams(params)
}

func extractLastPos(logits *tensor.Tensor) []float32 {
	v, l := logits.Shape[1], logits.Shape[2]
	lastL := l - 1
	result := make([]float32, v)
	stride1 := logits.Stride(1)
	for vi := 0; vi < v; vi++ {
		result[vi] = logits.Data[vi*stride1+lastL]
	}
	return result
}

func sample(logits []float32, cfg ARConfig) int {
	if cfg.Temperature > 0 && cfg.Temperature != 1.0 {
		for i := range logits {
			logits[i] /= cfg.Temperature
		}
	}

	if cfg.SamplingMode == Greedy {
		return argmax(logits)
	}

	if cfg.SamplingMode == TopK {
		return topKSample(logits, cfg.TopK)
	}

	return categoricalSample(logits)
}

func argmax(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// topKItem is an entry in the min-heap.
type topKItem struct {
	val float32
	idx int
}

type topKHeap []topKItem

func (h topKHeap) Len() int            { return len(h) }
func (h topKHeap) Less(i, j int) bool  { return h[i].val < h[j].val }
func (h topKHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *topKHeap) Push(x any)         { *h = append(*h, x.(topKItem)) }
func (h *topKHeap) Pop() any {
	n := len(*h)
	x := (*h)[n-1]
	*h = (*h)[:n-1]
	return x
}

// topKSample keeps the k highest logits via a size-k min-heap (O(V log k)).
func topKSample(logits []float32, k int) int {
	n := len(logits)
	if k > n {
		k = n
	}
	h := make(topKHeap, 0, k)
	for i, v := range logits {
		if len(h) < k {
			heap.Push(&h, topKItem{v, i})
		} else if v > h[0].val {
			h[0] = topKItem{v, i}
			heap.Fix(&h, 0)
		}
	}

	maxVal := h[0].val
	for i := 1; i < len(h); i++ {
		if h[i].val > maxVal {
			maxVal = h[i].val
		}
	}
	probs := make([]float32, len(h))
	var sum float32
	for i := range h {
		probs[i] = float32(math.Exp(float64(h[i].val - maxVal)))
		sum += probs[i]
	}
	r := rand.Float32() * sum
	var cum float32
	for i := range h {
		cum += probs[i]
		if r <= cum {
			return h[i].idx
		}
	}
	return h[len(h)-1].idx
}

func categoricalSample(logits []float32) int {
	var maxVal float32 = logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}
	var sum float32
	probs := make([]float32, len(logits))
	for i := range logits {
		probs[i] = float32(math.Exp(float64(logits[i] - maxVal)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	r := rand.Float32()
	var cum float32
	for i := 0; i < len(probs); i++ {
		cum += probs[i]
		if r <= cum {
			return i
		}
	}
	return len(probs) - 1
}

func intsToFloats(v []int) []float32 {
	r := make([]float32, len(v))
	for i, x := range v {
		r[i] = float32(x)
	}
	return r
}
