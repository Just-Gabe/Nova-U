package model

import (
	"fmt"
	"math"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

type Layer interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Params() map[string]*tensor.Tensor
	SetParams(m map[string]*tensor.Tensor, prefix string)
}

type Sequential struct {
	Layers []Layer
}

func Seq(layers ...Layer) *Sequential {
	return &Sequential{Layers: layers}
}

func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

func (s *Sequential) Params() map[string]*tensor.Tensor {
	m := make(map[string]*tensor.Tensor)
	for i, l := range s.Layers {
		for k, v := range l.Params() {
			m[fmt.Sprintf("%d.%s", i, k)] = v
		}
	}
	return m
}

func (s *Sequential) SetParams(m map[string]*tensor.Tensor, prefix string) {
	for i, l := range s.Layers {
		l.SetParams(m, fmt.Sprintf("%s%d.", prefix, i))
	}
}

type Embedding struct {
	Weight *tensor.Tensor
}

func NewEmbedding(vocabSize, dModel int) *Embedding {
	return &Embedding{
		Weight: tensor.Random([]int{vocabSize, dModel}, 0.02),
	}
}

func (e *Embedding) Forward(x *tensor.Tensor) *tensor.Tensor {
	b, l := x.Shape[0], x.Shape[1]
	dModel := e.Weight.Shape[1]
	out := tensor.Zeros([]int{b, dModel, l})
	xStride0 := x.Stride(0)
	wStride0 := e.Weight.Stride(0)
	outStride0 := out.Stride(0)
	outStride1 := out.Stride(1)
	for ba := 0; ba < b; ba++ {
		for li := 0; li < l; li++ {
			tok := int(x.Data[ba*xStride0+li])
			for d := 0; d < dModel; d++ {
				out.Data[ba*outStride0+d*outStride1+li] = e.Weight.Data[tok*wStride0+d]
			}
		}
	}
	return out
}

func (e *Embedding) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{"weight": e.Weight}
}

func (e *Embedding) SetParams(m map[string]*tensor.Tensor, prefix string) {
	if v, ok := m[prefix+"weight"]; ok {
		e.Weight = v
	}
}

type Conv1DLayer struct {
	Weight   *tensor.Tensor
	Bias     *tensor.Tensor
	Stride   int
	Padding  int
}

func NewConv1D(inCh, outCh, kernelSize, stride, padding int) *Conv1DLayer {
	k := math.Sqrt(1.0 / float64(inCh*kernelSize))
	return &Conv1DLayer{
		Weight:  tensor.Random([]int{outCh, inCh, kernelSize}, float32(k)),
		Bias:    tensor.Zeros([]int{outCh}),
		Stride:  stride,
		Padding: padding,
	}
}

func (c *Conv1DLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Conv1D(x, c.Weight, c.Bias, c.Stride, c.Padding)
}

func (c *Conv1DLayer) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{"weight": c.Weight, "bias": c.Bias}
}

func (c *Conv1DLayer) SetParams(m map[string]*tensor.Tensor, prefix string) {
	if v, ok := m[prefix+"weight"]; ok {
		c.Weight = v
	}
	if v, ok := m[prefix+"bias"]; ok {
		c.Bias = v
	}
}

type Conv1DTransposeLayer struct {
	Weight   *tensor.Tensor
	Bias     *tensor.Tensor
	Stride   int
	Padding  int
}

func NewConv1DTranspose(inCh, outCh, kernelSize, stride, padding int) *Conv1DTransposeLayer {
	k := math.Sqrt(1.0 / float64(inCh*kernelSize))
	return &Conv1DTransposeLayer{
		Weight:  tensor.Random([]int{inCh, outCh, kernelSize}, float32(k)),
		Bias:    tensor.Zeros([]int{outCh}),
		Stride:  stride,
		Padding: padding,
	}
}

func (c *Conv1DTransposeLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Conv1DTranspose(x, c.Weight, c.Bias, c.Stride, c.Padding)
}

func (c *Conv1DTransposeLayer) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{"weight": c.Weight, "bias": c.Bias}
}

func (c *Conv1DTransposeLayer) SetParams(m map[string]*tensor.Tensor, prefix string) {
	if v, ok := m[prefix+"weight"]; ok {
		c.Weight = v
	}
	if v, ok := m[prefix+"bias"]; ok {
		c.Bias = v
	}
}

type MaxPool1DLayer struct {
	KernelSize int
	Stride     int
}

func NewMaxPool1D(k, s int) *MaxPool1DLayer {
	return &MaxPool1DLayer{KernelSize: k, Stride: s}
}

func (m *MaxPool1DLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.MaxPool1D(x, m.KernelSize, m.Stride)
}

func (m *MaxPool1DLayer) Params() map[string]*tensor.Tensor { return nil }
func (m *MaxPool1DLayer) SetParams(_ map[string]*tensor.Tensor, _ string) {}

type UpsampleLayer struct {
	Scale int
}

func NewUpsample(scale int) *UpsampleLayer {
	return &UpsampleLayer{Scale: scale}
}

func (u *UpsampleLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Upsample1D(x, u.Scale)
}

func (u *UpsampleLayer) Params() map[string]*tensor.Tensor { return nil }
func (u *UpsampleLayer) SetParams(_ map[string]*tensor.Tensor, _ string) {}

type ReLULayer struct{}

func NewReLU() *ReLULayer { return &ReLULayer{} }

func (r *ReLULayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.ReLU(x)
}

func (r *ReLULayer) Params() map[string]*tensor.Tensor { return nil }
func (r *ReLULayer) SetParams(_ map[string]*tensor.Tensor, _ string) {}

type LayerNormLayer struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
	Eps    float32
}

func NewLayerNorm(channels int, eps float32) *LayerNormLayer {
	return &LayerNormLayer{
		Weight: tensor.Ones([]int{channels}),
		Bias:   tensor.Zeros([]int{channels}),
		Eps:    eps,
	}
}

func (l *LayerNormLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.LayerNorm(x, l.Weight, l.Bias, l.Eps)
}

func (l *LayerNormLayer) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{"weight": l.Weight, "bias": l.Bias}
}

func (l *LayerNormLayer) SetParams(m map[string]*tensor.Tensor, prefix string) {
	if v, ok := m[prefix+"weight"]; ok {
		l.Weight = v
	}
	if v, ok := m[prefix+"bias"]; ok {
		l.Bias = v
	}
}

type LinearLayer struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
}

func NewLinear(inFeat, outFeat int) *LinearLayer {
	k := math.Sqrt(1.0 / float64(inFeat))
	return &LinearLayer{
		Weight: tensor.Random([]int{outFeat, inFeat}, float32(k)),
		Bias:   tensor.Zeros([]int{outFeat}),
	}
}

func (l *LinearLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Dot(x, l.Weight)
}

func (l *LinearLayer) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{"weight": l.Weight, "bias": l.Bias}
}

func (l *LinearLayer) SetParams(m map[string]*tensor.Tensor, prefix string) {
	if v, ok := m[prefix+"weight"]; ok {
		l.Weight = v
	}
	if v, ok := m[prefix+"bias"]; ok {
		l.Bias = v
	}
}
