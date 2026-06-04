package model

import (
	"fmt"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

type UNetConfig struct {
	VocabSize    int
	DModel       int
	NumLevels    int
	KernelSize   int
	UpsampleMode string
	UseLayerNorm bool
	NormEps      float32
	MaxSeqLen    int
}

func DefaultUNetConfig() UNetConfig {
	return UNetConfig{
		VocabSize:    1000,
		DModel:       128,
		NumLevels:    3,
		KernelSize:   3,
		UpsampleMode: "nearest",
		UseLayerNorm: true,
		NormEps:      1e-5,
		MaxSeqLen:    512,
	}
}

type DoubleConv struct {
	Conv1 *Conv1DLayer
	Norm1 *LayerNormLayer
	Conv2 *Conv1DLayer
	Norm2 *LayerNormLayer
}

func NewDoubleConv(inCh, outCh, kSize int, useNorm bool, eps float32) *DoubleConv {
	pad := kSize / 2
	dc := &DoubleConv{
		Conv1: NewConv1D(inCh, outCh, kSize, 1, pad),
		Conv2: NewConv1D(outCh, outCh, kSize, 1, pad),
	}
	if useNorm {
		dc.Norm1 = NewLayerNorm(outCh, eps)
		dc.Norm2 = NewLayerNorm(outCh, eps)
	}
	return dc
}

func (d *DoubleConv) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = d.Conv1.Forward(x)
	if d.Norm1 != nil {
		x = d.Norm1.Forward(x)
	}
	x = tensor.ReLU(x)
	x = d.Conv2.Forward(x)
	if d.Norm2 != nil {
		x = d.Norm2.Forward(x)
	}
	x = tensor.ReLU(x)
	return x
}

func (d *DoubleConv) Params() map[string]*tensor.Tensor {
	m := make(map[string]*tensor.Tensor)
	for k, v := range d.Conv1.Params() {
		m["conv1."+k] = v
	}
	for k, v := range d.Conv2.Params() {
		m["conv2."+k] = v
	}
	if d.Norm1 != nil {
		for k, v := range d.Norm1.Params() {
			m["norm1."+k] = v
		}
	}
	if d.Norm2 != nil {
		for k, v := range d.Norm2.Params() {
			m["norm2."+k] = v
		}
	}
	return m
}

func (d *DoubleConv) SetParams(m map[string]*tensor.Tensor, prefix string) {
	d.Conv1.SetParams(m, prefix+"conv1.")
	d.Conv2.SetParams(m, prefix+"conv2.")
	if d.Norm1 != nil {
		d.Norm1.SetParams(m, prefix+"norm1.")
	}
	if d.Norm2 != nil {
		d.Norm2.SetParams(m, prefix+"norm2.")
	}
}

type DownBlock struct {
	Conv *DoubleConv
	Pool *MaxPool1DLayer
}

func NewDownBlock(inCh, outCh, kSize int, useNorm bool, eps float32) *DownBlock {
	return &DownBlock{
		Conv: NewDoubleConv(inCh, outCh, kSize, useNorm, eps),
		Pool: NewMaxPool1D(2, 2),
	}
}

func (d *DownBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	x = d.Conv.Forward(x)
	skip := x.Clone()
	x = d.Pool.Forward(x)
	return x, skip
}

func (d *DownBlock) Params() map[string]*tensor.Tensor {
	m := make(map[string]*tensor.Tensor)
	for k, v := range d.Conv.Params() {
		m["conv."+k] = v
	}
	return m
}

func (d *DownBlock) SetParams(m map[string]*tensor.Tensor, prefix string) {
	d.Conv.SetParams(m, prefix+"conv.")
}

type UpBlock struct {
	UpSample *UpsampleLayer
	Conv     *DoubleConv
}

func NewUpBlock(catCh, outCh, kSize int, useNorm bool, eps float32) *UpBlock {
	return &UpBlock{
		UpSample: NewUpsample(2),
		Conv:     NewDoubleConv(catCh, outCh, kSize, useNorm, eps),
	}
}

func (u *UpBlock) Forward(x *tensor.Tensor, skip *tensor.Tensor) *tensor.Tensor {
	x = u.UpSample.Forward(x)
	x = tensor.Cat([]*tensor.Tensor{skip, x}, 1)
	x = u.Conv.Forward(x)
	return x
}

func (u *UpBlock) Params() map[string]*tensor.Tensor {
	m := make(map[string]*tensor.Tensor)
	for k, v := range u.Conv.Params() {
		m["conv."+k] = v
	}
	return m
}

func (u *UpBlock) SetParams(m map[string]*tensor.Tensor, prefix string) {
	u.Conv.SetParams(m, prefix+"conv.")
}

type UNet1D struct {
	Config     UNetConfig
	Embed      *Embedding
	PosEncoder *tensor.Tensor
	InConv     *DoubleConv
	DownBlocks []*DownBlock
	Bottleneck *DoubleConv
	UpBlocks   []*UpBlock
	OutConv    *Conv1DLayer
}

func NewUNet1D(cfg UNetConfig) *UNet1D {
	posEnc := tensor.PositionalEncoding(cfg.MaxSeqLen, cfg.DModel)
	ch := cfg.DModel
	k := cfg.KernelSize
	useNorm := cfg.UseLayerNorm
	eps := cfg.NormEps

	downs := make([]*DownBlock, cfg.NumLevels)
	ups := make([]*UpBlock, cfg.NumLevels)

	inCh := ch
	for i := 0; i < cfg.NumLevels; i++ {
		outCh := ch * (1 << uint(i+1))
		downs[i] = NewDownBlock(inCh, outCh, k, useNorm, eps)
		inCh = outCh
	}

	bottleneck := NewDoubleConv(inCh, inCh, k, useNorm, eps)

	for i := 0; i < cfg.NumLevels; i++ {
		level := cfg.NumLevels - 1 - i
		upInCh := ch * (1 << uint(level+1))
		catCh := 2 * upInCh
		outCh := ch * (1 << uint(level))
		ups[i] = NewUpBlock(catCh, outCh, k, useNorm, eps)
	}

	return &UNet1D{
		Config:     cfg,
		Embed:      NewEmbedding(cfg.VocabSize, cfg.DModel),
		PosEncoder: posEnc,
		InConv:     NewDoubleConv(cfg.DModel, cfg.DModel, k, useNorm, eps),
		DownBlocks: downs,
		Bottleneck: bottleneck,
		UpBlocks:   ups,
		OutConv:    NewConv1D(ch, cfg.VocabSize, 1, 1, 0),
	}
}

func (u *UNet1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = u.Embed.Forward(x)
	seqLen := x.Shape[2]
	dModel := u.Config.DModel
	peData := u.PosEncoder.Data[:dModel*seqLen]
	pe := tensor.New(peData, []int{1, dModel, seqLen})

	batchSize := x.Shape[0]
	peExp := tensor.Zeros([]int{batchSize, dModel, seqLen})
	peStride := pe.Stride(0)
	peExpStride := peExp.Stride(0)
	for b := 0; b < batchSize; b++ {
		copy(peExp.Data[b*peExpStride:(b+1)*peExpStride], pe.Data[:peStride])
	}
	x = tensor.Add(x, peExp)

	x = u.InConv.Forward(x)

	skips := make([]*tensor.Tensor, len(u.DownBlocks))
	for i, down := range u.DownBlocks {
		var skip *tensor.Tensor
		x, skip = down.Forward(x)
		skips[i] = skip
	}

	x = u.Bottleneck.Forward(x)

	for i, up := range u.UpBlocks {
		skip := skips[len(skips)-1-i]
		x = up.Forward(x, skip)
	}

	x = u.OutConv.Forward(x)
	return x
}

func (u *UNet1D) Params() map[string]*tensor.Tensor {
	m := make(map[string]*tensor.Tensor)
	for k, v := range u.Embed.Params() {
		m["embed."+k] = v
	}
	m["pos_encoder"] = u.PosEncoder
	for k, v := range u.InConv.Params() {
		m["in_conv."+k] = v
	}
	for i, d := range u.DownBlocks {
		for k, v := range d.Params() {
			m[fmt.Sprintf("down.%d.%s", i, k)] = v
		}
	}
	for k, v := range u.Bottleneck.Params() {
		m["bottleneck."+k] = v
	}
	for i, up := range u.UpBlocks {
		for k, v := range up.Params() {
			m[fmt.Sprintf("up.%d.%s", i, k)] = v
		}
	}
	for k, v := range u.OutConv.Params() {
		m["out_conv."+k] = v
	}
	return m
}

func (u *UNet1D) SetParams(m map[string]*tensor.Tensor) {
	u.Embed.SetParams(m, "embed.")
	if v, ok := m["pos_encoder"]; ok {
		u.PosEncoder = v
	}
	u.InConv.SetParams(m, "in_conv.")
	for i, d := range u.DownBlocks {
		d.SetParams(m, fmt.Sprintf("down.%d.", i))
	}
	u.Bottleneck.SetParams(m, "bottleneck.")
	for i, up := range u.UpBlocks {
		up.SetParams(m, fmt.Sprintf("up.%d.", i))
	}
	u.OutConv.SetParams(m, "out_conv.")
}
