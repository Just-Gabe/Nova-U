package model

import (
	"testing"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

func smallUNetConfig() UNetConfig {
	cfg := DefaultUNetConfig()
	cfg.VocabSize = 256
	cfg.DModel = 64
	cfg.NumLevels = 2
	cfg.MaxSeqLen = 64
	return cfg
}

func TestUNetForwardShape(t *testing.T) {
	cfg := smallUNetConfig()
	u := NewUNet1D(cfg)
	x := tensor.Zeros([]int{2, 64})
	y := u.Forward(x)
	if y.Shape[0] != 2 || y.Shape[1] != cfg.VocabSize || y.Shape[2] != 64 {
		t.Fatalf("forward shape: got %v want [2 %d 64]", y.Shape, cfg.VocabSize)
	}
}

func BenchmarkUNetForward(b *testing.B) {
	cfg := smallUNetConfig()
	u := NewUNet1D(cfg)
	x := tensor.Zeros([]int{4, 64})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = u.Forward(x)
	}
}
