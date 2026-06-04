package tensor

import (
	"testing"
)

func TestDotShape(t *testing.T) {
	a := Random([]int{4, 8}, 0.1)
	b := Random([]int{8, 6}, 0.1)
	c := Dot(a, b)
	if c.Shape[0] != 4 || c.Shape[1] != 6 {
		t.Fatalf("dot shape: got %v want [4 6]", c.Shape)
	}
}

func TestConv1DShape(t *testing.T) {
	x := Random([]int{2, 4, 16}, 0.1)
	k := Random([]int{8, 4, 3}, 0.1)
	b := Zeros([]int{8})
	y := Conv1D(x, k, b, 1, 1)
	if y.Shape[0] != 2 || y.Shape[1] != 8 || y.Shape[2] != 16 {
		t.Fatalf("conv1d shape: got %v want [2 8 16]", y.Shape)
	}
}

func TestCatRoundtrip(t *testing.T) {
	a := Random([]int{1, 3, 4}, 0.1)
	b := Random([]int{1, 5, 4}, 0.1)
	c := Cat([]*Tensor{a, b}, 1)
	if c.Shape[1] != 8 {
		t.Fatalf("cat dim: got %d want 8", c.Shape[1])
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if c.At(0, i, j) != a.At(0, i, j) {
				t.Fatalf("cat preserved a")
			}
		}
	}
	for i := 0; i < 5; i++ {
		for j := 0; j < 4; j++ {
			if c.At(0, 3+i, j) != b.At(0, i, j) {
				t.Fatalf("cat preserved b")
			}
		}
	}
}

func TestSoftmaxSumToOne(t *testing.T) {
	x := Random([]int{2, 16}, 1.0)
	y := Softmax(x)
	for b := 0; b < 2; b++ {
		var sum float32
		for i := 0; i < 16; i++ {
			sum += y.Data[b*16+i]
		}
		if sum < 0.99 || sum > 1.01 {
			t.Fatalf("softmax row %d sum=%f", b, sum)
		}
	}
}

func BenchmarkDot(b *testing.B) {
	a := Random([]int{128, 256}, 0.1)
	w := Random([]int{256, 256}, 0.1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Dot(a, w)
	}
}

func BenchmarkDotLarge(b *testing.B) {
	a := Random([]int{512, 512}, 0.1)
	w := Random([]int{512, 512}, 0.1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Dot(a, w)
	}
}

func BenchmarkConv1D(b *testing.B) {
	x := Random([]int{8, 128, 128}, 0.1)
	k := Random([]int{128, 128, 3}, 0.1)
	bs := Zeros([]int{128})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Conv1D(x, k, bs, 1, 1)
	}
}

func BenchmarkLayerNorm(b *testing.B) {
	x := Random([]int{8, 128, 128}, 0.1)
	w := Ones([]int{128})
	bs := Zeros([]int{128})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = LayerNorm(x, w, bs, 1e-5)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	x := Random([]int{8, 1000}, 1.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(x)
	}
}

func BenchmarkGELU(b *testing.B) {
	x := Random([]int{8, 128, 128}, 1.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GELU(x)
	}
}
