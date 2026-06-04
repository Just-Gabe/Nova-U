package tensor

import (
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"math/rand"
	"runtime"
	"sync"
)

type Tensor struct {
	Data    []float32 `json:"data"`
	Shape   []int     `json:"shape"`
	strides []int
}

func computeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = s
		s *= shape[i]
	}
	return strides
}

func New(data []float32, shape []int) *Tensor {
	return &Tensor{Data: data, Shape: shape, strides: computeStrides(shape)}
}

func Zeros(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return New(make([]float32, size), shape)
}

func Ones(shape []int) *Tensor {
	t := Zeros(shape)
	for i := range t.Data {
		t.Data[i] = 1
	}
	return t
}

func Random(shape []int, std float32) *Tensor {
	t := Zeros(shape)
	for i := range t.Data {
		t.Data[i] = float32(rand.NormFloat64()) * std
	}
	return t
}

// --- buffer pool ---

var bufPools sync.Map // map[int]*sync.Pool

func poolFor(n int) *sync.Pool {
	if v, ok := bufPools.Load(n); ok {
		return v.(*sync.Pool)
	}
	p := &sync.Pool{New: func() any { return make([]float32, n) }}
	actual, _ := bufPools.LoadOrStore(n, p)
	return actual.(*sync.Pool)
}

func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	return 1 << bits.Len(uint(n-1))
}

// GetBuf returns a zeroed []float32 of length >= n from the pool.
func GetBuf(n int) []float32 {
	cap := nextPow2(n)
	b := poolFor(cap).Get().([]float32)
	b = b[:n]
	for i := range b {
		b[i] = 0
	}
	return b
}

// PutBuf returns a buffer to the pool. The caller must ensure no further use.
func PutBuf(b []float32) {
	if b == nil {
		return
	}
	c := cap(b)
	if c == 0 || c != nextPow2(c) {
		return
	}
	poolFor(c).Put(b[:c])
}

// ZerosPooled allocates from the pool. Release with t.Release().
func ZerosPooled(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{Data: GetBuf(size), Shape: shape, strides: computeStrides(shape)}
}

func (t *Tensor) Release() {
	if t == nil {
		return
	}
	PutBuf(t.Data)
	t.Data = nil
}

func (t *Tensor) Stride(i int) int {
	return t.strides[i]
}

func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return New(data, t.Shape)
}

func (t *Tensor) At(indices ...int) float32 {
	if len(indices) != len(t.Shape) {
		panic("at: index count mismatch")
	}
	off := 0
	for i, idx := range indices {
		off += idx * t.strides[i]
	}
	return t.Data[off]
}

func (t *Tensor) Set(val float32, indices ...int) {
	if len(indices) != len(t.Shape) {
		panic("set: index count mismatch")
	}
	off := 0
	for i, idx := range indices {
		off += idx * t.strides[i]
	}
	t.Data[off] = val
}

func (t *Tensor) Reshape(shape []int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if size != len(t.Data) {
		panic(fmt.Sprintf("reshape: size mismatch %d != %d", size, len(t.Data)))
	}
	return New(t.Data, shape)
}

// --- parallel helper ---

const parallelThreshold = 16384

func parallelFor(n, perItem int, fn func(start, end int)) {
	work := n * perItem
	workers := runtime.GOMAXPROCS(0)
	if work < parallelThreshold || workers < 2 || n < workers {
		fn(0, n)
		return
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		if start >= n {
			break
		}
		end := start + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}

// --- elementwise ops ---

func Add(a, b *Tensor) *Tensor {
	if !sameShape(a, b) {
		panic("add: shape mismatch")
	}
	r := Zeros(a.Shape)
	for i := range a.Data {
		r.Data[i] = a.Data[i] + b.Data[i]
	}
	return r
}

func AddInto(dst, a, b *Tensor) {
	for i := range a.Data {
		dst.Data[i] = a.Data[i] + b.Data[i]
	}
}

func Sub(a, b *Tensor) *Tensor {
	if !sameShape(a, b) {
		panic("sub: shape mismatch")
	}
	r := Zeros(a.Shape)
	for i := range a.Data {
		r.Data[i] = a.Data[i] - b.Data[i]
	}
	return r
}

func Mul(a, b *Tensor) *Tensor {
	if !sameShape(a, b) {
		panic("mul: shape mismatch")
	}
	r := Zeros(a.Shape)
	for i := range a.Data {
		r.Data[i] = a.Data[i] * b.Data[i]
	}
	return r
}

func Scale(t *Tensor, s float32) *Tensor {
	r := Zeros(t.Shape)
	for i := range t.Data {
		r.Data[i] = t.Data[i] * s
	}
	return r
}

func AddScalar(t *Tensor, s float32) *Tensor {
	r := Zeros(t.Shape)
	for i := range t.Data {
		r.Data[i] = t.Data[i] + s
	}
	return r
}

// --- matmul ---

// Dot computes a [M,K] @ b [K,N] = r [M,N] with i-k-j loop ordering
// (sequential access along the inner N dim) and row-parallelism over M.
func Dot(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("dot: need 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("dot: shape mismatch %v x %v", a.Shape, b.Shape))
	}
	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	r := Zeros([]int{m, n})
	dotKernel(a.Data, b.Data, r.Data, m, k, n)
	return r
}

func dotKernel(aData, bData, rData []float32, m, k, n int) {
	parallelFor(m, k*n, func(start, end int) {
		for i := start; i < end; i++ {
			rRow := rData[i*n : i*n+n]
			aRow := aData[i*k : i*k+k]
			for p := 0; p < k; p++ {
				av := aRow[p]
				bRow := bData[p*n : p*n+n]
				for j := 0; j < n; j++ {
					rRow[j] += av * bRow[j]
				}
			}
		}
	})
}

// --- conv ---

// Conv1D implements 1D convolution via im2col + a single batched matmul.
// Input [B, Cin, L], kernel [Cout, Cin, K], bias [Cout]. Output [B, Cout, Lout].
//
// We pack im2col as [Cin*K, B*Lout] so the whole batch becomes one matmul:
//   tmp = kFlat [Cout, Cin*K] @ col [Cin*K, B*Lout]   -> [Cout, B*Lout]
// Then we transpose tmp into out [B, Cout, Lout] and add bias.
func Conv1D(inp, kernel, bias *Tensor, stride, padding int) *Tensor {
	b, ic, il := inp.Shape[0], inp.Shape[1], inp.Shape[2]
	oc, _, ks := kernel.Shape[0], kernel.Shape[1], kernel.Shape[2]
	ol := (il+2*padding-ks)/stride + 1
	out := Zeros([]int{b, oc, ol})

	kCols := ic * ks
	N := b * ol

	col := GetBuf(kCols * N)
	tmp := GetBuf(oc * N)
	defer PutBuf(col)
	defer PutBuf(tmp)

	// Build batched im2col: col[c, ba*ol + oli] = inp[ba, c/ks, oli*stride - padding + c%ks]
	im2col1DBatched(inp.Data, col, b, ic, il, ks, stride, padding, ol)

	// One matmul over the whole batch.
	dotAddKernel(kernel.Data, col, tmp, oc, kCols, N)

	// Untranspose into [B, Cout, Lout] and add bias.
	parallelFor(b, oc*ol, func(start, end int) {
		for ba := start; ba < end; ba++ {
			for oci := 0; oci < oc; oci++ {
				bv := bias.Data[oci]
				src := tmp[oci*N+ba*ol : oci*N+ba*ol+ol]
				dst := out.Data[ba*oc*ol+oci*ol : ba*oc*ol+oci*ol+ol]
				for j, v := range src {
					dst[j] = v + bv
				}
			}
		}
	})
	return out
}

func im2col1DBatched(inp, col []float32, b, ic, il, ks, stride, padding, ol int) {
	N := b * ol
	parallelFor(ic, ks*N, func(start, end int) {
		for ici := start; ici < end; ici++ {
			for ki := 0; ki < ks; ki++ {
				row := col[(ici*ks+ki)*N : (ici*ks+ki)*N+N]
				for ba := 0; ba < b; ba++ {
					base := ba * ol
					for oli := 0; oli < ol; oli++ {
						ili := oli*stride - padding + ki
						if ili >= 0 && ili < il {
							row[base+oli] = inp[ba*ic*il+ici*il+ili]
						} else {
							row[base+oli] = 0
						}
					}
				}
			}
		}
	})
}

// dotAddKernel: r += a [M,K] @ b [K,N]
func dotAddKernel(aData, bData, rData []float32, m, k, n int) {
	parallelFor(m, k*n, func(start, end int) {
		for i := start; i < end; i++ {
			rRow := rData[i*n : i*n+n]
			aRow := aData[i*k : i*k+k]
			for p := 0; p < k; p++ {
				av := aRow[p]
				bRow := bData[p*n : p*n+n]
				for j := 0; j < n; j++ {
					rRow[j] += av * bRow[j]
				}
			}
		}
	})
}

func Conv1DTranspose(inp, kernel, bias *Tensor, stride, padding int) *Tensor {
	b, ic, il := inp.Shape[0], inp.Shape[1], inp.Shape[2]
	_, oc, ks := kernel.Shape[0], kernel.Shape[1], kernel.Shape[2]
	ol := (il-1)*stride - 2*padding + ks
	r := Zeros([]int{b, oc, ol})

	for ba := 0; ba < b; ba++ {
		for oci := 0; oci < oc; oci++ {
			for ici := 0; ici < ic; ici++ {
				for ili := 0; ili < il; ili++ {
					v := inp.Data[ba*ic*il+ici*il+ili]
					for ki := 0; ki < ks; ki++ {
						oli := ili*stride + ki - padding
						if oli >= 0 && oli < ol {
							r.Data[ba*oc*ol+oci*ol+oli] +=
								v * kernel.Data[ici*oc*ks+oci*ks+ki]
						}
					}
				}
			}
		}
	}
	for ba := 0; ba < b; ba++ {
		for oci := 0; oci < oc; oci++ {
			bv := bias.Data[oci]
			row := r.Data[ba*oc*ol+oci*ol : ba*oc*ol+oci*ol+ol]
			for j := range row {
				row[j] += bv
			}
		}
	}
	return r
}

// --- pooling ---

func MaxPool1D(inp *Tensor, ks, stride int) *Tensor {
	b, c, il := inp.Shape[0], inp.Shape[1], inp.Shape[2]
	ol := (il - ks) / stride + 1
	r := Zeros([]int{b, c, ol})
	parallelFor(b, c*ol*ks, func(start, end int) {
		for ba := start; ba < end; ba++ {
			for ci := 0; ci < c; ci++ {
				inRow := inp.Data[ba*c*il+ci*il : ba*c*il+ci*il+il]
				outRow := r.Data[ba*c*ol+ci*ol : ba*c*ol+ci*ol+ol]
				for oli := 0; oli < ol; oli++ {
					mx := float32(-math.MaxFloat32)
					base := oli * stride
					for ki := 0; ki < ks; ki++ {
						v := inRow[base+ki]
						if v > mx {
							mx = v
						}
					}
					outRow[oli] = mx
				}
			}
		}
	})
	return r
}

func AvgPool1D(inp *Tensor, ks, stride int) *Tensor {
	b, c, il := inp.Shape[0], inp.Shape[1], inp.Shape[2]
	ol := (il - ks) / stride + 1
	r := Zeros([]int{b, c, ol})
	for ba := 0; ba < b; ba++ {
		for ci := 0; ci < c; ci++ {
			for oli := 0; oli < ol; oli++ {
				var sum float32
				for ki := 0; ki < ks; ki++ {
					sum += inp.Data[ba*c*il+ci*il+oli*stride+ki]
				}
				r.Data[ba*c*ol+ci*ol+oli] = sum / float32(ks)
			}
		}
	}
	return r
}

func Upsample1D(inp *Tensor, scale int) *Tensor {
	b, c, il := inp.Shape[0], inp.Shape[1], inp.Shape[2]
	ol := il * scale
	r := Zeros([]int{b, c, ol})
	for ba := 0; ba < b; ba++ {
		for ci := 0; ci < c; ci++ {
			inRow := inp.Data[ba*c*il+ci*il : ba*c*il+ci*il+il]
			outRow := r.Data[ba*c*ol+ci*ol : ba*c*ol+ci*ol+ol]
			for oli := 0; oli < ol; oli++ {
				outRow[oli] = inRow[oli/scale]
			}
		}
	}
	return r
}

// --- cat ---

func Cat(inputs []*Tensor, dim int) *Tensor {
	if len(inputs) == 0 {
		return nil
	}
	if len(inputs) == 1 {
		return inputs[0].Clone()
	}
	rank := len(inputs[0].Shape)

	rShape := make([]int, rank)
	copy(rShape, inputs[0].Shape)
	for _, inp := range inputs[1:] {
		rShape[dim] += inp.Shape[dim]
	}
	r := Zeros(rShape)

	if rank == 3 && dim == 1 {
		rB, rC, rL := r.Shape[0], r.Shape[1], r.Shape[2]
		offset := 0
		for _, inp := range inputs {
			inpC, inpL := inp.Shape[1], inp.Shape[2]
			useL := inpL
			if useL > rL {
				useL = rL
			}
			for ba := 0; ba < rB; ba++ {
				for ci := 0; ci < inpC; ci++ {
					src := inp.Data[ba*inpC*inpL+ci*inpL : ba*inpC*inpL+ci*inpL+useL]
					dstStart := ba*rC*rL + (offset+ci)*rL
					copy(r.Data[dstStart:dstStart+useL], src)
				}
			}
			offset += inpC
		}
		return r
	}

	// general fallback
	for _, inp := range inputs[1:] {
		if len(inp.Shape) != rank {
			panic("cat: rank mismatch")
		}
		for d := 0; d < rank; d++ {
			if d != dim && inp.Shape[d] != inputs[0].Shape[d] {
				panic("cat: shape mismatch on non-concat dimension")
			}
		}
	}
	offset := 0
	idx := make([]int, rank)
	for _, inp := range inputs {
		for i := 0; i < len(inp.Data); i++ {
			rem := i
			for d := rank - 1; d >= 0; d-- {
				idx[d] = rem % inp.Shape[d]
				rem /= inp.Shape[d]
			}
			dstOff := 0
			for d := 0; d < rank; d++ {
				v := idx[d]
				if d == dim {
					v += offset
				}
				dstOff += v * r.strides[d]
			}
			r.Data[dstOff] = inp.Data[i]
		}
		offset += inp.Shape[dim]
	}
	return r
}

// --- activations ---

func ReLU(t *Tensor) *Tensor {
	r := Zeros(t.Shape)
	for i, v := range t.Data {
		if v > 0 {
			r.Data[i] = v
		}
	}
	return r
}

func ReLUInto(t *Tensor) {
	for i, v := range t.Data {
		if v < 0 {
			t.Data[i] = 0
		}
	}
}

const (
	geluC1 = float32(0.7978845608028654) // sqrt(2/pi)
	geluC2 = float32(0.044715)
)

// GELU uses the tanh approximation (default of nn.GELU(approximate="tanh") in PyTorch).
func GELU(t *Tensor) *Tensor {
	r := Zeros(t.Shape)
	parallelFor(len(t.Data), 1, func(start, end int) {
		for i := start; i < end; i++ {
			x := t.Data[i]
			inner := geluC1 * (x + geluC2*x*x*x)
			r.Data[i] = 0.5 * x * (1 + fastTanh(inner))
		}
	})
	return r
}

// fastTanh is a Padé-style rational approximation accurate to ~1e-4 over typical activation ranges.
// Saturates correctly outside [-3, 3].
func fastTanh(x float32) float32 {
	if x > 4.97 {
		return 1
	}
	if x < -4.97 {
		return -1
	}
	x2 := x * x
	num := x * (135135 + x2*(17325+x2*(378+x2)))
	den := 135135 + x2*(62370+x2*(3150+x2*28))
	return num / den
}

func Sigmoid(t *Tensor) *Tensor {
	r := Zeros(t.Shape)
	for i := range t.Data {
		r.Data[i] = 1.0 / (1.0 + float32(math.Exp(-float64(t.Data[i]))))
	}
	return r
}

// Softmax over the last dimension.
func Softmax(t *Tensor) *Tensor {
	last := t.Shape[len(t.Shape)-1]
	bs := len(t.Data) / last
	r := Zeros(t.Shape)
	parallelFor(bs, last, func(start, end int) {
		for b := start; b < end; b++ {
			off := b * last
			mx := t.Data[off]
			for i := 1; i < last; i++ {
				if v := t.Data[off+i]; v > mx {
					mx = v
				}
			}
			var sum float32
			for i := 0; i < last; i++ {
				e := float32(math.Exp(float64(t.Data[off+i] - mx)))
				r.Data[off+i] = e
				sum += e
			}
			inv := 1 / sum
			for i := 0; i < last; i++ {
				r.Data[off+i] *= inv
			}
		}
	})
	return r
}

// LayerNorm normalizes along channel dim (dim 1 of [B, C, L]).
func LayerNorm(inp, weight, bias *Tensor, eps float32) *Tensor {
	b, c, l := inp.Shape[0], inp.Shape[1], inp.Shape[2]
	r := Zeros(inp.Shape)
	invC := 1.0 / float32(c)
	parallelFor(b, c*l, func(start, end int) {
		for ba := start; ba < end; ba++ {
			for li := 0; li < l; li++ {
				var mean, sqSum float32
				for ci := 0; ci < c; ci++ {
					v := inp.Data[ba*c*l+ci*l+li]
					mean += v
				}
				mean *= invC
				for ci := 0; ci < c; ci++ {
					v := inp.Data[ba*c*l+ci*l+li] - mean
					sqSum += v * v
				}
				variance := sqSum * invC
				invStd := 1.0 / float32(math.Sqrt(float64(variance+eps)))
				for ci := 0; ci < c; ci++ {
					v := (inp.Data[ba*c*l+ci*l+li] - mean) * invStd
					r.Data[ba*c*l+ci*l+li] = v*weight.Data[ci] + bias.Data[ci]
				}
			}
		}
	})
	return r
}

func CrossEntropyLoss(pred, target *Tensor) float32 {
	b, v, l := pred.Shape[0], pred.Shape[1], pred.Shape[2]
	var totalLoss float32
	var mu sync.Mutex
	parallelFor(b, v*l, func(start, end int) {
		var local float32
		for ba := start; ba < end; ba++ {
			for li := 0; li < l; li++ {
				t := int(target.Data[ba*l+li])
				mx := pred.Data[ba*v*l+0*l+li]
				for vi := 1; vi < v; vi++ {
					val := pred.Data[ba*v*l+vi*l+li]
					if val > mx {
						mx = val
					}
				}
				var sum float32
				for vi := 0; vi < v; vi++ {
					sum += float32(math.Exp(float64(pred.Data[ba*v*l+vi*l+li] - mx)))
				}
				logSum := float32(math.Log(float64(sum)))
				logSoftmax := pred.Data[ba*v*l+t*l+li] - mx - logSum
				local -= logSoftmax
			}
		}
		mu.Lock()
		totalLoss += local
		mu.Unlock()
	})
	return totalLoss / float32(b*l)
}

// --- positional encoding (cached) ---

type posKey struct{ seqLen, dModel int }

var (
	posCache   = map[posKey]*Tensor{}
	posCacheMu sync.RWMutex
)

func PositionalEncoding(seqLen, dModel int) *Tensor {
	k := posKey{seqLen, dModel}
	posCacheMu.RLock()
	if pe, ok := posCache[k]; ok {
		posCacheMu.RUnlock()
		return pe
	}
	posCacheMu.RUnlock()

	pe := Zeros([]int{1, dModel, seqLen})
	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < dModel; i++ {
			angle := float64(pos) / math.Pow(10000, float64(i)/float64(dModel))
			if i%2 == 0 {
				pe.Data[i*seqLen+pos] = float32(math.Sin(angle))
			} else {
				pe.Data[i*seqLen+pos] = float32(math.Cos(angle))
			}
		}
	}

	posCacheMu.Lock()
	posCache[k] = pe
	posCacheMu.Unlock()
	return pe
}

// --- JSON serialization ---

func (t *Tensor) MarshalJSON() ([]byte, error) {
	w := struct {
		Data  []float32 `json:"data"`
		Shape []int     `json:"shape"`
	}{t.Data, t.Shape}
	return json.Marshal(w)
}

func (t *Tensor) UnmarshalJSON(b []byte) error {
	var w struct {
		Data  []float32 `json:"data"`
		Shape []int     `json:"shape"`
	}
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	t.Data = w.Data
	t.Shape = w.Shape
	t.strides = computeStrides(t.Shape)
	return nil
}

func sameShape(a, b *Tensor) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}
