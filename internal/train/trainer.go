// Package train holds training-side infrastructure for Nova-U.
//
// Real backpropagation is not yet implemented in Go — production training
// happens via scripts/train_colab.py. The pieces below (Config, Optimizer)
// are kept so that, once autograd lands, a Trainer can plug them in directly.
package train

import (
	"math"

	"github.com/justgabe/Nova-U/pkg/tensor"
)

// Optimizer implements AdamW. Step expects a `grads` map keyed by the same
// parameter names returned by Params() on the model.
type Optimizer struct {
	Params      map[string]*tensor.Tensor
	LR          float32
	WeightDecay float32
	Steps       int
	M           map[string][]float32
	V           map[string][]float32
}

func NewOptimizer(params map[string]*tensor.Tensor, lr, wd float32) *Optimizer {
	m := make(map[string][]float32, len(params))
	v := make(map[string][]float32, len(params))
	for name, p := range params {
		m[name] = make([]float32, len(p.Data))
		v[name] = make([]float32, len(p.Data))
	}
	return &Optimizer{Params: params, LR: lr, WeightDecay: wd, M: m, V: v}
}

func (opt *Optimizer) ZeroGrad() {}

func (opt *Optimizer) Step(grads map[string]*tensor.Tensor) {
	opt.Steps++
	const (
		beta1 = float32(0.9)
		beta2 = float32(0.999)
		eps   = float32(1e-8)
	)
	bc1 := 1 - float32(math.Pow(float64(beta1), float64(opt.Steps)))
	bc2 := 1 - float32(math.Pow(float64(beta2), float64(opt.Steps)))
	invBC1 := 1 / bc1
	invBC2 := 1 / bc2
	wd := opt.WeightDecay
	lr := opt.LR

	for name, param := range opt.Params {
		grad, ok := grads[name]
		if !ok {
			continue
		}
		m := opt.M[name]
		v := opt.V[name]
		pd := param.Data
		gd := grad.Data
		for i := range pd {
			g := gd[i] + wd*pd[i]
			mi := beta1*m[i] + (1-beta1)*g
			vi := beta2*v[i] + (1-beta2)*g*g
			m[i] = mi
			v[i] = vi
			mHat := mi * invBC1
			vHat := vi * invBC2
			pd[i] -= lr * mHat / (float32(math.Sqrt(float64(vHat))) + eps)
		}
	}
}
