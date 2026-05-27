package optim

import (
	"math"

	"gonn/tensor"
)

// AdamW implements Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
type AdamW struct {
	params      []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	m           map[*tensor.Tensor][]float64
	v           map[*tensor.Tensor][]float64
	t           int
}

// AdamWOption configures an AdamW optimizer.
type AdamWOption func(*AdamW)

// WithAdamWBeta1 sets the beta1 coefficient.
func WithAdamWBeta1(b float64) AdamWOption { return func(a *AdamW) { a.beta1 = b } }

// WithAdamWBeta2 sets the beta2 coefficient.
func WithAdamWBeta2(b float64) AdamWOption { return func(a *AdamW) { a.beta2 = b } }

// WithAdamWEps sets the epsilon term.
func WithAdamWEps(e float64) AdamWOption { return func(a *AdamW) { a.eps = e } }

// WithAdamWWeightDecay sets the decoupled weight decay coefficient.
func WithAdamWWeightDecay(wd float64) AdamWOption { return func(a *AdamW) { a.weightDecay = wd } }

// NewAdamW constructs an AdamW optimizer with defaults beta1=0.9, beta2=0.999, eps=1e-8, weightDecay=0.01.
func NewAdamW(params []*tensor.Tensor, lr float64, opts ...AdamWOption) *AdamW {
	a := &AdamW{
		params:      params,
		lr:          lr,
		beta1:       0.9,
		beta2:       0.999,
		eps:         1e-8,
		weightDecay: 0.01,
		m:           make(map[*tensor.Tensor][]float64),
		v:           make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single AdamW update.
func (a *AdamW) Step() {
	a.t++
	bc1 := 1 - math.Pow(a.beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.beta2, float64(a.t))
	for _, p := range a.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		m := a.m[p]
		if m == nil {
			m = make([]float64, len(data))
			a.m[p] = m
		}
		v := a.v[p]
		if v == nil {
			v = make([]float64, len(data))
			a.v[p] = v
		}
		for i := range data {
			g := grad[i]
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			// Decoupled weight decay: applied directly to the parameter.
			if a.weightDecay != 0 {
				data[i] -= a.lr * a.weightDecay * data[i]
			}
			data[i] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *AdamW) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *AdamW) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *AdamW) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *AdamW) SetLR(lr float64) { a.lr = lr }
