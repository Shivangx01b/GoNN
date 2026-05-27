package optim

import (
	"math"

	"gonn/tensor"
)

// Adam implements the Adam optimizer (Kingma & Ba, 2014).
type Adam struct {
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

// AdamOption configures an Adam optimizer.
type AdamOption func(*Adam)

// WithBeta1 sets the beta1 coefficient.
func WithBeta1(b float64) AdamOption { return func(a *Adam) { a.beta1 = b } }

// WithBeta2 sets the beta2 coefficient.
func WithBeta2(b float64) AdamOption { return func(a *Adam) { a.beta2 = b } }

// WithAdamEps sets the epsilon term.
func WithAdamEps(e float64) AdamOption { return func(a *Adam) { a.eps = e } }

// WithAdamWeightDecay sets the L2 weight decay coefficient.
func WithAdamWeightDecay(wd float64) AdamOption { return func(a *Adam) { a.weightDecay = wd } }

// NewAdam constructs an Adam optimizer with defaults beta1=0.9, beta2=0.999, eps=1e-8.
func NewAdam(params []*tensor.Tensor, lr float64, opts ...AdamOption) *Adam {
	a := &Adam{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-8,
		m:      make(map[*tensor.Tensor][]float64),
		v:      make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single Adam update.
func (a *Adam) Step() {
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
			if a.weightDecay != 0 {
				g += a.weightDecay * data[i]
			}
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			data[i] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *Adam) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *Adam) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *Adam) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *Adam) SetLR(lr float64) { a.lr = lr }
