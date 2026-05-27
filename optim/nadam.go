package optim

import (
	"math"

	"gonn/tensor"
)

// NAdam implements Adam with Nesterov momentum, using the PyTorch-style
// schedule for the momentum decay product mu_t.
type NAdam struct {
	params         []*tensor.Tensor
	lr             float64
	beta1          float64
	beta2          float64
	eps            float64
	weightDecay    float64
	momentumDecay  float64
	m              map[*tensor.Tensor][]float64
	v              map[*tensor.Tensor][]float64
	t              int
	muProduct      float64
}

// NAdamOption configures a NAdam optimizer.
type NAdamOption func(*NAdam)

// WithNAdamBeta1 sets the beta1 coefficient.
func WithNAdamBeta1(b float64) NAdamOption { return func(a *NAdam) { a.beta1 = b } }

// WithNAdamBeta2 sets the beta2 coefficient.
func WithNAdamBeta2(b float64) NAdamOption { return func(a *NAdam) { a.beta2 = b } }

// WithNAdamEps sets the epsilon term.
func WithNAdamEps(e float64) NAdamOption { return func(a *NAdam) { a.eps = e } }

// WithNAdamWeightDecay sets the L2 weight decay coefficient.
func WithNAdamWeightDecay(wd float64) NAdamOption { return func(a *NAdam) { a.weightDecay = wd } }

// WithNAdamMomentumDecay sets the momentum decay coefficient.
func WithNAdamMomentumDecay(d float64) NAdamOption { return func(a *NAdam) { a.momentumDecay = d } }

// NewNAdam constructs a NAdam optimizer with PyTorch defaults
// beta1=0.9, beta2=0.999, eps=1e-8, momentumDecay=4e-3.
func NewNAdam(params []*tensor.Tensor, lr float64, opts ...NAdamOption) *NAdam {
	a := &NAdam{
		params:        params,
		lr:            lr,
		beta1:         0.9,
		beta2:         0.999,
		eps:           1e-8,
		momentumDecay: 4e-3,
		m:             make(map[*tensor.Tensor][]float64),
		v:             make(map[*tensor.Tensor][]float64),
		muProduct:     1,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single NAdam update.
func (a *NAdam) Step() {
	a.t++
	t := float64(a.t)
	muT := a.beta1 * (1 - 0.5*math.Pow(0.96, t*a.momentumDecay))
	muTNext := a.beta1 * (1 - 0.5*math.Pow(0.96, (t+1)*a.momentumDecay))
	a.muProduct *= muT
	muProductNext := a.muProduct * muTNext
	bc2 := 1 - math.Pow(a.beta2, t)
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
			mHat := muTNext*m[i]/(1-muProductNext) + (1-muT)*g/(1-a.muProduct)
			vHat := v[i] / bc2
			data[i] -= a.lr * mHat / (math.Sqrt(vHat) + a.eps)
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *NAdam) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *NAdam) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *NAdam) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *NAdam) SetLR(lr float64) { a.lr = lr }
