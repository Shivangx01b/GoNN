package optim

import (
	"math"

	"gonn/tensor"
)

// Adagrad implements the Adagrad optimizer (Duchi et al., 2011).
type Adagrad struct {
	params      []*tensor.Tensor
	lr          float64
	lrDecay     float64
	weightDecay float64
	eps         float64
	sum         map[*tensor.Tensor][]float64
	t           int
}

// AdagradOption configures an Adagrad optimizer.
type AdagradOption func(*Adagrad)

// WithAdagradLRDecay sets the learning-rate decay applied per step.
func WithAdagradLRDecay(d float64) AdagradOption { return func(a *Adagrad) { a.lrDecay = d } }

// WithAdagradWeightDecay sets the L2 weight decay coefficient.
func WithAdagradWeightDecay(wd float64) AdagradOption {
	return func(a *Adagrad) { a.weightDecay = wd }
}

// WithAdagradEps sets the epsilon term.
func WithAdagradEps(e float64) AdagradOption { return func(a *Adagrad) { a.eps = e } }

// NewAdagrad constructs an Adagrad optimizer with default eps=1e-10.
func NewAdagrad(params []*tensor.Tensor, lr float64, opts ...AdagradOption) *Adagrad {
	a := &Adagrad{
		params: params,
		lr:     lr,
		eps:    1e-10,
		sum:    make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single Adagrad update.
func (a *Adagrad) Step() {
	a.t++
	clr := a.lr / (1 + float64(a.t-1)*a.lrDecay)
	for _, p := range a.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		sum := a.sum[p]
		if sum == nil {
			sum = make([]float64, len(data))
			a.sum[p] = sum
		}
		for i := range data {
			g := grad[i]
			if a.weightDecay != 0 {
				g += a.weightDecay * data[i]
			}
			sum[i] += g * g
			data[i] -= clr * g / (math.Sqrt(sum[i]) + a.eps)
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *Adagrad) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *Adagrad) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current (base) learning rate.
func (a *Adagrad) LR() float64 { return a.lr }

// SetLR updates the (base) learning rate.
func (a *Adagrad) SetLR(lr float64) { a.lr = lr }
