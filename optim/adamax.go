package optim

import (
	"math"

	"gonn/tensor"
)

// Adamax implements the Adamax optimizer (Kingma & Ba, 2014), a variant of
// Adam based on the infinity norm.
type Adamax struct {
	params      []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	m           map[*tensor.Tensor][]float64
	u           map[*tensor.Tensor][]float64
	t           int
}

// AdamaxOption configures an Adamax optimizer.
type AdamaxOption func(*Adamax)

// WithAdamaxBeta1 sets the beta1 coefficient.
func WithAdamaxBeta1(b float64) AdamaxOption { return func(a *Adamax) { a.beta1 = b } }

// WithAdamaxBeta2 sets the beta2 coefficient.
func WithAdamaxBeta2(b float64) AdamaxOption { return func(a *Adamax) { a.beta2 = b } }

// WithAdamaxEps sets the epsilon term.
func WithAdamaxEps(e float64) AdamaxOption { return func(a *Adamax) { a.eps = e } }

// WithAdamaxWeightDecay sets the L2 weight decay coefficient.
func WithAdamaxWeightDecay(wd float64) AdamaxOption { return func(a *Adamax) { a.weightDecay = wd } }

// NewAdamax constructs an Adamax optimizer with defaults beta1=0.9,
// beta2=0.999, eps=1e-8, weight_decay=0.
func NewAdamax(params []*tensor.Tensor, lr float64, opts ...AdamaxOption) *Adamax {
	a := &Adamax{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-8,
		m:      make(map[*tensor.Tensor][]float64),
		u:      make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single Adamax update.
func (a *Adamax) Step() {
	a.t++
	bc1 := 1 - math.Pow(a.beta1, float64(a.t))
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
		u := a.u[p]
		if u == nil {
			u = make([]float64, len(data))
			a.u[p] = u
		}
		for i := range data {
			g := grad[i]
			if a.weightDecay != 0 {
				g += a.weightDecay * data[i]
			}
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			absG := math.Abs(g)
			scaled := a.beta2 * u[i]
			if absG > scaled {
				u[i] = absG
			} else {
				u[i] = scaled
			}
			data[i] -= (a.lr / bc1) * m[i] / (u[i] + a.eps)
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *Adamax) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *Adamax) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *Adamax) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *Adamax) SetLR(lr float64) { a.lr = lr }
