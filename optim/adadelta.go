package optim

import (
	"math"

	"gonn/tensor"
)

// Adadelta implements the Adadelta optimizer (Zeiler, 2012).
type Adadelta struct {
	params      []*tensor.Tensor
	lr          float64
	rho         float64
	eps         float64
	weightDecay float64
	squareAvg   map[*tensor.Tensor][]float64
	accDelta    map[*tensor.Tensor][]float64
}

// AdadeltaOption configures an Adadelta optimizer.
type AdadeltaOption func(*Adadelta)

// WithAdadeltaRho sets the decay rate rho.
func WithAdadeltaRho(r float64) AdadeltaOption { return func(a *Adadelta) { a.rho = r } }

// WithAdadeltaEps sets the epsilon term.
func WithAdadeltaEps(e float64) AdadeltaOption { return func(a *Adadelta) { a.eps = e } }

// WithAdadeltaWeightDecay sets the L2 weight decay coefficient.
func WithAdadeltaWeightDecay(wd float64) AdadeltaOption {
	return func(a *Adadelta) { a.weightDecay = wd }
}

// NewAdadelta constructs an Adadelta optimizer with defaults rho=0.9, eps=1e-6, lr=1.0.
func NewAdadelta(params []*tensor.Tensor, lr float64, opts ...AdadeltaOption) *Adadelta {
	a := &Adadelta{
		params:    params,
		lr:        lr,
		rho:       0.9,
		eps:       1e-6,
		squareAvg: make(map[*tensor.Tensor][]float64),
		accDelta:  make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single Adadelta update.
func (a *Adadelta) Step() {
	for _, p := range a.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		sa := a.squareAvg[p]
		if sa == nil {
			sa = make([]float64, len(data))
			a.squareAvg[p] = sa
		}
		ad := a.accDelta[p]
		if ad == nil {
			ad = make([]float64, len(data))
			a.accDelta[p] = ad
		}
		for i := range data {
			g := grad[i]
			if a.weightDecay != 0 {
				g += a.weightDecay * data[i]
			}
			sa[i] = a.rho*sa[i] + (1-a.rho)*g*g
			delta := math.Sqrt(ad[i]+a.eps) / math.Sqrt(sa[i]+a.eps) * g
			ad[i] = a.rho*ad[i] + (1-a.rho)*delta*delta
			data[i] -= a.lr * delta
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *Adadelta) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *Adadelta) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *Adadelta) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *Adadelta) SetLR(lr float64) { a.lr = lr }
