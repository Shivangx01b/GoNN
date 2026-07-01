package optim

import (
	"math"

	"gonn/tensor"
)

// Adadelta implements the Adadelta optimizer (Zeiler, 2012).
type Adadelta struct {
	baseOptimizer
	rho float64
	eps float64
}

// AdadeltaOption configures an Adadelta optimizer.
type AdadeltaOption func(*Adadelta)

// WithAdadeltaRho sets the decay rate rho.
func WithAdadeltaRho(r float64) AdadeltaOption { return func(a *Adadelta) { a.rho = r } }

// WithAdadeltaEps sets the epsilon term.
func WithAdadeltaEps(e float64) AdadeltaOption { return func(a *Adadelta) { a.eps = e } }

// WithAdadeltaWeightDecay sets the L2 weight decay coefficient on every group.
// With NewAdadeltaGroups, prefer setting Group.WeightDecay directly.
func WithAdadeltaWeightDecay(wd float64) AdadeltaOption {
	return func(a *Adadelta) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// NewAdadelta constructs an Adadelta optimizer with defaults rho=0.9, eps=1e-6, lr=1.0.
func NewAdadelta(params []*tensor.Tensor, lr float64, opts ...AdadeltaOption) *Adadelta {
	return NewAdadeltaGroups(singleGroup(params, lr), opts...)
}

// NewAdadeltaGroups constructs an Adadelta optimizer over explicit parameter groups.
func NewAdadeltaGroups(groups []Group, opts ...AdadeltaOption) *Adadelta {
	a := &Adadelta{
		baseOptimizer: newBase(groups),
		rho:           0.9,
		eps:           1e-6,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single Adadelta update.
func (a *Adadelta) Step() {
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		sa := st.Buf("square_avg", len(data))
		ad := st.Buf("acc_delta", len(data))
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			sa[i] = a.rho*sa[i] + (1-a.rho)*g*g
			delta := math.Sqrt(ad[i]+a.eps) / math.Sqrt(sa[i]+a.eps) * g
			ad[i] = a.rho*ad[i] + (1-a.rho)*delta*delta
			data[i] -= grp.LR * delta
		}
	})
}
