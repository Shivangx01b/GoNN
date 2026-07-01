package optim

import (
	"math"

	"gonn/tensor"
)

// Adagrad implements the Adagrad optimizer (Duchi et al., 2011).
type Adagrad struct {
	baseOptimizer
	lrDecay float64
	eps     float64
	t       int
}

// AdagradOption configures an Adagrad optimizer.
type AdagradOption func(*Adagrad)

// WithAdagradLRDecay sets the learning-rate decay applied per step.
func WithAdagradLRDecay(d float64) AdagradOption { return func(a *Adagrad) { a.lrDecay = d } }

// WithAdagradWeightDecay sets the L2 weight decay coefficient on every group.
// With NewAdagradGroups, prefer setting Group.WeightDecay directly.
func WithAdagradWeightDecay(wd float64) AdagradOption {
	return func(a *Adagrad) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// WithAdagradEps sets the epsilon term.
func WithAdagradEps(e float64) AdagradOption { return func(a *Adagrad) { a.eps = e } }

// NewAdagrad constructs an Adagrad optimizer with default eps=1e-10.
func NewAdagrad(params []*tensor.Tensor, lr float64, opts ...AdagradOption) *Adagrad {
	return NewAdagradGroups(singleGroup(params, lr), opts...)
}

// NewAdagradGroups constructs an Adagrad optimizer over explicit parameter groups.
func NewAdagradGroups(groups []Group, opts ...AdagradOption) *Adagrad {
	a := &Adagrad{
		baseOptimizer: newBase(groups),
		eps:           1e-10,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single Adagrad update.
func (a *Adagrad) Step() {
	a.t++
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		clr := grp.LR / (1 + float64(a.t-1)*a.lrDecay)
		sum := st.Buf("sum", len(data))
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			sum[i] += g * g
			data[i] -= clr * g / (math.Sqrt(sum[i]) + a.eps)
		}
	})
}
