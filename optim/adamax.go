package optim

import (
	"math"

	"gonn/tensor"
)

// Adamax implements the Adamax optimizer (Kingma & Ba, 2014), a variant of
// Adam based on the infinity norm.
type Adamax struct {
	baseOptimizer
	beta1 float64
	beta2 float64
	eps   float64
	t     int
}

// AdamaxOption configures an Adamax optimizer.
type AdamaxOption func(*Adamax)

// WithAdamaxBeta1 sets the beta1 coefficient.
func WithAdamaxBeta1(b float64) AdamaxOption { return func(a *Adamax) { a.beta1 = b } }

// WithAdamaxBeta2 sets the beta2 coefficient.
func WithAdamaxBeta2(b float64) AdamaxOption { return func(a *Adamax) { a.beta2 = b } }

// WithAdamaxEps sets the epsilon term.
func WithAdamaxEps(e float64) AdamaxOption { return func(a *Adamax) { a.eps = e } }

// WithAdamaxWeightDecay sets the L2 weight decay coefficient on every group.
// With NewAdamaxGroups, prefer setting Group.WeightDecay directly.
func WithAdamaxWeightDecay(wd float64) AdamaxOption {
	return func(a *Adamax) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// NewAdamax constructs an Adamax optimizer with defaults beta1=0.9,
// beta2=0.999, eps=1e-8, weight_decay=0.
func NewAdamax(params []*tensor.Tensor, lr float64, opts ...AdamaxOption) *Adamax {
	return NewAdamaxGroups(singleGroup(params, lr), opts...)
}

// NewAdamaxGroups constructs an Adamax optimizer over explicit parameter groups.
func NewAdamaxGroups(groups []Group, opts ...AdamaxOption) *Adamax {
	a := &Adamax{
		baseOptimizer: newBase(groups),
		beta1:         0.9,
		beta2:         0.999,
		eps:           1e-8,
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
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		u := st.Buf("u", len(data))
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			absG := math.Abs(g)
			scaled := a.beta2 * u[i]
			if absG > scaled {
				u[i] = absG
			} else {
				u[i] = scaled
			}
			data[i] -= (grp.LR / bc1) * m[i] / (u[i] + a.eps)
		}
	})
}
