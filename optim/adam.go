package optim

import (
	"math"

	"gonn/tensor"
)

// Adam implements the Adam optimizer (Kingma & Ba, 2014). Weight decay is
// coupled L2 (added to the gradient); for decoupled decay use AdamW.
type Adam struct {
	baseOptimizer
	beta1 float64
	beta2 float64
	eps   float64
	t     int
}

// AdamOption configures an Adam optimizer.
type AdamOption func(*Adam)

// WithBeta1 sets the beta1 coefficient.
func WithBeta1(b float64) AdamOption { return func(a *Adam) { a.beta1 = b } }

// WithBeta2 sets the beta2 coefficient.
func WithBeta2(b float64) AdamOption { return func(a *Adam) { a.beta2 = b } }

// WithAdamEps sets the epsilon term.
func WithAdamEps(e float64) AdamOption { return func(a *Adam) { a.eps = e } }

// WithAdamWeightDecay sets the L2 weight decay coefficient on every group.
// With NewAdamGroups, prefer setting Group.WeightDecay directly.
func WithAdamWeightDecay(wd float64) AdamOption {
	return func(a *Adam) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// NewAdam constructs an Adam optimizer with defaults beta1=0.9, beta2=0.999, eps=1e-8.
func NewAdam(params []*tensor.Tensor, lr float64, opts ...AdamOption) *Adam {
	return NewAdamGroups(singleGroup(params, lr), opts...)
}

// NewAdamGroups constructs an Adam optimizer over explicit parameter groups.
func NewAdamGroups(groups []Group, opts ...AdamOption) *Adam {
	a := &Adam{
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

// Step performs a single Adam update.
func (a *Adam) Step() {
	a.t++
	bc1 := 1 - math.Pow(a.beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.beta2, float64(a.t))
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		v := st.Buf("v", len(data))
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			data[i] -= grp.LR * mHat / (math.Sqrt(vHat) + a.eps)
		}
	})
}
