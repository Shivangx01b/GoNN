package optim

import (
	"math"

	"gonn/tensor"
)

// NAdam implements Adam with Nesterov momentum, using the PyTorch-style
// schedule for the momentum decay product mu_t.
type NAdam struct {
	baseOptimizer
	beta1         float64
	beta2         float64
	eps           float64
	momentumDecay float64
	t             int
	muProduct     float64
}

// NAdamOption configures a NAdam optimizer.
type NAdamOption func(*NAdam)

// WithNAdamBeta1 sets the beta1 coefficient.
func WithNAdamBeta1(b float64) NAdamOption { return func(a *NAdam) { a.beta1 = b } }

// WithNAdamBeta2 sets the beta2 coefficient.
func WithNAdamBeta2(b float64) NAdamOption { return func(a *NAdam) { a.beta2 = b } }

// WithNAdamEps sets the epsilon term.
func WithNAdamEps(e float64) NAdamOption { return func(a *NAdam) { a.eps = e } }

// WithNAdamWeightDecay sets the L2 weight decay coefficient on every group.
// With NewNAdamGroups, prefer setting Group.WeightDecay directly.
func WithNAdamWeightDecay(wd float64) NAdamOption {
	return func(a *NAdam) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// WithNAdamMomentumDecay sets the momentum decay coefficient.
func WithNAdamMomentumDecay(d float64) NAdamOption { return func(a *NAdam) { a.momentumDecay = d } }

// NewNAdam constructs a NAdam optimizer with PyTorch defaults
// beta1=0.9, beta2=0.999, eps=1e-8, momentumDecay=4e-3.
func NewNAdam(params []*tensor.Tensor, lr float64, opts ...NAdamOption) *NAdam {
	return NewNAdamGroups(singleGroup(params, lr), opts...)
}

// NewNAdamGroups constructs a NAdam optimizer over explicit parameter groups.
func NewNAdamGroups(groups []Group, opts ...NAdamOption) *NAdam {
	a := &NAdam{
		baseOptimizer: newBase(groups),
		beta1:         0.9,
		beta2:         0.999,
		eps:           1e-8,
		momentumDecay: 4e-3,
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
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		v := st.Buf("v", len(data))
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := muTNext*m[i]/(1-muProductNext) + (1-muT)*g/(1-a.muProduct)
			vHat := v[i] / bc2
			data[i] -= grp.LR * mHat / (math.Sqrt(vHat) + a.eps)
		}
	})
}
