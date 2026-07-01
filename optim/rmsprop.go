package optim

import (
	"math"

	"gonn/tensor"
)

// RMSprop implements the RMSprop optimizer with optional momentum.
type RMSprop struct {
	baseOptimizer
	alpha    float64
	eps      float64
	momentum float64
}

// RMSpropOption configures an RMSprop optimizer.
type RMSpropOption func(*RMSprop)

// WithAlpha sets the smoothing constant alpha.
func WithAlpha(a float64) RMSpropOption { return func(r *RMSprop) { r.alpha = a } }

// WithRMSpropEps sets the epsilon term.
func WithRMSpropEps(e float64) RMSpropOption { return func(r *RMSprop) { r.eps = e } }

// WithRMSpropMomentum sets the momentum coefficient.
func WithRMSpropMomentum(m float64) RMSpropOption { return func(r *RMSprop) { r.momentum = m } }

// WithRMSpropWeightDecay sets the L2 weight decay coefficient on every group.
// With NewRMSpropGroups, prefer setting Group.WeightDecay directly.
func WithRMSpropWeightDecay(wd float64) RMSpropOption {
	return func(r *RMSprop) {
		for i := range r.groups {
			r.groups[i].WeightDecay = wd
		}
	}
}

// NewRMSprop constructs an RMSprop optimizer with defaults alpha=0.99, eps=1e-8.
func NewRMSprop(params []*tensor.Tensor, lr float64, opts ...RMSpropOption) *RMSprop {
	return NewRMSpropGroups(singleGroup(params, lr), opts...)
}

// NewRMSpropGroups constructs an RMSprop optimizer over explicit parameter groups.
func NewRMSpropGroups(groups []Group, opts ...RMSpropOption) *RMSprop {
	r := &RMSprop{
		baseOptimizer: newBase(groups),
		alpha:         0.99,
		eps:           1e-8,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Step performs a single RMSprop update.
func (r *RMSprop) Step() {
	r.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		sa := st.Buf("square_avg", len(data))
		var mb []float64
		if r.momentum != 0 {
			mb = st.Buf("momentum_buf", len(data))
		}
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			sa[i] = r.alpha*sa[i] + (1-r.alpha)*g*g
			denom := math.Sqrt(sa[i]) + r.eps
			if r.momentum != 0 {
				mb[i] = r.momentum*mb[i] + g/denom
				data[i] -= grp.LR * mb[i]
			} else {
				data[i] -= grp.LR * g / denom
			}
		}
	})
}
