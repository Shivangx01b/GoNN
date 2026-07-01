package optim

import (
	"math"

	"gonn/tensor"
)

// AdamW implements Adam with decoupled weight decay (Loshchilov & Hutter,
// 2019): the decay scales the parameter directly instead of entering the
// gradient/moments.
type AdamW struct {
	baseOptimizer
	beta1 float64
	beta2 float64
	eps   float64
	t     int
}

// AdamWOption configures an AdamW optimizer.
type AdamWOption func(*AdamW)

// WithAdamWBeta1 sets the beta1 coefficient.
func WithAdamWBeta1(b float64) AdamWOption { return func(a *AdamW) { a.beta1 = b } }

// WithAdamWBeta2 sets the beta2 coefficient.
func WithAdamWBeta2(b float64) AdamWOption { return func(a *AdamW) { a.beta2 = b } }

// WithAdamWEps sets the epsilon term.
func WithAdamWEps(e float64) AdamWOption { return func(a *AdamW) { a.eps = e } }

// WithAdamWWeightDecay sets the decoupled weight decay coefficient on every
// group. With NewAdamWGroups, prefer setting Group.WeightDecay directly.
func WithAdamWWeightDecay(wd float64) AdamWOption {
	return func(a *AdamW) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// NewAdamW constructs an AdamW optimizer with defaults beta1=0.9,
// beta2=0.999, eps=1e-8, weightDecay=0.01 (applied to the single group).
func NewAdamW(params []*tensor.Tensor, lr float64, opts ...AdamWOption) *AdamW {
	return NewAdamWGroups([]Group{{Params: params, LR: lr, WeightDecay: 0.01}}, opts...)
}

// NewAdamWGroups constructs an AdamW optimizer over explicit parameter
// groups. Group weight decays are taken verbatim (no 0.01 default injected).
func NewAdamWGroups(groups []Group, opts ...AdamWOption) *AdamW {
	a := &AdamW{
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

// Step performs a single AdamW update.
func (a *AdamW) Step() {
	a.t++
	bc1 := 1 - math.Pow(a.beta1, float64(a.t))
	bc2 := 1 - math.Pow(a.beta2, float64(a.t))
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		v := st.Buf("v", len(data))
		for i := range data {
			g := grad[i]
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			// Decoupled weight decay: applied directly to the parameter.
			if grp.WeightDecay != 0 {
				data[i] -= grp.LR * grp.WeightDecay * data[i]
			}
			data[i] -= grp.LR * mHat / (math.Sqrt(vHat) + a.eps)
		}
	})
}
