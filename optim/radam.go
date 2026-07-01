package optim

import (
	"math"

	"gonn/tensor"
)

// RAdam implements Rectified Adam (Liu et al., 2019). When the variance of
// the adaptive learning rate is not yet tractable it falls back to an
// SGD-with-momentum style update; otherwise it applies a variance rectified
// Adam step.
type RAdam struct {
	baseOptimizer
	beta1 float64
	beta2 float64
	eps   float64
	t     int
}

// RAdamOption configures a RAdam optimizer.
type RAdamOption func(*RAdam)

// WithRAdamBeta1 sets the beta1 coefficient.
func WithRAdamBeta1(b float64) RAdamOption { return func(a *RAdam) { a.beta1 = b } }

// WithRAdamBeta2 sets the beta2 coefficient.
func WithRAdamBeta2(b float64) RAdamOption { return func(a *RAdam) { a.beta2 = b } }

// WithRAdamEps sets the epsilon term.
func WithRAdamEps(e float64) RAdamOption { return func(a *RAdam) { a.eps = e } }

// WithRAdamWeightDecay sets the L2 weight decay coefficient on every group.
// With NewRAdamGroups, prefer setting Group.WeightDecay directly.
func WithRAdamWeightDecay(wd float64) RAdamOption {
	return func(a *RAdam) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// NewRAdam constructs a RAdam optimizer with defaults beta1=0.9,
// beta2=0.999, eps=1e-8, weight_decay=0.
func NewRAdam(params []*tensor.Tensor, lr float64, opts ...RAdamOption) *RAdam {
	return NewRAdamGroups(singleGroup(params, lr), opts...)
}

// NewRAdamGroups constructs a RAdam optimizer over explicit parameter groups.
func NewRAdamGroups(groups []Group, opts ...RAdamOption) *RAdam {
	a := &RAdam{
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

// Step performs a single RAdam update.
func (a *RAdam) Step() {
	a.t++
	t := float64(a.t)
	bc1 := 1 - math.Pow(a.beta1, t)
	bc2 := 1 - math.Pow(a.beta2, t)
	rhoInf := 2/(1-a.beta2) - 1
	rhoT := rhoInf - 2*t*math.Pow(a.beta2, t)/bc2

	var rectified float64
	useRect := rhoT > 4
	if useRect {
		// PyTorch formulation of the rectification term r_t.
		rectified = math.Sqrt(((rhoT - 4) * (rhoT - 2) * rhoInf) /
			((rhoInf - 4) * (rhoInf - 2) * rhoT))
	}

	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		v := st.Buf("v", len(data))
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := m[i] / bc1
			if useRect {
				vHat := math.Sqrt(v[i] / bc2)
				data[i] -= grp.LR * rectified * mHat / (vHat + a.eps)
			} else {
				// Variance is not tractable: fall back to unadapted update.
				data[i] -= grp.LR * mHat
			}
		}
	})
}
