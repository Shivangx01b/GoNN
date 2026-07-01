package optim

import (
	"math"

	"gonn/tensor"
)

// ASGD implements Averaged Stochastic Gradient Descent (Polyak & Juditsky,
// 1992), matching torch.optim.ASGD semantics.
//
// For each parameter the live param is updated with the standard SGD step
// scaled by the eta schedule, while a running average ax is maintained. PyTorch
// keeps the live params trained and tracks the averaged weights in ax. The
// per-step schedules are:
//
//	eta_{t} = lr / (1 + lambda*lr*t)^alpha
//	mu_{t}  = 1 / max(1, t - t0)
//
// and the averaged buffer ax accumulates ax += mu*(param - ax).
type ASGD struct {
	baseOptimizer
	lambda float64
	alpha  float64
	t0     float64
}

// ASGDOption configures an ASGD optimizer.
type ASGDOption func(*ASGD)

// WithASGDLambda sets the decay term lambda.
func WithASGDLambda(l float64) ASGDOption { return func(a *ASGD) { a.lambda = l } }

// WithASGDAlpha sets the power for the eta update.
func WithASGDAlpha(al float64) ASGDOption { return func(a *ASGD) { a.alpha = al } }

// WithASGDT0 sets the point at which averaging starts.
func WithASGDT0(t0 float64) ASGDOption { return func(a *ASGD) { a.t0 = t0 } }

// WithASGDWeightDecay sets the L2 weight decay coefficient on every group.
// With NewASGDGroups, prefer setting Group.WeightDecay directly.
func WithASGDWeightDecay(wd float64) ASGDOption {
	return func(a *ASGD) {
		for i := range a.groups {
			a.groups[i].WeightDecay = wd
		}
	}
}

// NewASGD constructs an ASGD optimizer with defaults lambda=1e-4, alpha=0.75,
// t0=1e6, weight_decay=0.
func NewASGD(params []*tensor.Tensor, lr float64, opts ...ASGDOption) *ASGD {
	return NewASGDGroups(singleGroup(params, lr), opts...)
}

// NewASGDGroups constructs an ASGD optimizer over explicit parameter groups.
func NewASGDGroups(groups []Group, opts ...ASGDOption) *ASGD {
	a := &ASGD{
		baseOptimizer: newBase(groups),
		lambda:        1e-4,
		alpha:         0.75,
		t0:            1e6,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single ASGD update.
func (a *ASGD) Step() {
	a.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		if !st.Has("ax") {
			st.SetScalar("eta", grp.LR)
			st.SetScalar("mu", 1)
			st.SetScalar("step", 0)
		}
		ax := st.Buf("ax", len(data))

		st.SetScalar("step", st.Scalar("step")+1)
		t := st.Scalar("step")
		eta := st.Scalar("eta")
		mu := st.Scalar("mu")

		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			// decay term (PyTorch: param.mul_(1 - lambda*eta))
			data[i] *= 1 - a.lambda*eta
			// gradient step
			data[i] -= eta * g
			// averaging
			if mu != 1 {
				ax[i] += mu * (data[i] - ax[i])
			} else {
				ax[i] = data[i]
			}
		}

		// update eta and mu schedules for next step
		st.SetScalar("eta", grp.LR/math.Pow(1+a.lambda*grp.LR*t, a.alpha))
		st.SetScalar("mu", 1/math.Max(1, t-a.t0))
	})
}

// AveragedParam returns the averaged weight buffer (ax) for a parameter, or nil
// if the parameter has not been stepped yet.
func (a *ASGD) AveragedParam(p *tensor.Tensor) []float64 {
	st := a.states[p]
	if st == nil || !st.Has("ax") {
		return nil
	}
	return st.Buf("ax", 0)
}
