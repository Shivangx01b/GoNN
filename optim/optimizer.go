// Package optim provides PyTorch-style optimizers and LR schedulers.
//
// Every optimizer embeds a shared base providing parameter groups (per-group
// learning rate and weight decay — see Group), ZeroGrad, and LR accessors;
// the per-optimizer code is just the update rule in Step. Free functions
// ClipGradNorm/ClipGradValue clip gradients between Backward() and Step().
// LBFGS is the one exception to this interface: its closure-based
// Step(closure) has a different signature, so it satisfies everything here
// except Step.
package optim

import "gonn/tensor"

// Optimizer is the common interface implemented by all optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
	Parameters() []*tensor.Tensor
	// Groups returns live pointers to the parameter groups; mutating a
	// group's LR/WeightDecay through them takes effect on the next Step.
	Groups() []*Group
	LR() float64
	SetLR(float64)
}

// zeroGradAll zeros the gradient buffers of every parameter in place.
func zeroGradAll(params []*tensor.Tensor) {
	for _, p := range params {
		if p == nil || p.Grad == nil {
			continue
		}
		for i := range p.Grad.Data {
			p.Grad.Data[i] = 0
		}
	}
}
