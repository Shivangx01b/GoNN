package optim

import "gonn/tensor"

// baseOptimizer carries the shared plumbing every optimizer needs: the
// parameter groups, the per-parameter state map, and the accessor methods of
// the Optimizer interface. Concrete optimizers embed it and implement only
// Step (plus their constructors/options).
type baseOptimizer struct {
	groups []Group
	states map[*tensor.Tensor]*State
}

// newBase copies the group slice (and each Params slice) so later caller
// mutations of the input don't alias optimizer state.
func newBase(groups []Group) baseOptimizer {
	gs := make([]Group, len(groups))
	for i, g := range groups {
		gs[i] = Group{
			Params:      append([]*tensor.Tensor(nil), g.Params...),
			LR:          g.LR,
			WeightDecay: g.WeightDecay,
		}
	}
	return baseOptimizer{groups: gs, states: make(map[*tensor.Tensor]*State)}
}

// singleGroup wraps the classic (params, lr) constructor arguments.
func singleGroup(params []*tensor.Tensor, lr float64) []Group {
	return []Group{{Params: params, LR: lr}}
}

// Groups returns live pointers to the optimizer's parameter groups; mutating
// a group's LR or WeightDecay through them takes effect immediately
// (PyTorch param_groups semantics).
func (b *baseOptimizer) Groups() []*Group {
	out := make([]*Group, len(b.groups))
	for i := range b.groups {
		out[i] = &b.groups[i]
	}
	return out
}

// Parameters returns all parameters in group order.
func (b *baseOptimizer) Parameters() []*tensor.Tensor {
	var out []*tensor.Tensor
	for i := range b.groups {
		out = append(out, b.groups[i].Params...)
	}
	return out
}

// ZeroGrad zeros the gradients of all parameters.
func (b *baseOptimizer) ZeroGrad() { zeroGradAll(b.Parameters()) }

// LR returns the first group's learning rate (the whole optimizer's LR in
// the common single-group case). For per-group values use Groups().
func (b *baseOptimizer) LR() float64 {
	if len(b.groups) == 0 {
		return 0
	}
	return b.groups[0].LR
}

// SetLR sets EVERY group's learning rate. With a single group this is the
// classic behavior; for per-group control use Groups(). (Schedulers do not
// use SetLR internally — they scale each group relative to its own base, so
// multi-group LR ratios survive scheduling.)
func (b *baseOptimizer) SetLR(lr float64) {
	for i := range b.groups {
		b.groups[i].LR = lr
	}
}

// state returns the (lazily created) per-parameter state.
func (b *baseOptimizer) state(p *tensor.Tensor) *State {
	s := b.states[p]
	if s == nil {
		s = &State{}
		b.states[p] = s
	}
	return s
}

// forEachParam runs fn for every parameter that has a gradient, handing it
// the owning group (for LR/WeightDecay), the raw data/grad slices, and the
// parameter's state. This is the shared Step() template: it absorbs the
// nil-checks and state plumbing that used to be copy-pasted per optimizer.
func (b *baseOptimizer) forEachParam(fn func(g *Group, p *tensor.Tensor, data, grad []float64, st *State)) {
	for gi := range b.groups {
		g := &b.groups[gi]
		for _, p := range g.Params {
			if p == nil || p.Grad == nil {
				continue
			}
			fn(g, p, p.Data, p.Grad.Data, b.state(p))
		}
	}
}
