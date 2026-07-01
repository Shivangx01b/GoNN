package optim

import "gonn/tensor"

// SGD implements stochastic gradient descent with optional momentum,
// weight decay (coupled L2), and Nesterov acceleration.
type SGD struct {
	baseOptimizer
	momentum float64
	nesterov bool
}

// SGDOption configures an SGD optimizer.
type SGDOption func(*SGD)

// WithMomentum sets the momentum coefficient.
func WithMomentum(m float64) SGDOption { return func(s *SGD) { s.momentum = m } }

// WithSGDWeightDecay sets the L2 weight decay coefficient on every group.
// With NewSGDGroups, prefer setting Group.WeightDecay directly.
func WithSGDWeightDecay(wd float64) SGDOption {
	return func(s *SGD) {
		for i := range s.groups {
			s.groups[i].WeightDecay = wd
		}
	}
}

// WithNesterov enables Nesterov-style momentum.
func WithNesterov(n bool) SGDOption { return func(s *SGD) { s.nesterov = n } }

// NewSGD constructs an SGD optimizer over a single parameter group.
func NewSGD(params []*tensor.Tensor, lr float64, opts ...SGDOption) *SGD {
	return NewSGDGroups(singleGroup(params, lr), opts...)
}

// NewSGDGroups constructs an SGD optimizer over explicit parameter groups.
func NewSGDGroups(groups []Group, opts ...SGDOption) *SGD {
	s := &SGD{baseOptimizer: newBase(groups)}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step performs a single optimization step.
func (s *SGD) Step() {
	s.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		var v []float64
		if s.momentum != 0 {
			v = st.Buf("momentum_buf", len(data))
		}
		for i := range data {
			g := coupledWD(grad[i], data[i], grp.WeightDecay)
			if s.momentum != 0 {
				v[i] = s.momentum*v[i] + g
				var update float64
				if s.nesterov {
					update = g + s.momentum*v[i]
				} else {
					update = v[i]
				}
				data[i] -= grp.LR * update
			} else {
				data[i] -= grp.LR * g
			}
		}
	})
}
