package optim

import (
	"gonn/tensor"
)

// Lion implements the Lion optimizer (EvoLved Sign Momentum, Chen et al. 2023).
//
// Per element the update is:
//
//	c      = beta1*m + (1-beta1)*g
//	param -= lr*(sign(c) + weight_decay*param)
//	m      = beta2*m + (1-beta2)*g
//
// Defaults: beta1=0.9, beta2=0.99, weight_decay=0.
type Lion struct {
	baseOptimizer
	beta1 float64
	beta2 float64
}

// LionOption configures a Lion optimizer.
type LionOption func(*Lion)

// WithLionBeta1 sets the beta1 coefficient (interpolation for the update sign).
func WithLionBeta1(b float64) LionOption { return func(l *Lion) { l.beta1 = b } }

// WithLionBeta2 sets the beta2 coefficient (momentum EMA).
func WithLionBeta2(b float64) LionOption { return func(l *Lion) { l.beta2 = b } }

// WithLionWeightDecay sets the decoupled weight decay coefficient on every
// group. With NewLionGroups, prefer setting Group.WeightDecay directly.
func WithLionWeightDecay(wd float64) LionOption {
	return func(l *Lion) {
		for i := range l.groups {
			l.groups[i].WeightDecay = wd
		}
	}
}

// NewLion constructs a Lion optimizer with defaults beta1=0.9, beta2=0.99,
// weight_decay=0.
func NewLion(params []*tensor.Tensor, lr float64, opts ...LionOption) *Lion {
	return NewLionGroups(singleGroup(params, lr), opts...)
}

// NewLionGroups constructs a Lion optimizer over explicit parameter groups.
func NewLionGroups(groups []Group, opts ...LionOption) *Lion {
	l := &Lion{
		baseOptimizer: newBase(groups),
		beta1:         0.9,
		beta2:         0.99,
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

// Step performs a single Lion update.
func (l *Lion) Step() {
	l.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		for i := range data {
			g := grad[i]
			c := l.beta1*m[i] + (1-l.beta1)*g
			update := sign(c)
			if grp.WeightDecay != 0 {
				data[i] -= grp.LR * (update + grp.WeightDecay*data[i])
			} else {
				data[i] -= grp.LR * update
			}
			m[i] = l.beta2*m[i] + (1-l.beta2)*g
		}
	})
}
