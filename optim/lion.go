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
	params      []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	weightDecay float64
	m           map[*tensor.Tensor][]float64
}

// LionOption configures a Lion optimizer.
type LionOption func(*Lion)

// WithLionBeta1 sets the beta1 coefficient (interpolation for the update sign).
func WithLionBeta1(b float64) LionOption { return func(l *Lion) { l.beta1 = b } }

// WithLionBeta2 sets the beta2 coefficient (momentum EMA).
func WithLionBeta2(b float64) LionOption { return func(l *Lion) { l.beta2 = b } }

// WithLionWeightDecay sets the decoupled weight decay coefficient.
func WithLionWeightDecay(wd float64) LionOption { return func(l *Lion) { l.weightDecay = wd } }

// NewLion constructs a Lion optimizer with defaults beta1=0.9, beta2=0.99,
// weight_decay=0.
func NewLion(params []*tensor.Tensor, lr float64, opts ...LionOption) *Lion {
	l := &Lion{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.99,
		m:      make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

// Step performs a single Lion update.
func (l *Lion) Step() {
	for _, p := range l.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		m := l.m[p]
		if m == nil {
			m = make([]float64, len(data))
			l.m[p] = m
		}
		for i := range data {
			g := grad[i]
			c := l.beta1*m[i] + (1-l.beta1)*g
			update := sign(c)
			if l.weightDecay != 0 {
				data[i] -= l.lr * (update + l.weightDecay*data[i])
			} else {
				data[i] -= l.lr * update
			}
			m[i] = l.beta2*m[i] + (1-l.beta2)*g
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (l *Lion) ZeroGrad() { zeroGradAll(l.params) }

// Parameters returns the parameter list.
func (l *Lion) Parameters() []*tensor.Tensor { return l.params }

// LR returns the current learning rate.
func (l *Lion) LR() float64 { return l.lr }

// SetLR updates the learning rate.
func (l *Lion) SetLR(lr float64) { l.lr = lr }
