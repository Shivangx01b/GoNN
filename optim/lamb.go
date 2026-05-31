package optim

import (
	"math"

	"gonn/tensor"
)

// LAMB implements the LAMB optimizer (You et al. 2019, "Large Batch
// Optimization for Deep Learning").
//
// It computes Adam-style bias-corrected first and second moments, forms the
// Adam update (including decoupled weight decay), then rescales it by a
// layer-wise (per parameter tensor) trust ratio:
//
//	r       = mHat / (sqrt(vHat) + eps)
//	update  = r + weight_decay*param
//	trust   = ||param|| / ||update||   (1 if either norm is 0)
//	param  -= lr * trust * update
//
// Defaults: beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.
type LAMB struct {
	params      []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	m           map[*tensor.Tensor][]float64
	v           map[*tensor.Tensor][]float64
	t           int
}

// LAMBOption configures a LAMB optimizer.
type LAMBOption func(*LAMB)

// WithLAMBBeta1 sets the beta1 coefficient.
func WithLAMBBeta1(b float64) LAMBOption { return func(l *LAMB) { l.beta1 = b } }

// WithLAMBBeta2 sets the beta2 coefficient.
func WithLAMBBeta2(b float64) LAMBOption { return func(l *LAMB) { l.beta2 = b } }

// WithLAMBEps sets the epsilon term.
func WithLAMBEps(e float64) LAMBOption { return func(l *LAMB) { l.eps = e } }

// WithLAMBWeightDecay sets the decoupled weight decay coefficient.
func WithLAMBWeightDecay(wd float64) LAMBOption { return func(l *LAMB) { l.weightDecay = wd } }

// NewLAMB constructs a LAMB optimizer with defaults beta1=0.9, beta2=0.999,
// eps=1e-6, weight_decay=0.
func NewLAMB(params []*tensor.Tensor, lr float64, opts ...LAMBOption) *LAMB {
	l := &LAMB{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-6,
		m:      make(map[*tensor.Tensor][]float64),
		v:      make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

// Step performs a single LAMB update.
func (l *LAMB) Step() {
	l.t++
	bc1 := 1 - math.Pow(l.beta1, float64(l.t))
	bc2 := 1 - math.Pow(l.beta2, float64(l.t))
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
		v := l.v[p]
		if v == nil {
			v = make([]float64, len(data))
			l.v[p] = v
		}

		update := make([]float64, len(data))
		var paramNorm, updNorm float64
		for i := range data {
			g := grad[i]
			m[i] = l.beta1*m[i] + (1-l.beta1)*g
			v[i] = l.beta2*v[i] + (1-l.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			u := mHat / (math.Sqrt(vHat) + l.eps)
			if l.weightDecay != 0 {
				u += l.weightDecay * data[i]
			}
			update[i] = u
			paramNorm += data[i] * data[i]
			updNorm += u * u
		}
		paramNorm = math.Sqrt(paramNorm)
		updNorm = math.Sqrt(updNorm)

		trust := 1.0
		if paramNorm > 0 && updNorm > 0 {
			trust = paramNorm / updNorm
		}
		for i := range data {
			data[i] -= l.lr * trust * update[i]
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (l *LAMB) ZeroGrad() { zeroGradAll(l.params) }

// Parameters returns the parameter list.
func (l *LAMB) Parameters() []*tensor.Tensor { return l.params }

// LR returns the current learning rate.
func (l *LAMB) LR() float64 { return l.lr }

// SetLR updates the learning rate.
func (l *LAMB) SetLR(lr float64) { l.lr = lr }
