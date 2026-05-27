package optim

import (
	"math"

	"gonn/tensor"
)

// RMSprop implements the RMSprop optimizer with optional momentum.
type RMSprop struct {
	params      []*tensor.Tensor
	lr          float64
	alpha       float64
	eps         float64
	momentum    float64
	weightDecay float64
	squareAvg   map[*tensor.Tensor][]float64
	momentumBuf map[*tensor.Tensor][]float64
}

// RMSpropOption configures an RMSprop optimizer.
type RMSpropOption func(*RMSprop)

// WithAlpha sets the smoothing constant alpha.
func WithAlpha(a float64) RMSpropOption { return func(r *RMSprop) { r.alpha = a } }

// WithRMSpropEps sets the epsilon term.
func WithRMSpropEps(e float64) RMSpropOption { return func(r *RMSprop) { r.eps = e } }

// WithRMSpropMomentum sets the momentum coefficient.
func WithRMSpropMomentum(m float64) RMSpropOption { return func(r *RMSprop) { r.momentum = m } }

// WithRMSpropWeightDecay sets the L2 weight decay coefficient.
func WithRMSpropWeightDecay(wd float64) RMSpropOption {
	return func(r *RMSprop) { r.weightDecay = wd }
}

// NewRMSprop constructs an RMSprop optimizer with defaults alpha=0.99, eps=1e-8.
func NewRMSprop(params []*tensor.Tensor, lr float64, opts ...RMSpropOption) *RMSprop {
	r := &RMSprop{
		params:      params,
		lr:          lr,
		alpha:       0.99,
		eps:         1e-8,
		squareAvg:   make(map[*tensor.Tensor][]float64),
		momentumBuf: make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Step performs a single RMSprop update.
func (r *RMSprop) Step() {
	for _, p := range r.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		sa := r.squareAvg[p]
		if sa == nil {
			sa = make([]float64, len(data))
			r.squareAvg[p] = sa
		}
		var mb []float64
		if r.momentum != 0 {
			mb = r.momentumBuf[p]
			if mb == nil {
				mb = make([]float64, len(data))
				r.momentumBuf[p] = mb
			}
		}
		for i := range data {
			g := grad[i]
			if r.weightDecay != 0 {
				g += r.weightDecay * data[i]
			}
			sa[i] = r.alpha*sa[i] + (1-r.alpha)*g*g
			denom := math.Sqrt(sa[i]) + r.eps
			if r.momentum != 0 {
				mb[i] = r.momentum*mb[i] + g/denom
				data[i] -= r.lr * mb[i]
			} else {
				data[i] -= r.lr * g / denom
			}
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (r *RMSprop) ZeroGrad() { zeroGradAll(r.params) }

// Parameters returns the parameter list.
func (r *RMSprop) Parameters() []*tensor.Tensor { return r.params }

// LR returns the current learning rate.
func (r *RMSprop) LR() float64 { return r.lr }

// SetLR updates the learning rate.
func (r *RMSprop) SetLR(lr float64) { r.lr = lr }
