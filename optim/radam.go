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

// RAdamOption configures a RAdam optimizer.
type RAdamOption func(*RAdam)

// WithRAdamBeta1 sets the beta1 coefficient.
func WithRAdamBeta1(b float64) RAdamOption { return func(a *RAdam) { a.beta1 = b } }

// WithRAdamBeta2 sets the beta2 coefficient.
func WithRAdamBeta2(b float64) RAdamOption { return func(a *RAdam) { a.beta2 = b } }

// WithRAdamEps sets the epsilon term.
func WithRAdamEps(e float64) RAdamOption { return func(a *RAdam) { a.eps = e } }

// WithRAdamWeightDecay sets the L2 weight decay coefficient.
func WithRAdamWeightDecay(wd float64) RAdamOption { return func(a *RAdam) { a.weightDecay = wd } }

// NewRAdam constructs a RAdam optimizer with defaults beta1=0.9,
// beta2=0.999, eps=1e-8, weight_decay=0.
func NewRAdam(params []*tensor.Tensor, lr float64, opts ...RAdamOption) *RAdam {
	a := &RAdam{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-8,
		m:      make(map[*tensor.Tensor][]float64),
		v:      make(map[*tensor.Tensor][]float64),
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

	for _, p := range a.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		m := a.m[p]
		if m == nil {
			m = make([]float64, len(data))
			a.m[p] = m
		}
		v := a.v[p]
		if v == nil {
			v = make([]float64, len(data))
			a.v[p] = v
		}
		for i := range data {
			g := grad[i]
			if a.weightDecay != 0 {
				g += a.weightDecay * data[i]
			}
			m[i] = a.beta1*m[i] + (1-a.beta1)*g
			v[i] = a.beta2*v[i] + (1-a.beta2)*g*g
			mHat := m[i] / bc1
			if useRect {
				vHat := math.Sqrt(v[i] / bc2)
				data[i] -= a.lr * rectified * mHat / (vHat + a.eps)
			} else {
				// Variance is not tractable: fall back to unadapted update.
				data[i] -= a.lr * mHat
			}
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *RAdam) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *RAdam) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *RAdam) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *RAdam) SetLR(lr float64) { a.lr = lr }
