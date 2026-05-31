package optim

import (
	"math"

	"gonn/tensor"
)

// Adafactor implements a simplified Adafactor optimizer (Shazeer & Stern, 2018,
// "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost").
//
// SIMPLIFICATION: The defining feature of Adafactor is the factored second
// moment (row/column statistics) that gives sublinear memory. Because GoNN
// treats every parameter as a flat vector and has no notion of matrix shape at
// this layer, this implementation uses the NON-FACTORED second moment:
//
//	beta2hat_t = 1 - t^(decay_rate)
//	v          = beta2hat*v + (1-beta2hat)*(g^2 + eps1)
//	update     = g / sqrt(v)
//	update    /= max(1, RMS(update)/clip_threshold)     // update clipping
//	alpha      = max(eps2, RMS(param)) * relative_step   // relative step size
//	param     -= alpha * update
//
// where the relative step size schedule is relative_step = min(1/sqrt(t), lr),
// matching Adafactor's scale-by-parameter-RMS rule. When lr is passed as the
// fixed external rate it caps the relative step. eps1=1e-30, eps2=1e-3,
// clip_threshold=1.0, decay_rate=-0.8 by default. No momentum (beta1=0) and no
// weight decay, matching torch.optim defaults.
type Adafactor struct {
	params        []*tensor.Tensor
	lr            float64
	eps1          float64
	eps2          float64
	clipThreshold float64
	decayRate     float64

	v    map[*tensor.Tensor][]float64
	step map[*tensor.Tensor]int
}

// AdafactorOption configures an Adafactor optimizer.
type AdafactorOption func(*Adafactor)

// WithAdafactorEps1 sets the regularization constant eps1 added to g^2.
func WithAdafactorEps1(e float64) AdafactorOption { return func(a *Adafactor) { a.eps1 = e } }

// WithAdafactorEps2 sets the regularization constant eps2 for the param RMS floor.
func WithAdafactorEps2(e float64) AdafactorOption { return func(a *Adafactor) { a.eps2 = e } }

// WithAdafactorClipThreshold sets the update clipping threshold.
func WithAdafactorClipThreshold(c float64) AdafactorOption {
	return func(a *Adafactor) { a.clipThreshold = c }
}

// WithAdafactorDecayRate sets the second-moment decay-rate exponent.
func WithAdafactorDecayRate(d float64) AdafactorOption {
	return func(a *Adafactor) { a.decayRate = d }
}

// NewAdafactor constructs an Adafactor optimizer with defaults eps1=1e-30,
// eps2=1e-3, clip_threshold=1.0, decay_rate=-0.8. The lr argument acts as the
// upper bound on the relative step size (PyTorch's relative_step cap).
func NewAdafactor(params []*tensor.Tensor, lr float64, opts ...AdafactorOption) *Adafactor {
	a := &Adafactor{
		params:        params,
		lr:            lr,
		eps1:          1e-30,
		eps2:          1e-3,
		clipThreshold: 1.0,
		decayRate:     -0.8,
		v:             make(map[*tensor.Tensor][]float64),
		step:          make(map[*tensor.Tensor]int),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

func rms(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	var s float64
	for _, e := range x {
		s += e * e
	}
	return math.Sqrt(s / float64(len(x)))
}

// Step performs a single Adafactor update.
func (a *Adafactor) Step() {
	for _, p := range a.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		v := a.v[p]
		if v == nil {
			v = make([]float64, len(data))
			a.v[p] = v
		}
		a.step[p]++
		t := float64(a.step[p])

		beta2hat := 1 - math.Pow(t, a.decayRate)

		// relative step size: min(1/sqrt(t), lr), scaled by param RMS.
		relStep := math.Min(1/math.Sqrt(t), a.lr)
		paramRMS := math.Max(a.eps2, rms(data))
		alpha := paramRMS * relStep

		update := make([]float64, len(data))
		for i := range data {
			g := grad[i]
			v[i] = beta2hat*v[i] + (1-beta2hat)*(g*g+a.eps1)
			update[i] = g / math.Sqrt(v[i])
		}

		// update clipping by RMS / clip_threshold.
		uRMS := rms(update)
		denom := math.Max(1.0, uRMS/a.clipThreshold)
		for i := range data {
			data[i] -= alpha * update[i] / denom
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (a *Adafactor) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *Adafactor) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *Adafactor) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *Adafactor) SetLR(lr float64) { a.lr = lr }
