package optim

import (
	"math"

	"gonn/tensor"
)

// Rprop implements the resilient backpropagation optimizer (Riedmiller &
// Braun, 1993). Each parameter element keeps its own step size that grows
// when the gradient sign is stable and shrinks when it flips.
type Rprop struct {
	params       []*tensor.Tensor
	lr           float64 // base step size used to initialize per-element sizes
	etaMinus     float64
	etaPlus      float64
	stepMin      float64
	stepMax      float64
	prevGrad     map[*tensor.Tensor][]float64
	stepSize     map[*tensor.Tensor][]float64
}

// RpropOption configures a Rprop optimizer.
type RpropOption func(*Rprop)

// WithRpropEtaMinus sets the shrink factor used when gradient signs flip.
func WithRpropEtaMinus(e float64) RpropOption { return func(r *Rprop) { r.etaMinus = e } }

// WithRpropEtaPlus sets the grow factor used when gradient signs agree.
func WithRpropEtaPlus(e float64) RpropOption { return func(r *Rprop) { r.etaPlus = e } }

// WithRpropStepBounds sets the (min, max) clamp applied to per-element steps.
func WithRpropStepBounds(min, max float64) RpropOption {
	return func(r *Rprop) {
		r.stepMin = min
		r.stepMax = max
	}
}

// NewRprop constructs a Rprop optimizer. lr is used as the initial per-element
// step size (PyTorch defaults to 0.01; we follow the task spec and use it as
// the user-supplied initial step). Defaults: eta_minus=0.5, eta_plus=1.2,
// step bounds = (1e-6, 50).
func NewRprop(params []*tensor.Tensor, lr float64, opts ...RpropOption) *Rprop {
	r := &Rprop{
		params:   params,
		lr:       lr,
		etaMinus: 0.5,
		etaPlus:  1.2,
		stepMin:  1e-6,
		stepMax:  50,
		prevGrad: make(map[*tensor.Tensor][]float64),
		stepSize: make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Step performs a single Rprop update.
func (r *Rprop) Step() {
	for _, p := range r.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		prev := r.prevGrad[p]
		if prev == nil {
			prev = make([]float64, len(data))
			r.prevGrad[p] = prev
		}
		step := r.stepSize[p]
		if step == nil {
			step = make([]float64, len(data))
			for i := range step {
				step[i] = r.lr
			}
			r.stepSize[p] = step
		}
		for i := range data {
			g := grad[i]
			signProd := prev[i] * g
			switch {
			case signProd > 0:
				step[i] *= r.etaPlus
				if step[i] > r.stepMax {
					step[i] = r.stepMax
				}
			case signProd < 0:
				step[i] *= r.etaMinus
				if step[i] < r.stepMin {
					step[i] = r.stepMin
				}
				// When sign flips, suppress the update on this step.
				g = 0
			}
			if g > 0 {
				data[i] -= step[i]
			} else if g < 0 {
				data[i] += step[i]
			}
			// Remember the gradient that drove this update (0 if suppressed
			// after a sign flip, matching PyTorch's behaviour).
			prev[i] = sign(g) * math.Abs(grad[i])
			if signProd < 0 {
				prev[i] = 0
			}
		}
	}
}

func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	}
	return 0
}

// ZeroGrad zeros the gradients of all parameters.
func (r *Rprop) ZeroGrad() { zeroGradAll(r.params) }

// Parameters returns the parameter list.
func (r *Rprop) Parameters() []*tensor.Tensor { return r.params }

// LR returns the base step size.
func (r *Rprop) LR() float64 { return r.lr }

// SetLR updates the base step size (does not retroactively rescale existing
// per-element steps).
func (r *Rprop) SetLR(lr float64) { r.lr = lr }
