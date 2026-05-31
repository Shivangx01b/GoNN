package optim

import (
	"math"

	"gonn/tensor"
)

// ASGD implements Averaged Stochastic Gradient Descent (Polyak & Juditsky,
// 1992), matching torch.optim.ASGD semantics.
//
// For each parameter the live param is updated with the standard SGD step
// scaled by the eta schedule, while a running average ax is maintained. PyTorch
// keeps the live params trained and tracks the averaged weights in ax. The
// per-step schedules are:
//
//	eta_{t} = lr / (1 + lambda*lr*t)^alpha
//	mu_{t}  = 1 / max(1, t - t0)
//
// and the averaged buffer ax accumulates ax += mu*(param - ax).
type ASGD struct {
	params      []*tensor.Tensor
	lr          float64
	lambda      float64
	alpha       float64
	t0          float64
	weightDecay float64

	// per-param state
	ax   map[*tensor.Tensor][]float64
	eta  map[*tensor.Tensor]float64
	mu   map[*tensor.Tensor]float64
	step map[*tensor.Tensor]float64
}

// ASGDOption configures an ASGD optimizer.
type ASGDOption func(*ASGD)

// WithASGDLambda sets the decay term lambda.
func WithASGDLambda(l float64) ASGDOption { return func(a *ASGD) { a.lambda = l } }

// WithASGDAlpha sets the power for the eta update.
func WithASGDAlpha(al float64) ASGDOption { return func(a *ASGD) { a.alpha = al } }

// WithASGDT0 sets the point at which averaging starts.
func WithASGDT0(t0 float64) ASGDOption { return func(a *ASGD) { a.t0 = t0 } }

// WithASGDWeightDecay sets the L2 weight decay coefficient.
func WithASGDWeightDecay(wd float64) ASGDOption { return func(a *ASGD) { a.weightDecay = wd } }

// NewASGD constructs an ASGD optimizer with defaults lambda=1e-4, alpha=0.75,
// t0=1e6, weight_decay=0.
func NewASGD(params []*tensor.Tensor, lr float64, opts ...ASGDOption) *ASGD {
	a := &ASGD{
		params: params,
		lr:     lr,
		lambda: 1e-4,
		alpha:  0.75,
		t0:     1e6,
		ax:     make(map[*tensor.Tensor][]float64),
		eta:    make(map[*tensor.Tensor]float64),
		mu:     make(map[*tensor.Tensor]float64),
		step:   make(map[*tensor.Tensor]float64),
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// Step performs a single ASGD update.
func (a *ASGD) Step() {
	for _, p := range a.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		ax := a.ax[p]
		if ax == nil {
			ax = make([]float64, len(data))
			a.ax[p] = ax
			a.eta[p] = a.lr
			a.mu[p] = 1
			a.step[p] = 0
		}

		a.step[p]++
		t := a.step[p]
		eta := a.eta[p]
		mu := a.mu[p]

		for i := range data {
			g := grad[i]
			if a.weightDecay != 0 {
				g += a.weightDecay * data[i]
			}
			// decay term (PyTorch: param.mul_(1 - lambda*eta))
			data[i] *= 1 - a.lambda*eta
			// gradient step
			data[i] -= eta * g
			// averaging
			if mu != 1 {
				ax[i] += mu * (data[i] - ax[i])
			} else {
				ax[i] = data[i]
			}
		}

		// update eta and mu schedules for next step
		a.eta[p] = a.lr / math.Pow(1+a.lambda*a.lr*t, a.alpha)
		a.mu[p] = 1 / math.Max(1, t-a.t0)
	}
}

// AveragedParam returns the averaged weight buffer (ax) for a parameter, or nil
// if the parameter has not been stepped yet.
func (a *ASGD) AveragedParam(p *tensor.Tensor) []float64 { return a.ax[p] }

// ZeroGrad zeros the gradients of all parameters.
func (a *ASGD) ZeroGrad() { zeroGradAll(a.params) }

// Parameters returns the parameter list.
func (a *ASGD) Parameters() []*tensor.Tensor { return a.params }

// LR returns the current learning rate.
func (a *ASGD) LR() float64 { return a.lr }

// SetLR updates the learning rate.
func (a *ASGD) SetLR(lr float64) { a.lr = lr }
