package optim

import "gonn/tensor"

// SGD implements stochastic gradient descent with optional momentum,
// weight decay, and Nesterov acceleration.
type SGD struct {
	params       []*tensor.Tensor
	lr           float64
	momentum     float64
	weightDecay  float64
	nesterov     bool
	velocity     map[*tensor.Tensor][]float64
}

// SGDOption configures an SGD optimizer.
type SGDOption func(*SGD)

// WithMomentum sets the momentum coefficient.
func WithMomentum(m float64) SGDOption { return func(s *SGD) { s.momentum = m } }

// WithSGDWeightDecay sets the L2 weight decay coefficient.
func WithSGDWeightDecay(wd float64) SGDOption { return func(s *SGD) { s.weightDecay = wd } }

// WithNesterov enables Nesterov-style momentum.
func WithNesterov(n bool) SGDOption { return func(s *SGD) { s.nesterov = n } }

// NewSGD constructs an SGD optimizer.
func NewSGD(params []*tensor.Tensor, lr float64, opts ...SGDOption) *SGD {
	s := &SGD{
		params:   params,
		lr:       lr,
		velocity: make(map[*tensor.Tensor][]float64),
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step performs a single optimization step.
func (s *SGD) Step() {
	for _, p := range s.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		var v []float64
		if s.momentum != 0 {
			v = s.velocity[p]
			if v == nil {
				v = make([]float64, len(data))
				s.velocity[p] = v
			}
		}
		for i := range data {
			g := grad[i]
			if s.weightDecay != 0 {
				g += s.weightDecay * data[i]
			}
			if s.momentum != 0 {
				v[i] = s.momentum*v[i] + g
				var update float64
				if s.nesterov {
					update = g + s.momentum*v[i]
				} else {
					update = v[i]
				}
				data[i] -= s.lr * update
			} else {
				data[i] -= s.lr * g
			}
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (s *SGD) ZeroGrad() { zeroGradAll(s.params) }

// Parameters returns the parameter list.
func (s *SGD) Parameters() []*tensor.Tensor { return s.params }

// LR returns the current learning rate.
func (s *SGD) LR() float64 { return s.lr }

// SetLR updates the learning rate.
func (s *SGD) SetLR(lr float64) { s.lr = lr }
