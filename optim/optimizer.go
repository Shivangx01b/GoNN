package optim

import "gonn/tensor"

// Optimizer is the common interface implemented by all optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
	Parameters() []*tensor.Tensor
	LR() float64
	SetLR(float64)
}

// zeroGradAll zeros the gradient buffers of every parameter in place.
func zeroGradAll(params []*tensor.Tensor) {
	for _, p := range params {
		if p == nil || p.Grad == nil {
			continue
		}
		for i := range p.Grad.Data {
			p.Grad.Data[i] = 0
		}
	}
}
