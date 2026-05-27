package nn

import "gonn/tensor"

// Module is the basic neural-network building block.
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}

// Sequential chains a list of modules.
type Sequential struct {
	Layers []Module
}

// NewSequential builds a Sequential from the given modules.
func NewSequential(layers ...Module) *Sequential {
	return &Sequential{Layers: layers}
}

// Forward applies each layer in order.
func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

// Parameters returns all parameters from contained modules.
func (s *Sequential) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range s.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// Add appends a module.
func (s *Sequential) Add(m Module) *Sequential {
	s.Layers = append(s.Layers, m)
	return s
}
