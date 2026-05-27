package nn

import "gonn/tensor"

// PReLU is the parametric ReLU: y = max(0, x) + weight * min(0, x).
// NumParams is either 1 (shared slope across all channels) or num_channels
// (per-channel slope, where the channel axis is treated as axis 1 of x).
// The weight is initialized to 0.25 (PyTorch default).
type PReLU struct {
	NumParams int
	Weight    *tensor.Tensor // shape (NumParams,)
}

// NewPReLU creates a PReLU module. Pass numParams = 1 for a shared slope, or
// numParams = num_channels for per-channel slopes.
func NewPReLU(numParams int) *PReLU {
	if numParams < 1 {
		panic("NewPReLU: numParams must be >= 1")
	}
	w := tensor.Full(0.25, numParams).SetRequiresGrad(true)
	return &PReLU{NumParams: numParams, Weight: w}
}

// Forward computes y = max(0, x) + w * min(0, x).
// For NumParams == 1 the scalar weight broadcasts to all of x. For
// NumParams > 1 the weight is broadcast along axis 1 (the channel axis).
func (p *PReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	pos := x.ReLU()
	neg := x.Sub(pos) // negative part (or zero where x >= 0)
	var w *tensor.Tensor
	if p.NumParams == 1 {
		w = p.Weight
	} else {
		// Reshape weight so it broadcasts along the channel (axis 1) of x.
		// Build a shape [1, C, 1, 1, ...] matching x's rank.
		if len(x.Shape) < 2 {
			panic("PReLU.Forward: per-channel weight requires x with rank >= 2")
		}
		if x.Shape[1] != p.NumParams {
			panic("PReLU.Forward: channel dim does not match NumParams")
		}
		shape := make([]int, len(x.Shape))
		for i := range shape {
			shape[i] = 1
		}
		shape[1] = p.NumParams
		w = p.Weight.Reshape(shape...)
	}
	return pos.Add(w.Mul(neg))
}

// Parameters returns the slope parameter.
func (p *PReLU) Parameters() []*tensor.Tensor { return []*tensor.Tensor{p.Weight} }
