package nn

import (
	"math/rand"

	"gonn/tensor"
)

// Dropout zeros elements with probability P during training. Identity at
// eval. Train/eval mode comes from the embedded Base and propagates through
// containers: model.Eval() switches every Dropout in the tree.
type Dropout struct {
	Base
	P float64
}

// NewDropout returns a Dropout module (training mode by default).
func NewDropout(p float64) *Dropout { return &Dropout{P: p} }

// Forward applies dropout in training mode; otherwise returns x unchanged.
func (d *Dropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !d.Training() || d.P <= 0 {
		return x
	}
	if d.P >= 1 {
		return x.MulScalar(0)
	}
	keep := 1.0 - d.P
	scale := 1.0 / keep
	mask := tensor.Zeros(x.Shape...)
	for i := range mask.Data {
		if rand.Float64() < keep {
			mask.Data[i] = scale
		}
	}
	return x.Mul(mask)
}
