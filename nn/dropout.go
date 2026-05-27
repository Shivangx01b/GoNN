package nn

import (
	"math/rand"

	"gonn/tensor"
)

// Dropout zeros elements with probability P during training. Identity at eval.
type Dropout struct {
	P        float64
	Training bool
}

// NewDropout returns a Dropout module set to training mode.
func NewDropout(p float64) *Dropout { return &Dropout{P: p, Training: true} }

// Forward applies dropout if Training; otherwise returns x unchanged.
func (d *Dropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !d.Training || d.P <= 0 {
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

// Parameters returns nothing (Dropout has no learnable params).
func (d *Dropout) Parameters() []*tensor.Tensor { return nil }

// SetTraining toggles train/eval mode.
func (d *Dropout) SetTraining(b bool) { d.Training = b }
