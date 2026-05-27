package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// Linear implements y = x @ W^T + b.
type Linear struct {
	InFeatures  int
	OutFeatures int
	Weight      *tensor.Tensor // shape (OutFeatures, InFeatures)
	Bias        *tensor.Tensor // shape (OutFeatures,) or nil
}

// NewLinear creates a Linear layer with He/Kaiming uniform init.
func NewLinear(in, out int, bias bool) *Linear {
	bound := math.Sqrt(1.0 / float64(in))
	wData := make([]float64, out*in)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, out, in).SetRequiresGrad(true)
	l := &Linear{InFeatures: in, OutFeatures: out, Weight: w}
	if bias {
		bData := make([]float64, out)
		for i := range bData {
			bData[i] = -bound + rand.Float64()*(2*bound)
		}
		l.Bias = tensor.New(bData, out).SetRequiresGrad(true)
	}
	return l
}

// Forward computes x @ W^T (+ b). x can be (..., InFeatures); we treat the last
// dim as features and matmul as 2D after flattening the leading dims.
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Flatten leading dims to a single batch dim for the 2D matmul.
	origShape := x.Shape
	feat := origShape[len(origShape)-1]
	if feat != l.InFeatures {
		panic("Linear.Forward: input last dim does not match InFeatures")
	}
	batch := 1
	for i := 0; i < len(origShape)-1; i++ {
		batch *= origShape[i]
	}
	x2 := x.Reshape(batch, feat)
	// y = x2 @ W^T -> (batch, out)
	y := x2.MatMul(l.Weight.Transpose())
	if l.Bias != nil {
		y = y.Add(l.Bias)
	}
	// Restore leading shape.
	outShape := append([]int(nil), origShape[:len(origShape)-1]...)
	outShape = append(outShape, l.OutFeatures)
	return y.Reshape(outShape...)
}

// Parameters returns weight and bias (if present).
func (l *Linear) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{l.Weight}
	if l.Bias != nil {
		ps = append(ps, l.Bias)
	}
	return ps
}
