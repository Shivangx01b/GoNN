package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// Linear implements y = x @ W^T + b.
type Linear struct {
	Base
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
	l := &Linear{InFeatures: in, OutFeatures: out}
	l.Weight = l.reg("weight", tensor.New(wData, out, in).SetRequiresGrad(true))
	if bias {
		bData := make([]float64, out)
		for i := range bData {
			bData[i] = -bound + rand.Float64()*(2*bound)
		}
		l.Bias = l.reg("bias", tensor.New(bData, out).SetRequiresGrad(true))
	}
	return l
}

// Forward computes x @ W^T (+ b). x can be (..., InFeatures); we treat the last
// dim as features and matmul as 2D after flattening the leading dims.
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
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
	outShape := append([]int(nil), origShape[:len(origShape)-1]...)
	outShape = append(outShape, l.OutFeatures)
	return y.Reshape(outShape...)
}

// Identity is a no-op module: Forward returns its input unchanged.
type Identity struct{ Base }

// NewIdentity constructs an Identity module.
func NewIdentity() *Identity { return &Identity{} }

// Forward returns x unchanged.
func (*Identity) Forward(x *tensor.Tensor) *tensor.Tensor { return x }

// Flatten flattens the dims in the inclusive range [StartDim, EndDim] into a
// single dimension, mirroring torch.nn.Flatten. Negative dims count from the
// end. StartDim=1, EndDim=-1 flattens everything but the batch.
type Flatten struct {
	Base
	StartDim int
	EndDim   int
}

// NewFlatten builds a Flatten over [start, end]. Use end = -1 for the last dim.
func NewFlatten(start, end int) *Flatten {
	return &Flatten{StartDim: start, EndDim: end}
}

// Forward collapses dims [StartDim, EndDim] into one.
func (f *Flatten) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(x.Shape)
	start := f.StartDim
	end := f.EndDim
	if start < 0 {
		start += rank
	}
	if end < 0 {
		end += rank
	}
	if start < 0 || end >= rank || start > end {
		panic("Flatten.Forward: invalid dim range")
	}
	newShape := make([]int, 0, rank)
	newShape = append(newShape, x.Shape[:start]...)
	merged := 1
	for d := start; d <= end; d++ {
		merged *= x.Shape[d]
	}
	newShape = append(newShape, merged)
	newShape = append(newShape, x.Shape[end+1:]...)
	return x.Reshape(newShape...)
}

// Unflatten expands a single dimension Dim into the multi-dim shape Sizes,
// mirroring torch.nn.Unflatten. The product of Sizes must equal the size of Dim.
type Unflatten struct {
	Base
	Dim   int
	Sizes []int
}

// NewUnflatten builds an Unflatten that reshapes dim into sizes.
func NewUnflatten(dim int, sizes ...int) *Unflatten {
	return &Unflatten{Dim: dim, Sizes: sizes}
}

// Forward expands Dim into Sizes.
func (u *Unflatten) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(x.Shape)
	dim := u.Dim
	if dim < 0 {
		dim += rank
	}
	if dim < 0 || dim >= rank {
		panic("Unflatten.Forward: invalid dim")
	}
	prod := 1
	for _, s := range u.Sizes {
		prod *= s
	}
	if prod != x.Shape[dim] {
		panic("Unflatten.Forward: product of sizes must equal the size of dim")
	}
	newShape := make([]int, 0, rank-1+len(u.Sizes))
	newShape = append(newShape, x.Shape[:dim]...)
	newShape = append(newShape, u.Sizes...)
	newShape = append(newShape, x.Shape[dim+1:]...)
	return x.Reshape(newShape...)
}

// Bilinear implements y = x1^T A x2 + b, mirroring torch.nn.Bilinear.
// Weight has shape (OutFeatures, In1, In2); Bias has shape (OutFeatures,).
// For inputs x1 (N, In1) and x2 (N, In2), output is (N, OutFeatures) with
//
//	y[n,o] = sum_{i,j} x1[n,i] * Weight[o,i,j] * x2[n,j] + b[o].
//
// Bilinear takes two inputs, so it satisfies Child but not Module.
type Bilinear struct {
	Base
	In1Features int
	In2Features int
	OutFeatures int
	Weight      *tensor.Tensor // (Out, In1, In2)
	Bias        *tensor.Tensor // (Out,) or nil
}

// NewBilinear creates a Bilinear layer with uniform init in [-bound, bound]
// where bound = 1/sqrt(In1), matching PyTorch's default initialization.
func NewBilinear(in1, in2, out int, bias bool) *Bilinear {
	bound := 1.0 / math.Sqrt(float64(in1))
	wData := make([]float64, out*in1*in2)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	b := &Bilinear{In1Features: in1, In2Features: in2, OutFeatures: out}
	b.Weight = b.reg("weight", tensor.New(wData, out, in1, in2).SetRequiresGrad(true))
	if bias {
		bData := make([]float64, out)
		for i := range bData {
			bData[i] = -bound + rand.Float64()*(2*bound)
		}
		b.Bias = b.reg("bias", tensor.New(bData, out).SetRequiresGrad(true))
	}
	return b
}

// Forward computes y = x1^T A x2 + b for batched inputs.
func (b *Bilinear) Forward(x1, x2 *tensor.Tensor) *tensor.Tensor {
	out := b.OutFeatures
	// For each output channel o: tmp = x1 @ W[o] -> (N, In2); elementwise
	// multiply with x2 and sum over In2 -> (N, 1). Concatenate over channels.
	cols := make([]*tensor.Tensor, out)
	for o := 0; o < out; o++ {
		wo := b.sliceWeight(o)          // (In1, In2)
		tmp := x1.MatMul(wo)            // (N, In2)
		prod := tmp.Mul(x2)             // (N, In2)
		cols[o] = prod.SumAxis(1, true) // (N, 1)
	}
	y := tensor.Concat(1, cols...) // (N, out)
	if b.Bias != nil {
		y = y.Add(b.Bias.Reshape(1, out))
	}
	return y
}

// sliceWeight extracts the (In1, In2) weight block for output channel o with
// autograd preserved (IndexSelect has a scatter-add backward).
func (b *Bilinear) sliceWeight(o int) *tensor.Tensor {
	return b.Weight.
		IndexSelect(0, tensor.New([]float64{float64(o)}, 1)).
		Reshape(b.In1Features, b.In2Features)
}
