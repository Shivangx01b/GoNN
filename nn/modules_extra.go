package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// Identity is a no-op module: Forward returns its input unchanged.
type Identity struct{}

// Forward returns x unchanged.
func (Identity) Forward(x *tensor.Tensor) *tensor.Tensor { return x }

// Parameters returns nothing.
func (Identity) Parameters() []*tensor.Tensor { return nil }

// Flatten flattens the dims in the inclusive range [StartDim, EndDim] into a
// single dimension, mirroring torch.nn.Flatten. Negative dims count from the
// end. The default (StartDim=1, EndDim=-1) flattens everything but the batch.
type Flatten struct {
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

// Parameters returns nothing.
func (f *Flatten) Parameters() []*tensor.Tensor { return nil }

// Unflatten expands a single dimension Dim into the multi-dim shape Sizes,
// mirroring torch.nn.Unflatten. The product of Sizes must equal the size of Dim.
type Unflatten struct {
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

// Parameters returns nothing.
func (u *Unflatten) Parameters() []*tensor.Tensor { return nil }

// Bilinear implements y = x1^T A x2 + b, mirroring torch.nn.Bilinear.
// Weight has shape (OutFeatures, In1, In2); Bias has shape (OutFeatures,).
// For inputs x1 (N, In1) and x2 (N, In2), output is (N, OutFeatures) with
//
//	y[n,o] = sum_{i,j} x1[n,i] * Weight[o,i,j] * x2[n,j] + b[o].
type Bilinear struct {
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
	w := tensor.New(wData, out, in1, in2).SetRequiresGrad(true)
	b := &Bilinear{In1Features: in1, In2Features: in2, OutFeatures: out, Weight: w}
	if bias {
		bData := make([]float64, out)
		for i := range bData {
			bData[i] = -bound + rand.Float64()*(2*bound)
		}
		b.Bias = tensor.New(bData, out).SetRequiresGrad(true)
	}
	return b
}

// Forward computes y = x1^T A x2 + b for batched inputs.
func (b *Bilinear) Forward(x1, x2 *tensor.Tensor) *tensor.Tensor {
	out := b.OutFeatures
	// For each output channel o: tmp = x1 @ W[o]  -> (N, In2); then elementwise
	// multiply with x2 and sum over In2 -> (N, 1). Concatenate over channels.
	cols := make([]*tensor.Tensor, out)
	for o := 0; o < out; o++ {
		// W[o] : slice the (In1, In2) block; build via reshape of a gathered view.
		wo := b.sliceWeight(o) // (In1, In2)
		tmp := x1.MatMul(wo)   // (N, In2)
		prod := tmp.Mul(x2)    // (N, In2)
		s := prod.SumAxis(1, true) // (N, 1)
		cols[o] = s
	}
	y := tensor.Concat(1, cols...) // (N, out)
	if b.Bias != nil {
		y = y.Add(b.Bias.Reshape(1, out))
	}
	return y
}

// sliceWeight extracts the (In1, In2) weight block for output channel o as a
// differentiable view using a gather matmul so gradients flow to Weight.
func (b *Bilinear) sliceWeight(o int) *tensor.Tensor {
	in1, in2, out := b.In1Features, b.In2Features, b.OutFeatures
	// Flatten weight to (out, in1*in2) then select row o via a (1, out) selector.
	wFlat := b.Weight.Reshape(out, in1*in2)
	selData := make([]float64, out)
	selData[o] = 1
	sel := tensor.New(selData, 1, out)
	row := sel.MatMul(wFlat) // (1, in1*in2)
	return row.Reshape(in1, in2)
}

// Parameters returns the weight and bias (if present).
func (b *Bilinear) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{b.Weight}
	if b.Bias != nil {
		ps = append(ps, b.Bias)
	}
	return ps
}

// Softmin applies softmax(-x) along Axis, mirroring torch.nn.Softmin.
type Softmin struct{ Axis int }

// Forward returns softmax(-x).
func (s Softmin) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Neg().Softmax(s.Axis)
}

// Parameters returns nothing.
func (Softmin) Parameters() []*tensor.Tensor { return nil }

// Softmax2d applies softmax over the channel dimension (axis 1) of an NCHW
// tensor, mirroring torch.nn.Softmax2d.
type Softmax2d struct{}

// Forward applies channel-wise softmax over (N, C, H, W).
func (Softmax2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Softmax(1)
}

// Parameters returns nothing.
func (Softmax2d) Parameters() []*tensor.Tensor { return nil }

// CosineSimilarity computes the cosine similarity between x1 and x2 along Dim,
// mirroring torch.nn.CosineSimilarity. Eps avoids division by zero.
type CosineSimilarity struct {
	Dim int
	Eps float64
}

// NewCosineSimilarity builds a CosineSimilarity over dim with the given eps.
func NewCosineSimilarity(dim int, eps float64) *CosineSimilarity {
	if eps == 0 {
		eps = 1e-8
	}
	return &CosineSimilarity{Dim: dim, Eps: eps}
}

// Forward computes sum(x1*x2) / (||x1|| * ||x2||) along Dim, reducing that dim.
func (c *CosineSimilarity) Forward(x1, x2 *tensor.Tensor) *tensor.Tensor {
	dot := x1.Mul(x2).SumAxis(c.Dim, false)
	n1 := x1.Mul(x1).SumAxis(c.Dim, false).Sqrt()
	n2 := x2.Mul(x2).SumAxis(c.Dim, false).Sqrt()
	denom := n1.Mul(n2).AddScalar(c.Eps)
	return dot.Div(denom)
}

// Parameters returns nothing.
func (c *CosineSimilarity) Parameters() []*tensor.Tensor { return nil }

// PairwiseDistance computes the batched p-norm distance between x1 and x2 along
// the last dimension, mirroring torch.nn.PairwiseDistance.
type PairwiseDistance struct {
	P   float64
	Eps float64
}

// NewPairwiseDistance builds a PairwiseDistance with norm P and stabilizer Eps.
func NewPairwiseDistance(p, eps float64) *PairwiseDistance {
	if p == 0 {
		p = 2
	}
	if eps == 0 {
		eps = 1e-6
	}
	return &PairwiseDistance{P: p, Eps: eps}
}

// Forward computes (sum |x1 - x2 + eps|^p)^(1/p) along the last dim.
func (d *PairwiseDistance) Forward(x1, x2 *tensor.Tensor) *tensor.Tensor {
	diff := x1.Sub(x2).AddScalar(d.Eps)
	last := len(x1.Shape) - 1
	powed := diff.Abs().Pow(d.P).SumAxis(last, false)
	return powed.Pow(1.0 / d.P)
}

// Parameters returns nothing.
func (d *PairwiseDistance) Parameters() []*tensor.Tensor { return nil }
