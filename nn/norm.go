package nn

import (
	"gonn/tensor"
)

// LayerNorm normalizes over the last len(NormalizedShape) dims.
type LayerNorm struct {
	NormalizedShape []int
	Eps             float64
	Weight          *tensor.Tensor // affine gain
	Bias            *tensor.Tensor // affine shift
}

// NewLayerNorm constructs a LayerNorm over the trailing `dim` axis.
func NewLayerNorm(dim int) *LayerNorm {
	return NewLayerNormShape([]int{dim})
}

// NewLayerNormShape allows normalizing over multiple trailing dims.
func NewLayerNormShape(shape []int) *LayerNorm {
	n := 1
	for _, d := range shape {
		n *= d
	}
	w := tensor.Ones(shape...).SetRequiresGrad(true)
	b := tensor.Zeros(shape...).SetRequiresGrad(true)
	return &LayerNorm{NormalizedShape: append([]int(nil), shape...), Eps: 1e-5, Weight: w, Bias: b}
}

// Forward normalizes the trailing dims.
func (ln *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Number of trailing dims to normalize over.
	nDims := len(ln.NormalizedShape)
	rank := len(x.Shape)
	// Reduce one axis at a time from the last.
	mean := x
	for i := 0; i < nDims; i++ {
		mean = mean.MeanAxis(rank-1-i, true)
	}
	xc := x.Sub(mean)
	sq := xc.Square()
	v := sq
	for i := 0; i < nDims; i++ {
		v = v.MeanAxis(rank-1-i, true)
	}
	std := v.AddScalar(ln.Eps).Sqrt()
	norm := xc.Div(std)
	return norm.Mul(ln.Weight).Add(ln.Bias)
}

// Parameters returns weight and bias.
func (ln *LayerNorm) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{ln.Weight, ln.Bias}
}

// RMSNorm normalizes by RMS (no mean subtraction).
type RMSNorm struct {
	Dim    int
	Eps    float64
	Weight *tensor.Tensor
}

// NewRMSNorm constructs an RMSNorm over the last dim of size `dim`.
func NewRMSNorm(dim int) *RMSNorm {
	w := tensor.Ones(dim).SetRequiresGrad(true)
	return &RMSNorm{Dim: dim, Eps: 1e-6, Weight: w}
}

// Forward applies RMSNorm.
func (r *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(x.Shape)
	ms := x.Square().MeanAxis(rank-1, true)
	rms := ms.AddScalar(r.Eps).Sqrt()
	return x.Div(rms).Mul(r.Weight)
}

// Parameters returns the weight.
func (r *RMSNorm) Parameters() []*tensor.Tensor { return []*tensor.Tensor{r.Weight} }

// BatchNorm1d normalizes (N, C) or (N, C, L) over the batch (and length) per channel.
type BatchNorm1d struct {
	NumFeatures int
	Eps         float64
	Momentum    float64
	Weight      *tensor.Tensor // (C,)
	Bias        *tensor.Tensor // (C,)
	RunMean     *tensor.Tensor // (C,)
	RunVar      *tensor.Tensor // (C,)
	Training    bool
}

// NewBatchNorm1d constructs a BatchNorm1d with C features.
func NewBatchNorm1d(c int) *BatchNorm1d {
	return &BatchNorm1d{
		NumFeatures: c,
		Eps:         1e-5,
		Momentum:    0.1,
		Weight:      tensor.Ones(c).SetRequiresGrad(true),
		Bias:        tensor.Zeros(c).SetRequiresGrad(true),
		RunMean:     tensor.Zeros(c),
		RunVar:      tensor.Ones(c),
		Training:    true,
	}
}

// Forward applies batch norm over the channel dim.
func (b *BatchNorm1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	// Accept (N, C) or (N, C, L). For (N,C,L), reduce over N and L.
	switch len(x.Shape) {
	case 2:
		return bnForward2D(x, b.Weight, b.Bias, b.RunMean, b.RunVar, b.Training, b.Eps, b.Momentum)
	case 3:
		return bnForward3D(x, b.Weight, b.Bias, b.RunMean, b.RunVar, b.Training, b.Eps, b.Momentum)
	default:
		panic("BatchNorm1d: expected 2D or 3D input")
	}
}

// Parameters returns weight and bias.
func (b *BatchNorm1d) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{b.Weight, b.Bias}
}

// SetTraining toggles train/eval.
func (b *BatchNorm1d) SetTraining(t bool) { b.Training = t }

// BatchNorm2d normalizes (N, C, H, W) per channel over N, H, W.
type BatchNorm2d struct {
	NumFeatures int
	Eps         float64
	Momentum    float64
	Weight      *tensor.Tensor
	Bias        *tensor.Tensor
	RunMean     *tensor.Tensor
	RunVar      *tensor.Tensor
	Training    bool
}

// NewBatchNorm2d constructs a BatchNorm2d with C features.
func NewBatchNorm2d(c int) *BatchNorm2d {
	return &BatchNorm2d{
		NumFeatures: c,
		Eps:         1e-5,
		Momentum:    0.1,
		Weight:      tensor.Ones(c).SetRequiresGrad(true),
		Bias:        tensor.Zeros(c).SetRequiresGrad(true),
		RunMean:     tensor.Zeros(c),
		RunVar:      tensor.Ones(c),
		Training:    true,
	}
}

// Forward applies BN over (N, H, W) per channel.
func (b *BatchNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("BatchNorm2d: expected 4D input")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	// Move channel to last: (N, H, W, C)
	xp := x.Permute(0, 2, 3, 1).Reshape(N*H*W, C)
	out := bnForward2D(xp, b.Weight, b.Bias, b.RunMean, b.RunVar, b.Training, b.Eps, b.Momentum)
	return out.Reshape(N, H, W, C).Permute(0, 3, 1, 2)
}

// Parameters returns weight and bias.
func (b *BatchNorm2d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{b.Weight, b.Bias} }

// SetTraining toggles train/eval.
func (b *BatchNorm2d) SetTraining(t bool) { b.Training = t }

// bnForward2D normalizes x of shape (N, C) over axis 0.
func bnForward2D(x, w, bb, runMean, runVar *tensor.Tensor, training bool, eps, momentum float64) *tensor.Tensor {
	if training {
		mean := x.MeanAxis(0, true) // (1, C)
		xc := x.Sub(mean)
		v := xc.Square().MeanAxis(0, true)
		std := v.AddScalar(eps).Sqrt()
		norm := xc.Div(std)
		out := norm.Mul(w).Add(bb)
		// update running stats (no grad)
		for c := 0; c < runMean.Shape[0]; c++ {
			runMean.Data[c] = (1-momentum)*runMean.Data[c] + momentum*mean.Data[c]
			runVar.Data[c] = (1-momentum)*runVar.Data[c] + momentum*v.Data[c]
		}
		return out
	}
	// eval: use running stats
	mean := runMean.Reshape(1, runMean.Shape[0])
	v := runVar.Reshape(1, runVar.Shape[0])
	xc := x.Sub(mean)
	std := v.AddScalar(eps).Sqrt()
	norm := xc.Div(std)
	return norm.Mul(w).Add(bb)
}

// bnForward3D handles (N, C, L) by reducing over N and L.
func bnForward3D(x, w, bb, runMean, runVar *tensor.Tensor, training bool, eps, momentum float64) *tensor.Tensor {
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	xp := x.Permute(0, 2, 1).Reshape(N*L, C)
	out := bnForward2D(xp, w, bb, runMean, runVar, training, eps, momentum)
	return out.Reshape(N, L, C).Permute(0, 2, 1)
}

// GroupNorm normalizes (N, C, *) by splitting C into G groups.
type GroupNorm struct {
	NumGroups   int
	NumChannels int
	Eps         float64
	Weight      *tensor.Tensor // (C,)
	Bias        *tensor.Tensor // (C,)
}

// NewGroupNorm constructs a GroupNorm with G groups over C channels.
func NewGroupNorm(numGroups, numChannels int) *GroupNorm {
	if numChannels%numGroups != 0 {
		panic("GroupNorm: numChannels must be divisible by numGroups")
	}
	return &GroupNorm{
		NumGroups:   numGroups,
		NumChannels: numChannels,
		Eps:         1e-5,
		Weight:      tensor.Ones(numChannels).SetRequiresGrad(true),
		Bias:        tensor.Zeros(numChannels).SetRequiresGrad(true),
	}
}

// Forward applies GroupNorm over (N, C, *).
func (g *GroupNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) < 2 {
		panic("GroupNorm: need at least 2D input (N, C, ...)")
	}
	N, C := x.Shape[0], x.Shape[1]
	if C != g.NumChannels {
		panic("GroupNorm: channel mismatch")
	}
	rest := 1
	for i := 2; i < len(x.Shape); i++ {
		rest *= x.Shape[i]
	}
	cpg := C / g.NumGroups
	// Reshape to (N, G, cpg*rest)
	flat := x.Reshape(N, g.NumGroups, cpg*rest)
	mean := flat.MeanAxis(2, true)
	xc := flat.Sub(mean)
	v := xc.Square().MeanAxis(2, true)
	std := v.AddScalar(g.Eps).Sqrt()
	norm := xc.Div(std)
	// Restore to (N, C, ...) shape.
	norm = norm.Reshape(x.Shape...)
	// Apply per-channel affine: weight/bias have shape (C,), need broadcasting.
	// Build (1, C, 1, 1, ...) shape.
	affineShape := make([]int, len(x.Shape))
	for i := range affineShape {
		affineShape[i] = 1
	}
	affineShape[1] = C
	w := g.Weight.Reshape(affineShape...)
	b := g.Bias.Reshape(affineShape...)
	return norm.Mul(w).Add(b)
}

// Parameters returns weight and bias.
func (g *GroupNorm) Parameters() []*tensor.Tensor { return []*tensor.Tensor{g.Weight, g.Bias} }
