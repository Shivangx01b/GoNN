package nn

import (
	"fmt"

	"gonn/tensor"
)

// batchNormNd is the shared BatchNorm core: normalize per channel over every
// other axis using batch statistics in training mode, running statistics in
// eval mode. BatchNorm1d/2d are thin wrappers fixing the accepted ranks.
// RunMean/RunVar are registered buffers (serialized, never trained).
type batchNormNd struct {
	Base
	NumFeatures int
	Eps         float64
	Momentum    float64
	Weight      *tensor.Tensor // (C,)
	Bias        *tensor.Tensor // (C,)
	RunMean     *tensor.Tensor // (C,)
	RunVar      *tensor.Tensor // (C,)
}

func newBatchNorm(c int, opts []NormOpt) batchNormNd {
	o := resolveNormOpts(1e-5, true, opts)
	b := batchNormNd{NumFeatures: c, Eps: o.eps, Momentum: o.momentum}
	if o.affine {
		b.Weight = b.reg("weight", tensor.Ones(c).SetRequiresGrad(true))
		b.Bias = b.reg("bias", tensor.Zeros(c).SetRequiresGrad(true))
	} else {
		// Non-affine still needs unit gain / zero shift for the shared path.
		b.Weight = tensor.Ones(c)
		b.Bias = tensor.Zeros(c)
	}
	b.RunMean = b.regBuf("running_mean", tensor.Zeros(c))
	b.RunVar = b.regBuf("running_var", tensor.Ones(c))
	return b
}

// forward2D normalizes x of shape (N, C) over axis 0.
func (b *batchNormNd) forward2D(x *tensor.Tensor) *tensor.Tensor {
	if b.Training() {
		norm, mean, v := normalizeAxes(x, []int{0}, b.Eps, true)
		out := norm.Mul(b.Weight).Add(b.Bias)
		b.updateRunningStats(x.Shape[0], mean, v)
		return out
	}
	// eval: use running stats
	mean := b.RunMean.Reshape(1, b.NumFeatures)
	v := b.RunVar.Reshape(1, b.NumFeatures)
	xc := x.Sub(mean)
	std := v.AddScalar(b.Eps).Sqrt()
	norm := xc.Div(std)
	return norm.Mul(b.Weight).Add(b.Bias)
}

// updateRunningStats folds the batch statistics into the running estimates
// (no grad). PyTorch normalizes with the biased variance but tracks the
// unbiased variance (Bessel's correction) in the running estimate used at
// eval time.
func (b *batchNormNd) updateRunningStats(n int, mean, biasedVar *tensor.Tensor) {
	unbiased := 1.0
	if n > 1 {
		unbiased = float64(n) / float64(n-1)
	}
	for c := 0; c < b.NumFeatures; c++ {
		b.RunMean.Data[c] = (1-b.Momentum)*b.RunMean.Data[c] + b.Momentum*mean.Data[c]
		b.RunVar.Data[c] = (1-b.Momentum)*b.RunVar.Data[c] + b.Momentum*biasedVar.Data[c]*unbiased
	}
}

// forwardChannelsFirst handles rank >= 3 inputs (N, C, spatial...) by moving
// the channel axis last, flattening to (N*spatial, C), and running the 2D
// path.
func (b *batchNormNd) forwardChannelsFirst(x *tensor.Tensor) *tensor.Tensor {
	rank := len(x.Shape)
	N, C := x.Shape[0], x.Shape[1]
	rest := 1
	for i := 2; i < rank; i++ {
		rest *= x.Shape[i]
	}
	// (N, C, rest) -> (N, rest, C)
	xp := x.Reshape(N, C, rest).Permute(0, 2, 1).Reshape(N*rest, C)
	out := b.forward2D(xp)
	return out.Reshape(N, rest, C).Permute(0, 2, 1).Reshape(x.Shape...)
}

// BatchNorm1d normalizes (N, C) or (N, C, L) per channel over the batch
// (and length).
type BatchNorm1d struct{ batchNormNd }

// NewBatchNorm1d constructs a BatchNorm1d with C features.
func NewBatchNorm1d(c int, opts ...NormOpt) *BatchNorm1d {
	return &BatchNorm1d{newBatchNorm(c, opts)}
}

// Forward applies batch norm over the channel dim.
func (b *BatchNorm1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	switch len(x.Shape) {
	case 2:
		return b.forward2D(x)
	case 3:
		return b.forwardChannelsFirst(x)
	default:
		panic(fmt.Sprintf("BatchNorm1d: expected 2D or 3D input, got shape %v", x.Shape))
	}
}

// BatchNorm2d normalizes (N, C, H, W) per channel over N, H, W.
type BatchNorm2d struct{ batchNormNd }

// NewBatchNorm2d constructs a BatchNorm2d with C features.
func NewBatchNorm2d(c int, opts ...NormOpt) *BatchNorm2d {
	return &BatchNorm2d{newBatchNorm(c, opts)}
}

// Forward applies BN over (N, H, W) per channel.
func (b *BatchNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic(fmt.Sprintf("BatchNorm2d: expected 4D input, got shape %v", x.Shape))
	}
	return b.forwardChannelsFirst(x)
}
