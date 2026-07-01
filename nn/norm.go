package nn

import (
	"fmt"

	"gonn/tensor"
)

// All normalization layers share one functional core, normalizeAxes, and one
// options type. Defaults preserve the historical constants: eps 1e-5
// (LayerNorm/BatchNorm/GroupNorm/InstanceNorm), 1e-6 (RMSNorm), BatchNorm
// momentum 0.1.

// NormOpt configures a normalization layer.
type NormOpt func(*normOpts)

type normOpts struct {
	eps      float64
	momentum float64
	affine   bool
}

// WithEps overrides the numerical-stability epsilon.
func WithEps(e float64) NormOpt { return func(o *normOpts) { o.eps = e } }

// WithMomentum overrides the running-stats momentum (BatchNorm only; other
// norms ignore it).
func WithMomentum(m float64) NormOpt { return func(o *normOpts) { o.momentum = m } }

// WithAffine enables/disables the learnable gain+shift. Defaults: LayerNorm,
// BatchNorm, GroupNorm, RMSNorm true; InstanceNorm false (PyTorch parity).
func WithAffine(on bool) NormOpt { return func(o *normOpts) { o.affine = on } }

func resolveNormOpts(defEps float64, defAffine bool, opts []NormOpt) normOpts {
	o := normOpts{eps: defEps, momentum: 0.1, affine: defAffine}
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// normalizeAxes normalizes x over the given axes (reduced with keepDim, in
// descending axis order — matching the historical reduction order exactly):
//
//	norm = (x - mean) / sqrt(var + eps)        (subtractMean == true)
//	norm = x / sqrt(mean(x^2) + eps)           (subtractMean == false, RMS)
//
// It returns norm plus the (keepDim) mean and biased variance so BatchNorm
// can update its running statistics.
func normalizeAxes(x *tensor.Tensor, axes []int, eps float64, subtractMean bool) (norm, mean, biasedVar *tensor.Tensor) {
	// Reduce in descending axis order (last axis first), as the historical
	// per-layer implementations did.
	sorted := append([]int(nil), axes...)
	for i := 0; i < len(sorted); i++ { // insertion sort, descending
		for j := i; j > 0 && sorted[j] > sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	if subtractMean {
		mean = x
		for _, ax := range sorted {
			mean = mean.MeanAxis(ax, true)
		}
		xc := x.Sub(mean)
		biasedVar = xc.Square()
		for _, ax := range sorted {
			biasedVar = biasedVar.MeanAxis(ax, true)
		}
		std := biasedVar.AddScalar(eps).Sqrt()
		return xc.Div(std), mean, biasedVar
	}
	ms := x.Square()
	for _, ax := range sorted {
		ms = ms.MeanAxis(ax, true)
	}
	rms := ms.AddScalar(eps).Sqrt()
	return x.Div(rms), nil, ms
}

// trailingAxes returns the last n axes of a rank-r tensor.
func trailingAxes(rank, n int) []int {
	axes := make([]int, n)
	for i := 0; i < n; i++ {
		axes[i] = rank - n + i
	}
	return axes
}

// LayerNorm normalizes over the last len(NormalizedShape) dims.
type LayerNorm struct {
	Base
	NormalizedShape []int
	Eps             float64
	Weight          *tensor.Tensor // affine gain, or nil
	Bias            *tensor.Tensor // affine shift, or nil
}

// NewLayerNorm constructs a LayerNorm over the trailing `dim` axis.
func NewLayerNorm(dim int, opts ...NormOpt) *LayerNorm {
	return NewLayerNormShape([]int{dim}, opts...)
}

// NewLayerNormShape allows normalizing over multiple trailing dims.
func NewLayerNormShape(shape []int, opts ...NormOpt) *LayerNorm {
	o := resolveNormOpts(1e-5, true, opts)
	ln := &LayerNorm{NormalizedShape: append([]int(nil), shape...), Eps: o.eps}
	if o.affine {
		ln.Weight = ln.reg("weight", tensor.Ones(shape...).SetRequiresGrad(true))
		ln.Bias = ln.reg("bias", tensor.Zeros(shape...).SetRequiresGrad(true))
	}
	return ln
}

// Forward normalizes the trailing dims.
func (ln *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	norm, _, _ := normalizeAxes(x, trailingAxes(len(x.Shape), len(ln.NormalizedShape)), ln.Eps, true)
	if ln.Weight != nil {
		norm = norm.Mul(ln.Weight).Add(ln.Bias)
	}
	return norm
}

// RMSNorm normalizes by RMS (no mean subtraction) over the last dim.
type RMSNorm struct {
	Base
	Dim    int
	Eps    float64
	Weight *tensor.Tensor // or nil
}

// NewRMSNorm constructs an RMSNorm over the last dim of size `dim`.
func NewRMSNorm(dim int, opts ...NormOpt) *RMSNorm {
	o := resolveNormOpts(1e-6, true, opts)
	r := &RMSNorm{Dim: dim, Eps: o.eps}
	if o.affine {
		r.Weight = r.reg("weight", tensor.Ones(dim).SetRequiresGrad(true))
	}
	return r
}

// Forward applies RMSNorm.
func (r *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	norm, _, _ := normalizeAxes(x, []int{len(x.Shape) - 1}, r.Eps, false)
	if r.Weight != nil {
		norm = norm.Mul(r.Weight)
	}
	return norm
}

// GroupNorm normalizes (N, C, *) by splitting C into G groups.
type GroupNorm struct {
	Base
	NumGroups   int
	NumChannels int
	Eps         float64
	Weight      *tensor.Tensor // (C,) or nil
	Bias        *tensor.Tensor // (C,) or nil
}

// NewGroupNorm constructs a GroupNorm with G groups over C channels.
func NewGroupNorm(numGroups, numChannels int, opts ...NormOpt) *GroupNorm {
	if numChannels%numGroups != 0 {
		panic("GroupNorm: numChannels must be divisible by numGroups")
	}
	o := resolveNormOpts(1e-5, true, opts)
	g := &GroupNorm{NumGroups: numGroups, NumChannels: numChannels, Eps: o.eps}
	if o.affine {
		g.Weight = g.reg("weight", tensor.Ones(numChannels).SetRequiresGrad(true))
		g.Bias = g.reg("bias", tensor.Zeros(numChannels).SetRequiresGrad(true))
	}
	return g
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
	flat := x.Reshape(N, g.NumGroups, cpg*rest)
	norm, _, _ := normalizeAxes(flat, []int{2}, g.Eps, true)
	norm = norm.Reshape(x.Shape...)
	if g.Weight != nil {
		norm = norm.Mul(channelShape(g.Weight, len(x.Shape), C)).
			Add(channelShape(g.Bias, len(x.Shape), C))
	}
	return norm
}

// channelShape reshapes a (C,) tensor to (1, C, 1, ...) for broadcasting
// along the channel axis of a rank-`rank` tensor.
func channelShape(t *tensor.Tensor, rank, c int) *tensor.Tensor {
	shape := make([]int, rank)
	for i := range shape {
		shape[i] = 1
	}
	shape[1] = c
	return t.Reshape(shape...)
}

// instanceNormNd normalizes (N, C, spatial...) per (sample, channel) over
// all spatial axes. InstanceNorm1d/2d are thin wrappers fixing the rank.
type instanceNormNd struct {
	Base
	NumFeatures int
	Eps         float64
	Weight      *tensor.Tensor // (C,) or nil
	Bias        *tensor.Tensor // (C,) or nil
	rank        int            // expected input rank
}

func newInstanceNorm(rank, c int, opts []NormOpt) instanceNormNd {
	o := resolveNormOpts(1e-5, false, opts)
	in := instanceNormNd{NumFeatures: c, Eps: o.eps, rank: rank}
	if o.affine {
		in.Weight = in.reg("weight", tensor.Ones(c).SetRequiresGrad(true))
		in.Bias = in.reg("bias", tensor.Zeros(c).SetRequiresGrad(true))
	}
	return in
}

// Forward normalizes each (sample, channel) slice over the spatial axes.
func (in *instanceNormNd) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != in.rank {
		panic(fmt.Sprintf("InstanceNorm: expected %dD input, got shape %v", in.rank, x.Shape))
	}
	if x.Shape[1] != in.NumFeatures {
		panic("InstanceNorm: channel mismatch")
	}
	axes := make([]int, len(x.Shape)-2)
	for i := range axes {
		axes[i] = 2 + i
	}
	norm, _, _ := normalizeAxes(x, axes, in.Eps, true)
	if in.Weight != nil {
		norm = norm.Mul(channelShape(in.Weight, len(x.Shape), in.NumFeatures)).
			Add(channelShape(in.Bias, len(x.Shape), in.NumFeatures))
	}
	return norm
}

// InstanceNorm1d normalizes (N, C, L) per (sample, channel) over L.
// Affine gain/shift is off by default; enable with WithAffine(true).
type InstanceNorm1d struct{ instanceNormNd }

// NewInstanceNorm1d constructs an InstanceNorm1d.
func NewInstanceNorm1d(c int, opts ...NormOpt) *InstanceNorm1d {
	return &InstanceNorm1d{newInstanceNorm(3, c, opts)}
}

// InstanceNorm2d normalizes (N, C, H, W) per (sample, channel) over (H, W).
// Affine gain/shift is off by default; enable with WithAffine(true).
type InstanceNorm2d struct{ instanceNormNd }

// NewInstanceNorm2d constructs an InstanceNorm2d.
func NewInstanceNorm2d(c int, opts ...NormOpt) *InstanceNorm2d {
	return &InstanceNorm2d{newInstanceNorm(4, c, opts)}
}
