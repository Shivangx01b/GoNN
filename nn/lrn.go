package nn

import (
	"fmt"

	"gonn/tensor"
)

// LocalResponseNorm applies local response normalization across channels,
// matching torch.nn.LocalResponseNorm:
//
//	b_c = a_c / (k + alpha/n * sum_{c'} a_{c'}^2)^beta
//
// where the sum runs over the size-n window of channels around c. Like
// PyTorch's implementation (zero-pad by size/2 before and (size-1)/2 after,
// then average-pool over channels), the window is [c - n/2, c + (n-1)/2]
// clamped to valid channels — out-of-range channels contribute zero — and the
// divisor is always n regardless of clamping. Defaults: alpha 1e-4, beta
// 0.75, k 1.0.
//
// Input is (N, C, spatial...) with rank >= 3 (PyTorch parity); any number of
// spatial dims is supported. The channel-window sum is expressed as a (C, C)
// 0/1 band-matrix MatMul over differentiable ops, so autograd works by
// construction and the layer dispatches to cuBLAS automatically on -tags
// cuda builds.
type LocalResponseNorm struct {
	Base
	Size  int
	Alpha float64
	Beta  float64
	K     float64
}

// LRNOpt configures LocalResponseNorm.
type LRNOpt func(*lrnOpts)

type lrnOpts struct {
	alpha, beta, k float64
}

// WithLRNAlpha overrides the multiplicative factor alpha (default 1e-4).
func WithLRNAlpha(a float64) LRNOpt { return func(o *lrnOpts) { o.alpha = a } }

// WithLRNBeta overrides the exponent beta (default 0.75).
func WithLRNBeta(b float64) LRNOpt { return func(o *lrnOpts) { o.beta = b } }

// WithLRNK overrides the additive constant k (default 1.0).
func WithLRNK(k float64) LRNOpt { return func(o *lrnOpts) { o.k = k } }

// NewLocalResponseNorm constructs a LocalResponseNorm over a window of
// `size` neighbouring channels.
func NewLocalResponseNorm(size int, opts ...LRNOpt) *LocalResponseNorm {
	if size <= 0 {
		panic("LocalResponseNorm: size must be positive")
	}
	o := lrnOpts{alpha: 1e-4, beta: 0.75, k: 1.0}
	for _, fn := range opts {
		fn(&o)
	}
	return &LocalResponseNorm{Size: size, Alpha: o.alpha, Beta: o.beta, K: o.k}
}

// Forward applies LRN across the channel axis of (N, C, spatial...).
func (l *LocalResponseNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) < 3 {
		panic(fmt.Sprintf("LocalResponseNorm: expected 3D or higher input (N, C, ...), got shape %v", x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	rest := 1
	for i := 2; i < len(x.Shape); i++ {
		rest *= x.Shape[i]
	}

	// (C, C) band matrix: band[c][c'] = 1 for c' in [c-n/2, c+(n-1)/2],
	// clamped to the valid channel range (zeros outside == zero padding).
	lo, hi := l.Size/2, (l.Size-1)/2
	band := tensor.Zeros(C, C)
	for c := 0; c < C; c++ {
		for cp := c - lo; cp <= c+hi; cp++ {
			if cp >= 0 && cp < C {
				band.Data[c*C+cp] = 1
			}
		}
	}

	// sum[n, c, s] = sum_{c'} band[c, c'] * x[n, c', s]^2 via one batched
	// MatMul: (C, C) @ (N, C, rest) broadcasts over the batch dim.
	sq := x.Square().Reshape(N, C, rest)
	sum := band.MatMul(sq).Reshape(x.Shape...)
	denom := sum.MulScalar(l.Alpha / float64(l.Size)).AddScalar(l.K).Pow(l.Beta)
	return x.Div(denom)
}
