package nn

import (
	"fmt"
	"math"

	"gonn/tensor"
)

// LP (power-average) pooling, PyTorch torch.nn.LPPool{1,2,3}d semantics:
//
//	f(X) = (sum_{x in X} x^p)^(1/p)
//
// over each kernel window — note the SUM (not mean) inside the root, exactly
// as PyTorch computes it (avg_pool of x^p rescaled by the window size).
// p = 1 is sum pooling; p -> inf approaches max pooling but infinite p is not
// supported (use MaxPool); p must be finite and > 0.
//
// Sign handling matches PyTorch: none. Negative inputs raised to a fractional
// p produce NaN there and here alike (math.Pow(neg, frac) = NaN). Even-integer
// p is always safe.
//
// Gradient flows through the differentiable composition
// win.Pow(p).SumAxis(...).Pow(1/p). One documented deviation: when the window
// sum of x^p is exactly zero (e.g. an all-zero window with p=2), PyTorch
// defines the gradient as zero while this composition yields a non-finite
// gradient (d/ds s^(1/p) at s=0 diverges for p>1).
//
// Windows are extracted with the same gatherMatrix machinery as poolNd
// (pad 0, dilation 1); stride defaults to the kernel, as in PyTorch.

// lpPoolNd is the shared LP-pooling core. It reuses poolNd's window gather
// and cache; poolNd's mode field is unused because Forward is overridden.
type lpPoolNd struct {
	poolNd
	NormType float64
}

func (p *lpPoolNd) initLP(rank int, normType float64, kernel int, opts []PoolOpt) {
	if !(normType > 0) || math.IsInf(normType, 1) {
		panic(fmt.Sprintf("nn: LPPool norm type must be finite and > 0, got %v", normType))
	}
	p.initPool(rank, kernel, avgPool, opts) // mode unused; Forward is LP
	p.NormType = normType
}

// Forward gathers every window and reduces it to (sum x^p)^(1/p).
func (p *lpPoolNd) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(p.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: LP pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	g, out, numWin, winSize := p.gatherFor(x.Shape[2:])

	win := x.Reshape(N*C, prodInts(x.Shape[2:])).
		MatMul(g.Transpose()).
		Reshape(N*C, numWin, winSize)
	r := win.Pow(p.NormType).SumAxis(2, false).Pow(1 / p.NormType)
	return r.Reshape(append([]int{N, C}, out...)...)
}

// LPPool1d applies power-average pooling on (N, C, L) inputs.
type LPPool1d struct{ lpPoolNd }

// NewLPPool1d creates an LPPool1d with norm type normType (p > 0, finite);
// stride defaults to the kernel size.
func NewLPPool1d(normType float64, kernel int, opts ...PoolOpt) *LPPool1d {
	p := &LPPool1d{}
	p.initLP(1, normType, kernel, opts)
	return p
}

// LPPool2d applies power-average pooling on (N, C, H, W) inputs.
type LPPool2d struct{ lpPoolNd }

// NewLPPool2d creates an LPPool2d with norm type normType (p > 0, finite);
// stride defaults to the kernel size.
func NewLPPool2d(normType float64, kernel int, opts ...PoolOpt) *LPPool2d {
	p := &LPPool2d{}
	p.initLP(2, normType, kernel, opts)
	return p
}

// LPPool3d applies power-average pooling on (N, C, D, H, W) inputs.
type LPPool3d struct{ lpPoolNd }

// NewLPPool3d creates an LPPool3d with norm type normType (p > 0, finite);
// stride defaults to the kernel size.
func NewLPPool3d(normType float64, kernel int, opts ...PoolOpt) *LPPool3d {
	p := &LPPool3d{}
	p.initLP(3, normType, kernel, opts)
	return p
}
