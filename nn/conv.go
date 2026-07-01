package nn

import (
	"fmt"
	"math"
	"math/rand"

	"gonn/tensor"
)

// All convolution layers (Conv1d/2d/3d, ConvTranspose1d/2d/3d) share one
// core, convNd: build (and cache) the per-channel gather matrix for the
// input's spatial size, unfold to an im2col matrix, and run a single GEMM
// against the flattened weight. See gather.go for the window machinery.

// ConvOpt configures a convolution layer.
type ConvOpt func(*convOpts)

type convOpts struct {
	kernel   []int
	stride   []int
	pad      []int
	dilation []int
	bias     bool
}

// WithKernel overrides the symmetric kernel size with per-dimension sizes
// (e.g. WithKernel(3, 2) for a 3x2 kernel). One value broadcasts to all dims.
func WithKernel(k ...int) ConvOpt { return func(o *convOpts) { o.kernel = k } }

// WithStride sets the stride: one value broadcasts to all spatial dims,
// or pass one per dim (e.g. WithStride(2, 1)). Default 1.
func WithStride(s ...int) ConvOpt { return func(o *convOpts) { o.stride = s } }

// WithPad sets the zero padding per side: one value broadcasts, or one per
// spatial dim. Default 0.
func WithPad(p ...int) ConvOpt { return func(o *convOpts) { o.pad = p } }

// WithDilation sets the kernel dilation: one value broadcasts, or one per
// spatial dim. Default 1.
func WithDilation(d ...int) ConvOpt { return func(o *convOpts) { o.dilation = d } }

// WithNoBias disables the bias term. Default is bias on.
func WithNoBias() ConvOpt { return func(o *convOpts) { o.bias = false } }

// broadcastDims expands a 1-element slice to rank entries and validates.
func broadcastDims(what string, v []int, rank, def int) []int {
	switch len(v) {
	case 0:
		out := make([]int, rank)
		for i := range out {
			out[i] = def
		}
		return out
	case 1:
		out := make([]int, rank)
		for i := range out {
			out[i] = v[0]
		}
		return out
	case rank:
		return append([]int(nil), v...)
	default:
		panic(fmt.Sprintf("nn: %s needs 1 or %d values, got %d", what, rank, len(v)))
	}
}

func resolveConvOpts(rank, kernel int, opts []ConvOpt) convOpts {
	o := convOpts{bias: true}
	for _, fn := range opts {
		fn(&o)
	}
	o.kernel = broadcastDims("kernel", o.kernel, rank, kernel)
	o.stride = broadcastDims("stride", o.stride, rank, 1)
	o.pad = broadcastDims("pad", o.pad, rank, 0)
	o.dilation = broadcastDims("dilation", o.dilation, rank, 1)
	for d := 0; d < rank; d++ {
		if o.kernel[d] < 1 || o.stride[d] < 1 || o.dilation[d] < 1 || o.pad[d] < 0 {
			panic(fmt.Sprintf("nn: invalid conv geometry kernel=%v stride=%v pad=%v dilation=%v",
				o.kernel, o.stride, o.pad, o.dilation))
		}
	}
	return o
}

// convNd is the shared conv/conv-transpose core.
type convNd struct {
	Base
	InC, OutC                     int
	Kernel, Stride, Pad, Dilation []int
	Transposed                    bool
	Weight                        *tensor.Tensor // conv: (OutC, InC, K...); transposed: (InC, OutC, K...)
	Bias                          *tensor.Tensor // (OutC,) or nil

	// Single-entry gather cache keyed by input spatial size: the matrix is a
	// constant leaf, so it is safe to reuse across forwards (training loops
	// have fixed shapes; a shape change rebuilds it).
	cacheKey                    []int
	cachedG                     *tensor.Tensor
	cachedOut                   []int
	cachedNumWin, cachedWinSize int
}

// initConv draws weights and bias exactly like the historical constructors:
// bound = sqrt(1/(inC*prod(kernel))), weights first (flat row-major), then
// bias — preserving the RNG draw order that seeded tests depend on.
func (c *convNd) initConv(rank, inC, outC, kernel int, transposed bool, opts []ConvOpt) {
	o := resolveConvOpts(rank, kernel, opts)
	c.InC, c.OutC = inC, outC
	c.Kernel, c.Stride, c.Pad, c.Dilation = o.kernel, o.stride, o.pad, o.dilation
	c.Transposed = transposed

	winSize := prodInts(o.kernel)
	fanIn := inC * winSize
	bound := math.Sqrt(1.0 / float64(fanIn))
	wData := make([]float64, outC*inC*winSize)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	var wShape []int
	if transposed {
		wShape = append([]int{inC, outC}, o.kernel...)
	} else {
		wShape = append([]int{outC, inC}, o.kernel...)
	}
	c.Weight = c.reg("weight", tensor.New(wData, wShape...).SetRequiresGrad(true))
	if o.bias {
		bData := make([]float64, outC)
		for i := range bData {
			bData[i] = -bound + rand.Float64()*(2*bound)
		}
		c.Bias = c.reg("bias", tensor.New(bData, outC).SetRequiresGrad(true))
	}
}

// gatherFor returns the (cached) gather matrix for the given spatial size.
func (c *convNd) gatherFor(spatial []int) (*tensor.Tensor, []int, int, int) {
	if c.cachedG != nil && intsEqual(c.cacheKey, spatial) {
		return c.cachedG, c.cachedOut, c.cachedNumWin, c.cachedWinSize
	}
	spec := slidingSpec{
		In: append([]int(nil), spatial...), Kernel: c.Kernel,
		Stride: c.Stride, Pad: c.Pad, Dilation: c.Dilation,
	}
	var g *tensor.Tensor
	var out []int
	var numWin, winSize int
	if c.Transposed {
		g, out, numWin, winSize = transposedGatherMatrix(spec)
	} else {
		g, out, numWin, winSize = gatherMatrix(spec)
	}
	c.cacheKey = spec.In
	c.cachedG, c.cachedOut = g, out
	c.cachedNumWin, c.cachedWinSize = numWin, winSize
	return g, out, numWin, winSize
}

// weightMat returns the weight flattened to (OutC, InC*winSize), matching
// unfold's (channel, kernel-offset) column order.
func (c *convNd) weightMat() *tensor.Tensor {
	winSize := prodInts(c.Kernel)
	if c.Transposed {
		// (InC, OutC, K...) -> (OutC, InC, K...) -> flatten.
		perm := make([]int, 2+len(c.Kernel))
		perm[0], perm[1] = 1, 0
		for d := range c.Kernel {
			perm[2+d] = 2 + d
		}
		return c.Weight.Permute(perm...).Reshape(c.OutC, c.InC*winSize)
	}
	return c.Weight.Reshape(c.OutC, c.InC*winSize)
}

// Forward runs the shared im2col + GEMM convolution.
func (c *convNd) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(c.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: conv expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	if x.Shape[1] != c.InC {
		panic(fmt.Sprintf("nn: conv input channels %d != InC %d", x.Shape[1], c.InC))
	}
	N := x.Shape[0]
	g, out, numWin, winSize := c.gatherFor(x.Shape[2:])

	col := unfold(x, g, numWin, winSize)       // (N*numWin, C*winSize)
	y := col.MatMul(c.weightMat().Transpose()) // (N*numWin, OutC)
	if c.Bias != nil {
		y = y.Add(c.Bias)
	}
	// (N, out..., OutC) -> (N, OutC, out...)
	shape := append(append([]int{N}, out...), c.OutC)
	perm := make([]int, rank+2)
	perm[0], perm[1] = 0, rank+1
	for d := 0; d < rank; d++ {
		perm[2+d] = 1 + d
	}
	return y.Reshape(shape...).Permute(perm...)
}

func intsEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---- Public conv layers -----------------------------------------------------

// Conv1d performs 1D convolution on inputs of shape (N, C, L).
type Conv1d struct{ convNd }

// NewConv1d creates a Conv1d. Defaults: stride 1, no padding, dilation 1,
// bias on; configure with WithStride/WithPad/WithDilation/WithNoBias.
func NewConv1d(inC, outC, kernel int, opts ...ConvOpt) *Conv1d {
	c := &Conv1d{}
	c.initConv(1, inC, outC, kernel, false, opts)
	return c
}

// Conv2d performs 2D convolution on inputs of shape (N, C, H, W).
type Conv2d struct{ convNd }

// NewConv2d creates a Conv2d. Asymmetric geometry via multi-arg options,
// e.g. NewConv2d(3, 16, 3, WithStride(2, 1), WithPad(1)).
func NewConv2d(inC, outC, kernel int, opts ...ConvOpt) *Conv2d {
	c := &Conv2d{}
	c.initConv(2, inC, outC, kernel, false, opts)
	return c
}

// Conv3d performs 3D convolution on inputs of shape (N, C, D, H, W).
type Conv3d struct{ convNd }

// NewConv3d creates a Conv3d.
func NewConv3d(inC, outC, kernel int, opts ...ConvOpt) *Conv3d {
	c := &Conv3d{}
	c.initConv(3, inC, outC, kernel, false, opts)
	return c
}

// ConvTranspose1d performs 1D transposed convolution ("deconvolution") on
// inputs of shape (N, InC, L). Weight shape is (InC, OutC, K).
type ConvTranspose1d struct{ convNd }

// NewConvTranspose1d creates a ConvTranspose1d.
// out_len = (L-1)*stride - 2*pad + dilation*(K-1) + 1.
func NewConvTranspose1d(inC, outC, kernel int, opts ...ConvOpt) *ConvTranspose1d {
	c := &ConvTranspose1d{}
	c.initConv(1, inC, outC, kernel, true, opts)
	return c
}

// ConvTranspose2d performs 2D transposed convolution on (N, InC, H, W).
// Weight shape is (InC, OutC, KH, KW).
type ConvTranspose2d struct{ convNd }

// NewConvTranspose2d creates a ConvTranspose2d.
func NewConvTranspose2d(inC, outC, kernel int, opts ...ConvOpt) *ConvTranspose2d {
	c := &ConvTranspose2d{}
	c.initConv(2, inC, outC, kernel, true, opts)
	return c
}

// ConvTranspose3d performs 3D transposed convolution on (N, InC, D, H, W).
// Weight shape is (InC, OutC, KD, KH, KW).
type ConvTranspose3d struct{ convNd }

// NewConvTranspose3d creates a ConvTranspose3d.
func NewConvTranspose3d(inC, outC, kernel int, opts ...ConvOpt) *ConvTranspose3d {
	c := &ConvTranspose3d{}
	c.initConv(3, inC, outC, kernel, true, opts)
	return c
}
