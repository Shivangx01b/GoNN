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
	outPad   []int
	groups   int
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

// WithGroups splits the convolution into g groups (PyTorch groups semantics):
// input channels and output channels are each divided into g equal groups,
// and group i's outputs see only group i's inputs. Requires inC%g == 0 and
// outC%g == 0. Weight shape becomes (OutC, InC/g, K...) for convolutions and
// (InC, OutC/g, K...) for transposed convolutions, and the init bound uses
// fanIn = (inC/g)*prod(kernel). groups > 1 runs g smaller unfold+GEMMs (one
// per group) and concatenates along the channel axis. Default 1.
func WithGroups(g int) ConvOpt { return func(o *convOpts) { o.groups = g } }

// WithOutputPadding adds extra zero rows/columns at the high edge of each
// spatial dim of a transposed convolution's output (PyTorch output_padding):
// out = (in-1)*stride - 2*pad + dilation*(kernel-1) + output_padding + 1.
// One value broadcasts to all spatial dims, or pass one per dim. Each value
// must satisfy 0 <= output_padding < stride. Only valid on ConvTransposeNd.
// Default 0.
func WithOutputPadding(p ...int) ConvOpt { return func(o *convOpts) { o.outPad = p } }

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
	o := convOpts{bias: true, groups: 1}
	for _, fn := range opts {
		fn(&o)
	}
	o.kernel = broadcastDims("kernel", o.kernel, rank, kernel)
	o.stride = broadcastDims("stride", o.stride, rank, 1)
	o.pad = broadcastDims("pad", o.pad, rank, 0)
	o.dilation = broadcastDims("dilation", o.dilation, rank, 1)
	o.outPad = broadcastDims("output_padding", o.outPad, rank, 0)
	for d := 0; d < rank; d++ {
		if o.kernel[d] < 1 || o.stride[d] < 1 || o.dilation[d] < 1 || o.pad[d] < 0 || o.outPad[d] < 0 {
			panic(fmt.Sprintf("nn: invalid conv geometry kernel=%v stride=%v pad=%v dilation=%v output_padding=%v",
				o.kernel, o.stride, o.pad, o.dilation, o.outPad))
		}
	}
	if o.groups < 1 {
		panic(fmt.Sprintf("nn: groups must be >= 1, got %d", o.groups))
	}
	return o
}

// convNd is the shared conv/conv-transpose core.
type convNd struct {
	Base
	InC, OutC                     int
	Groups                        int
	Kernel, Stride, Pad, Dilation []int
	OutputPadding                 []int // transposed convs only; all-zero otherwise
	Transposed                    bool
	Weight                        *tensor.Tensor // conv: (OutC, InC/Groups, K...); transposed: (InC, OutC/Groups, K...)
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
// bound = sqrt(1/((inC/groups)*prod(kernel))), weights first (flat row-major),
// then bias — preserving the RNG draw order that seeded tests depend on. With
// groups == 1 the draw count and sequence are bit-identical to the historical
// (ungrouped) constructors; groups > 1 shrinks the weight to InC/groups
// (or OutC/groups for transposed) per-group input channels, which changes
// both the fan-in and the number of draws, matching PyTorch.
func (c *convNd) initConv(rank, inC, outC, kernel int, transposed bool, opts []ConvOpt) {
	o := resolveConvOpts(rank, kernel, opts)
	if inC%o.groups != 0 || outC%o.groups != 0 {
		panic(fmt.Sprintf("nn: groups=%d must divide both inC=%d and outC=%d", o.groups, inC, outC))
	}
	if transposed {
		for d := 0; d < rank; d++ {
			if o.outPad[d] >= o.stride[d] {
				panic(fmt.Sprintf("nn: output_padding %v must be smaller than stride %v per dim", o.outPad, o.stride))
			}
		}
	} else {
		for d := 0; d < rank; d++ {
			if o.outPad[d] != 0 {
				panic("nn: output_padding is only valid for transposed convolutions")
			}
		}
	}
	c.InC, c.OutC, c.Groups = inC, outC, o.groups
	c.Kernel, c.Stride, c.Pad, c.Dilation = o.kernel, o.stride, o.pad, o.dilation
	c.OutputPadding = o.outPad
	c.Transposed = transposed

	winSize := prodInts(o.kernel)
	fanIn := (inC / o.groups) * winSize
	bound := math.Sqrt(1.0 / float64(fanIn))
	var wShape []int
	if transposed {
		wShape = append([]int{inC, outC / o.groups}, o.kernel...)
	} else {
		wShape = append([]int{outC, inC / o.groups}, o.kernel...)
	}
	wData := make([]float64, prodInts(wShape))
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
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
		OutPad: c.OutputPadding,
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

// weightMat returns the full (groups == 1) weight flattened to
// (OutC, InC*winSize), matching unfold's (channel, kernel-offset) column
// order.
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

// groupWeightMat returns group i's weight slice flattened to
// (OutC/g, (InC/g)*winSize), matching unfold's column order for that group's
// input-channel slice. Slicing goes through IndexSelect, so weight gradients
// flow back to the right rows.
func (c *convNd) groupWeightMat(i int) *tensor.Tensor {
	winSize := prodInts(c.Kernel)
	inCg, outCg := c.InC/c.Groups, c.OutC/c.Groups
	if c.Transposed {
		// (InC, OutC/g, K...): group i owns input-channel rows
		// [i*inCg, (i+1)*inCg) -> (inCg, outCg, K...) -> (outCg, inCg, K...).
		perm := make([]int, 2+len(c.Kernel))
		perm[0], perm[1] = 1, 0
		for d := range c.Kernel {
			perm[2+d] = 2 + d
		}
		return c.Weight.IndexSelect(0, rangeIndex(i*inCg, inCg)).
			Permute(perm...).Reshape(outCg, inCg*winSize)
	}
	// (OutC, InC/g, K...): group i owns output-channel rows
	// [i*outCg, (i+1)*outCg).
	return c.Weight.IndexSelect(0, rangeIndex(i*outCg, outCg)).
		Reshape(outCg, inCg*winSize)
}

// rangeIndex returns a 1-D index tensor [start, start+1, ..., start+n-1].
func rangeIndex(start, n int) *tensor.Tensor {
	d := make([]float64, n)
	for i := range d {
		d[i] = float64(start + i)
	}
	return tensor.New(d, n)
}

// Forward runs the shared im2col + GEMM convolution. With Groups == 1 it is
// a single unfold + GEMM; with Groups > 1 it runs g smaller GEMMs (one per
// channel group, each on its slice of the input and weight) and concatenates
// the per-group outputs along the channel axis.
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

	var y *tensor.Tensor
	if c.Groups == 1 {
		col := unfold(x, g, numWin, winSize)      // (N*numWin, C*winSize)
		y = col.MatMul(c.weightMat().Transpose()) // (N*numWin, OutC)
	} else {
		inCg := c.InC / c.Groups
		parts := make([]*tensor.Tensor, c.Groups)
		for i := 0; i < c.Groups; i++ {
			xi := x.IndexSelect(1, rangeIndex(i*inCg, inCg)) // (N, InC/g, spatial...)
			col := unfold(xi, g, numWin, winSize)            // (N*numWin, (InC/g)*winSize)
			parts[i] = col.MatMul(c.groupWeightMat(i).Transpose())
		}
		y = tensor.Concat(1, parts...) // (N*numWin, OutC)
	}
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
// groups 1, bias on; configure with WithStride/WithPad/WithDilation/
// WithGroups/WithNoBias.
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
// inputs of shape (N, InC, L). Weight shape is (InC, OutC/groups, K).
type ConvTranspose1d struct{ convNd }

// NewConvTranspose1d creates a ConvTranspose1d.
// out_len = (L-1)*stride - 2*pad + dilation*(K-1) + output_padding + 1.
// Configure with WithStride/WithPad/WithDilation/WithOutputPadding/
// WithGroups/WithNoBias.
func NewConvTranspose1d(inC, outC, kernel int, opts ...ConvOpt) *ConvTranspose1d {
	c := &ConvTranspose1d{}
	c.initConv(1, inC, outC, kernel, true, opts)
	return c
}

// ConvTranspose2d performs 2D transposed convolution on (N, InC, H, W).
// Weight shape is (InC, OutC/groups, KH, KW).
type ConvTranspose2d struct{ convNd }

// NewConvTranspose2d creates a ConvTranspose2d.
func NewConvTranspose2d(inC, outC, kernel int, opts ...ConvOpt) *ConvTranspose2d {
	c := &ConvTranspose2d{}
	c.initConv(2, inC, outC, kernel, true, opts)
	return c
}

// ConvTranspose3d performs 3D transposed convolution on (N, InC, D, H, W).
// Weight shape is (InC, OutC/groups, KD, KH, KW).
type ConvTranspose3d struct{ convNd }

// NewConvTranspose3d creates a ConvTranspose3d.
func NewConvTranspose3d(inC, outC, kernel int, opts ...ConvOpt) *ConvTranspose3d {
	c := &ConvTranspose3d{}
	c.initConv(3, inC, outC, kernel, true, opts)
	return c
}
