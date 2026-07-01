package nn

import (
	"fmt"

	"gonn/tensor"
)

// All fixed-window pooling layers share one core, poolNd, built on the same
// gatherMatrix as the conv layers (pad 0, dilation 1): gather every window,
// then MaxAxis/MeanAxis over the window axis.

type poolMode int

const (
	maxPool poolMode = iota
	avgPool
)

// PoolOpt configures a pooling layer.
type PoolOpt func(*poolOpts)

type poolOpts struct {
	kernel []int
	stride []int
}

// WithPoolKernel overrides the symmetric kernel with per-dimension sizes.
func WithPoolKernel(k ...int) PoolOpt { return func(o *poolOpts) { o.kernel = k } }

// WithPoolStride sets the stride (one value broadcasts, or one per dim).
// Default: the kernel size (non-overlapping windows).
func WithPoolStride(s ...int) PoolOpt { return func(o *poolOpts) { o.stride = s } }

// poolNd is the shared pooling core.
type poolNd struct {
	Base
	Kernel, Stride []int
	mode           poolMode

	cacheKey                    []int
	cachedG                     *tensor.Tensor
	cachedOut                   []int
	cachedNumWin, cachedWinSize int
}

func (p *poolNd) initPool(rank, kernel int, mode poolMode, opts []PoolOpt) {
	o := poolOpts{}
	for _, fn := range opts {
		fn(&o)
	}
	p.Kernel = broadcastDims("kernel", o.kernel, rank, kernel)
	if len(o.stride) == 0 {
		p.Stride = append([]int(nil), p.Kernel...) // default stride = kernel
	} else {
		p.Stride = broadcastDims("stride", o.stride, rank, kernel)
	}
	for d := 0; d < rank; d++ {
		if p.Kernel[d] < 1 || p.Stride[d] < 1 {
			panic(fmt.Sprintf("nn: invalid pool geometry kernel=%v stride=%v", p.Kernel, p.Stride))
		}
	}
	p.mode = mode
}

func (p *poolNd) gatherFor(spatial []int) (*tensor.Tensor, []int, int, int) {
	if p.cachedG != nil && intsEqual(p.cacheKey, spatial) {
		return p.cachedG, p.cachedOut, p.cachedNumWin, p.cachedWinSize
	}
	ones := make([]int, len(spatial))
	zeros := make([]int, len(spatial))
	for d := range ones {
		ones[d] = 1
	}
	spec := slidingSpec{
		In: append([]int(nil), spatial...), Kernel: p.Kernel,
		Stride: p.Stride, Pad: zeros, Dilation: ones,
	}
	g, out, numWin, winSize := gatherMatrix(spec)
	p.cacheKey = spec.In
	p.cachedG, p.cachedOut = g, out
	p.cachedNumWin, p.cachedWinSize = numWin, winSize
	return g, out, numWin, winSize
}

// Forward gathers windows and reduces each with max or mean.
func (p *poolNd) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(p.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	g, out, numWin, winSize := p.gatherFor(x.Shape[2:])

	win := x.Reshape(N*C, prodInts(x.Shape[2:])).
		MatMul(g.Transpose()).
		Reshape(N*C, numWin, winSize)
	var r *tensor.Tensor
	if p.mode == maxPool {
		r = win.MaxAxis(2, false)
	} else {
		r = win.MeanAxis(2, false)
	}
	return r.Reshape(append([]int{N, C}, out...)...)
}

// forwardWithIndices is Forward plus, for max pooling, an indices tensor of
// the same shape as the output holding the FLAT SPATIAL index of each selected
// maximum within its (n, c) plane — PyTorch's return_indices convention (e.g.
// for a (N, C, H, W) input the index is h*W + w). The pooled output is
// computed by the exact same ops as Forward, so numerics are identical; ties
// resolve to the first (lowest-index) maximum in both paths. Indices carry no
// gradient.
func (p *poolNd) forwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	if p.mode != maxPool {
		panic("nn: forwardWithIndices is only defined for max pooling")
	}
	rank := len(p.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	g, out, numWin, winSize := p.gatherFor(x.Shape[2:])

	win := x.Reshape(N*C, prodInts(x.Shape[2:])).
		MatMul(g.Transpose()).
		Reshape(N*C, numWin, winSize)
	r := win.MaxAxis(2, false)

	// Window-relative argmax, then translate each kernel offset back to the
	// input spatial position it reads: pos[d] = winIdx[d]*stride[d] + kIdx[d]
	// (pooling has no padding or dilation, so every offset is in range).
	arg := win.ArgMax(2) // (N*C, numWin)
	inStrides := rowMajorStrides(x.Shape[2:])
	lut := make([]int, numWin*winSize)
	winIdx := make([]int, rank)
	for w := 0; w < numWin; w++ {
		kIdx := make([]int, rank)
		for kk := 0; kk < winSize; kk++ {
			flat := 0
			for d := 0; d < rank; d++ {
				flat += (winIdx[d]*p.Stride[d] + kIdx[d]) * inStrides[d]
			}
			lut[w*winSize+kk] = flat
			incMultiIndex(kIdx, p.Kernel)
		}
		incMultiIndex(winIdx, out)
	}
	idx := tensor.Zeros(append([]int{N, C}, out...)...)
	for i := 0; i < N*C; i++ {
		for w := 0; w < numWin; w++ {
			idx.Data[i*numWin+w] = float64(lut[w*winSize+int(arg.Data[i*numWin+w])])
		}
	}
	return r.Reshape(append([]int{N, C}, out...)...), idx
}

// ---- Public pooling layers --------------------------------------------------

// MaxPool1d performs 1D max pooling on (N, C, L) inputs.
type MaxPool1d struct{ poolNd }

// NewMaxPool1d creates a MaxPool1d; stride defaults to the kernel size.
func NewMaxPool1d(kernel int, opts ...PoolOpt) *MaxPool1d {
	p := &MaxPool1d{}
	p.initPool(1, kernel, maxPool, opts)
	return p
}

// ForwardWithIndices returns the pooled output plus the flat spatial index
// (within each (n, c) plane) of every selected maximum, for use with
// MaxUnpool1d. The output matches Forward exactly.
func (p *MaxPool1d) ForwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	return p.forwardWithIndices(x)
}

// AvgPool1d performs 1D average pooling on (N, C, L) inputs.
type AvgPool1d struct{ poolNd }

// NewAvgPool1d creates an AvgPool1d; stride defaults to the kernel size.
func NewAvgPool1d(kernel int, opts ...PoolOpt) *AvgPool1d {
	p := &AvgPool1d{}
	p.initPool(1, kernel, avgPool, opts)
	return p
}

// MaxPool2d performs 2D max pooling on (N, C, H, W) inputs.
type MaxPool2d struct{ poolNd }

// NewMaxPool2d creates a MaxPool2d; stride defaults to the kernel size.
func NewMaxPool2d(kernel int, opts ...PoolOpt) *MaxPool2d {
	p := &MaxPool2d{}
	p.initPool(2, kernel, maxPool, opts)
	return p
}

// ForwardWithIndices returns the pooled output plus the flat spatial index
// (h*W + w within each (n, c) plane) of every selected maximum, for use with
// MaxUnpool2d. The output matches Forward exactly.
func (p *MaxPool2d) ForwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	return p.forwardWithIndices(x)
}

// AvgPool2d performs 2D average pooling on (N, C, H, W) inputs.
type AvgPool2d struct{ poolNd }

// NewAvgPool2d creates an AvgPool2d; stride defaults to the kernel size.
func NewAvgPool2d(kernel int, opts ...PoolOpt) *AvgPool2d {
	p := &AvgPool2d{}
	p.initPool(2, kernel, avgPool, opts)
	return p
}

// MaxPool3d performs 3D max pooling on (N, C, D, H, W) inputs.
type MaxPool3d struct{ poolNd }

// NewMaxPool3d creates a MaxPool3d; stride defaults to the kernel size.
func NewMaxPool3d(kernel int, opts ...PoolOpt) *MaxPool3d {
	p := &MaxPool3d{}
	p.initPool(3, kernel, maxPool, opts)
	return p
}

// ForwardWithIndices returns the pooled output plus the flat spatial index
// ((d*H + h)*W + w within each (n, c) plane) of every selected maximum, for
// use with MaxUnpool3d. The output matches Forward exactly.
func (p *MaxPool3d) ForwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	return p.forwardWithIndices(x)
}

// AvgPool3d performs 3D average pooling on (N, C, D, H, W) inputs.
type AvgPool3d struct{ poolNd }

// NewAvgPool3d creates an AvgPool3d; stride defaults to the kernel size.
func NewAvgPool3d(kernel int, opts ...PoolOpt) *AvgPool3d {
	p := &AvgPool3d{}
	p.initPool(3, kernel, avgPool, opts)
	return p
}
