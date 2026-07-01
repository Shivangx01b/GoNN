package nn

import (
	"fmt"
	"math/rand"

	"gonn/tensor"
)

// Fractional max pooling (Ben Graham, "Fractional Max-Pooling",
// arXiv:1412.6071), PyTorch torch.nn.FractionalMaxPool{2,3}d semantics: the
// output size is given directly (WithOutputSize) or as a ratio of the input
// size (WithOutputRatio, out = floor(in*ratio)), and the kernel-sized pooling
// windows are placed at pseudorandom positions that vary per forward pass.
//
// Window starts per dim follow PyTorch's exact construction
// (aten/src/ATen/native/FractionalMaxPooling.h, generate_intervals): with
// u ~ U(0,1) and alpha = (in - kernel) / (out - 1),
//
//	start_i = floor((i + u) * alpha) - floor(u * alpha)   for i < out-1
//	start_{out-1} = in - kernel
//
// which yields a non-decreasing sequence of kernel-sized windows covering the
// input, subject to PyTorch's feasibility constraint kernel + out - 1 <= in.
//
// Documented deviation: PyTorch draws one u per (n, c, dim) via its
// _random_samples tensor; here a single u per dim is drawn per forward and
// shared across the batch and channels (equivalent to a broadcast
// _random_samples), which keeps the window gather shared across (n, c).
// WithFractionalSamples injects fixed u values for deterministic tests.
//
// Max over each region reuses the same gather + MaxAxis machinery as the
// other pools, so gradients flow to the selected maxima by construction.

// FracPoolOpt configures a fractional max-pooling layer.
type FracPoolOpt func(*fracPoolOpts)

type fracPoolOpts struct {
	kernel  []int
	size    []int
	ratio   []float64
	samples []float64
}

// WithFracKernel overrides the symmetric kernel with per-dimension sizes.
func WithFracKernel(k ...int) FracPoolOpt { return func(o *fracPoolOpts) { o.kernel = k } }

// WithOutputSize fixes the output spatial size (one value broadcasts, or one
// per dim). Exactly one of WithOutputSize / WithOutputRatio must be given.
func WithOutputSize(s ...int) FracPoolOpt { return func(o *fracPoolOpts) { o.size = s } }

// WithOutputRatio sets the output size as a fraction of the input size,
// out = floor(in*ratio), with each ratio in (0, 1). One value broadcasts.
func WithOutputRatio(r ...float64) FracPoolOpt { return func(o *fracPoolOpts) { o.ratio = r } }

// WithFractionalSamples injects the per-dimension random samples u in [0, 1)
// used to place the pooling windows (one value broadcasts, or one per dim),
// making the layer deterministic — the fractional analogue of PyTorch's
// _random_samples argument.
func WithFractionalSamples(u ...float64) FracPoolOpt {
	return func(o *fracPoolOpts) { o.samples = u }
}

// fracPoolNd is the shared fractional max-pooling core.
type fracPoolNd struct {
	Base
	Kernel   []int
	OutSize  []int     // nil when OutRatio drives the output size
	OutRatio []float64 // nil when OutSize is fixed
	samples  []float64 // injected u per dim; nil means draw fresh per forward
}

func (p *fracPoolNd) initFrac(rank, kernel int, opts []FracPoolOpt) {
	o := fracPoolOpts{}
	for _, fn := range opts {
		fn(&o)
	}
	p.Kernel = broadcastDims("kernel", o.kernel, rank, kernel)
	for _, k := range p.Kernel {
		if k < 1 {
			panic(fmt.Sprintf("nn: invalid fractional pool kernel %v", p.Kernel))
		}
	}
	if (len(o.size) == 0) == (len(o.ratio) == 0) {
		panic("nn: fractional max pool needs exactly one of WithOutputSize or WithOutputRatio")
	}
	if len(o.size) > 0 {
		p.OutSize = broadcastDims("output size", o.size, rank, 0)
		for _, s := range p.OutSize {
			if s < 1 {
				panic(fmt.Sprintf("nn: invalid fractional pool output size %v", p.OutSize))
			}
		}
	} else {
		p.OutRatio = broadcastFloats("output ratio", o.ratio, rank)
		for _, r := range p.OutRatio {
			if !(r > 0 && r < 1) {
				panic(fmt.Sprintf("nn: fractional pool output ratio must be in (0,1), got %v", p.OutRatio))
			}
		}
	}
	if len(o.samples) > 0 {
		p.samples = broadcastFloats("fractional samples", o.samples, rank)
		for _, u := range p.samples {
			if !(u >= 0 && u < 1) {
				panic(fmt.Sprintf("nn: fractional samples must be in [0,1), got %v", p.samples))
			}
		}
	}
}

// broadcastFloats mirrors broadcastDims for float options: one value
// broadcasts to every dim, or one value per dim.
func broadcastFloats(what string, v []float64, rank int) []float64 {
	switch len(v) {
	case 1:
		out := make([]float64, rank)
		for i := range out {
			out[i] = v[0]
		}
		return out
	case rank:
		return append([]float64(nil), v...)
	}
	panic(fmt.Sprintf("nn: %s needs 1 or %d values, got %d", what, rank, len(v)))
}

// outSizesFor resolves the output spatial size for the given input size and
// checks PyTorch's feasibility constraint kernel + out - 1 <= in.
func (p *fracPoolNd) outSizesFor(in []int) []int {
	rank := len(p.Kernel)
	out := make([]int, rank)
	if p.OutSize != nil {
		copy(out, p.OutSize)
	} else {
		for d := range out {
			out[d] = int(float64(in[d]) * p.OutRatio[d])
		}
	}
	for d := range out {
		if out[d] < 1 {
			panic(fmt.Sprintf("nn: fractional pool output size %v must be positive (input %v)", out, in))
		}
		if p.Kernel[d]+out[d]-1 > in[d] {
			panic(fmt.Sprintf("nn: fractional pool needs kernel+out-1 <= in per dim, got kernel=%v out=%v in=%v",
				p.Kernel, out, in))
		}
	}
	return out
}

// fractionalIntervals generates the window start positions for one dimension
// using PyTorch's generate_intervals formula (see file comment).
func fractionalIntervals(u float64, in, out, kernel int) []int {
	seq := make([]int, out)
	if out > 1 {
		alpha := float64(in-kernel) / float64(out-1)
		for i := 0; i < out-1; i++ {
			seq[i] = int((float64(i)+u)*alpha) - int(u*alpha)
		}
	}
	seq[out-1] = in - kernel
	return seq
}

// Forward places pseudorandom kernel-sized windows and takes the max of each.
func (p *fracPoolNd) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(p.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: fractional pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	in := x.Shape[2:]
	out := p.outSizesFor(in)

	starts := make([][]int, rank)
	for d := 0; d < rank; d++ {
		u := rand.Float64()
		if p.samples != nil {
			u = p.samples[d]
		}
		starts[d] = fractionalIntervals(u, in[d], out[d], p.Kernel[d])
	}

	// Gather matrix: row (w*winSize + kk) reads input position
	// starts[d][winIdx[d]] + kIdx[d] per dim (always in range by the
	// feasibility constraint).
	numWin := prodInts(out)
	winSize := prodInts(p.Kernel)
	cols := prodInts(in)
	inStrides := rowMajorStrides(in)
	gData := make([]float64, numWin*winSize*cols)
	winIdx := make([]int, rank)
	for w := 0; w < numWin; w++ {
		kIdx := make([]int, rank)
		for kk := 0; kk < winSize; kk++ {
			col := 0
			for d := 0; d < rank; d++ {
				col += (starts[d][winIdx[d]] + kIdx[d]) * inStrides[d]
			}
			gData[(w*winSize+kk)*cols+col] = 1
			incMultiIndex(kIdx, p.Kernel)
		}
		incMultiIndex(winIdx, out)
	}
	g := tensor.New(gData, numWin*winSize, cols)

	win := x.Reshape(N*C, cols).
		MatMul(g.Transpose()).
		Reshape(N*C, numWin, winSize)
	return win.MaxAxis(2, false).Reshape(append([]int{N, C}, out...)...)
}

// FractionalMaxPool2d applies fractional max pooling on (N, C, H, W) inputs.
type FractionalMaxPool2d struct{ fracPoolNd }

// NewFractionalMaxPool2d creates a FractionalMaxPool2d with a symmetric
// kernel (override per-dim with WithFracKernel); exactly one of
// WithOutputSize / WithOutputRatio is required.
func NewFractionalMaxPool2d(kernel int, opts ...FracPoolOpt) *FractionalMaxPool2d {
	p := &FractionalMaxPool2d{}
	p.initFrac(2, kernel, opts)
	return p
}

// FractionalMaxPool3d applies fractional max pooling on (N, C, D, H, W) inputs.
type FractionalMaxPool3d struct{ fracPoolNd }

// NewFractionalMaxPool3d creates a FractionalMaxPool3d with a symmetric
// kernel (override per-dim with WithFracKernel); exactly one of
// WithOutputSize / WithOutputRatio is required.
func NewFractionalMaxPool3d(kernel int, opts ...FracPoolOpt) *FractionalMaxPool3d {
	p := &FractionalMaxPool3d{}
	p.initFrac(3, kernel, opts)
	return p
}
