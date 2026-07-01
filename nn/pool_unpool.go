package nn

import (
	"fmt"

	"gonn/tensor"
)

// MaxUnpool, PyTorch torch.nn.MaxUnpool{1,2,3}d semantics: the partial
// inverse of max pooling. Forward takes the pooled values plus the indices
// tensor produced by MaxPool*d.ForwardWithIndices (flat spatial index within
// each (n, c) plane) and scatters each value into a zero tensor of the
// unpooled size at its recorded position; every non-maximal cell stays zero.
//
// The default output spatial size is the inverse of the pooling formula,
// out[d] = (in[d]-1)*stride[d] + kernel[d]; pass explicit sizes to Forward to
// resolve the ambiguity when the pooled size does not determine the input
// size uniquely (e.g. stride < kernel or an input that was not a multiple of
// the stride). Padding is not supported (the pooling layers here are pad-0).
//
// Differentiable w.r.t. the values: backward gathers the output gradient at
// the same indices. Duplicate indices within one (n, c) plane are invalid
// input (PyTorch leaves that case nondeterministic); here the last write wins
// in the forward and both sources receive the gradient of the shared cell.

// maxUnpoolNd is the shared unpooling core.
type maxUnpoolNd struct {
	Base
	Kernel, Stride []int
}

func (u *maxUnpoolNd) initUnpool(rank, kernel int, opts []PoolOpt) {
	o := poolOpts{}
	for _, fn := range opts {
		fn(&o)
	}
	u.Kernel = broadcastDims("kernel", o.kernel, rank, kernel)
	if len(o.stride) == 0 {
		u.Stride = append([]int(nil), u.Kernel...) // default stride = kernel
	} else {
		u.Stride = broadcastDims("stride", o.stride, rank, kernel)
	}
	for d := 0; d < rank; d++ {
		if u.Kernel[d] < 1 || u.Stride[d] < 1 {
			panic(fmt.Sprintf("nn: invalid unpool geometry kernel=%v stride=%v", u.Kernel, u.Stride))
		}
	}
}

// forward scatters x's values to the positions in indices. outputSpatial, if
// non-empty, overrides the default (in-1)*stride + kernel per-dim size.
func (u *maxUnpoolNd) forward(x, indices *tensor.Tensor, outputSpatial []int) *tensor.Tensor {
	rank := len(u.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: max unpool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	if !intsEqual(x.Shape, indices.Shape) {
		panic(fmt.Sprintf("nn: max unpool indices shape %v != input shape %v", indices.Shape, x.Shape))
	}
	in := x.Shape[2:]
	outSp := make([]int, rank)
	switch len(outputSpatial) {
	case 0:
		for d := 0; d < rank; d++ {
			outSp[d] = (in[d]-1)*u.Stride[d] + u.Kernel[d]
		}
	case rank:
		copy(outSp, outputSpatial)
		for _, s := range outSp {
			if s < 1 {
				panic(fmt.Sprintf("nn: max unpool output size %v must be positive", outputSpatial))
			}
		}
	default:
		panic(fmt.Sprintf("nn: max unpool expected %d output sizes, got %d", rank, len(outputSpatial)))
	}

	N, C := x.Shape[0], x.Shape[1]
	cells := prodInts(in) // pooled cells per (n, c) plane
	S := prodInts(outSp)  // unpooled cells per (n, c) plane
	out := tensor.Zeros(append([]int{N, C}, outSp...)...)
	dst := make([]int, N*C*cells) // flat out position per input element (for backward)
	for i := 0; i < N*C; i++ {
		for j := 0; j < cells; j++ {
			flat := int(indices.Data[i*cells+j])
			if flat < 0 || flat >= S {
				panic(fmt.Sprintf("nn: max unpool index %d out of range [0,%d) for output %v", flat, S, outSp))
			}
			out.Data[i*S+flat] = x.Data[i*cells+j]
			dst[i*cells+j] = i*S + flat
		}
	}

	xShape := append([]int(nil), x.Shape...)
	tensor.MakeNode(out, "MaxUnpool", []*tensor.Tensor{x}, func(grad *tensor.Tensor) []*tensor.Tensor {
		g := tensor.Zeros(xShape...)
		for j, d := range dst {
			g.Data[j] = grad.Data[d]
		}
		return []*tensor.Tensor{g}
	})
	return out
}

// MaxUnpool1d inverts MaxPool1d on (N, C, L) pooled inputs.
type MaxUnpool1d struct{ maxUnpoolNd }

// NewMaxUnpool1d creates a MaxUnpool1d; stride defaults to the kernel size
// (use WithPoolStride/WithPoolKernel to mirror the pooling layer's geometry).
func NewMaxUnpool1d(kernel int, opts ...PoolOpt) *MaxUnpool1d {
	u := &MaxUnpool1d{}
	u.initUnpool(1, kernel, opts)
	return u
}

// Forward scatters x at indices (from MaxPool1d.ForwardWithIndices) into a
// zero tensor of length (L-1)*stride + kernel, or outputSpatial if given.
func (u *MaxUnpool1d) Forward(x, indices *tensor.Tensor, outputSpatial ...int) *tensor.Tensor {
	return u.forward(x, indices, outputSpatial)
}

// MaxUnpool2d inverts MaxPool2d on (N, C, H, W) pooled inputs.
type MaxUnpool2d struct{ maxUnpoolNd }

// NewMaxUnpool2d creates a MaxUnpool2d; stride defaults to the kernel size.
func NewMaxUnpool2d(kernel int, opts ...PoolOpt) *MaxUnpool2d {
	u := &MaxUnpool2d{}
	u.initUnpool(2, kernel, opts)
	return u
}

// Forward scatters x at indices (from MaxPool2d.ForwardWithIndices) into a
// zero tensor of the default (in-1)*stride + kernel size per dim, or
// outputSpatial (H, W) if given.
func (u *MaxUnpool2d) Forward(x, indices *tensor.Tensor, outputSpatial ...int) *tensor.Tensor {
	return u.forward(x, indices, outputSpatial)
}

// MaxUnpool3d inverts MaxPool3d on (N, C, D, H, W) pooled inputs.
type MaxUnpool3d struct{ maxUnpoolNd }

// NewMaxUnpool3d creates a MaxUnpool3d; stride defaults to the kernel size.
func NewMaxUnpool3d(kernel int, opts ...PoolOpt) *MaxUnpool3d {
	u := &MaxUnpool3d{}
	u.initUnpool(3, kernel, opts)
	return u
}

// Forward scatters x at indices (from MaxPool3d.ForwardWithIndices) into a
// zero tensor of the default (in-1)*stride + kernel size per dim, or
// outputSpatial (D, H, W) if given.
func (u *MaxUnpool3d) Forward(x, indices *tensor.Tensor, outputSpatial ...int) *tensor.Tensor {
	return u.forward(x, indices, outputSpatial)
}
