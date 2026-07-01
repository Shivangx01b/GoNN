package nn

import (
	"fmt"

	"gonn/tensor"
)

// Unfold and Fold (PyTorch torch.nn.Unfold / torch.nn.Fold, 2D) expose the
// im2col machinery behind the conv layers as standalone modules. Both are
// built purely from the shared gatherMatrix and differentiable tensor ops,
// so autograd works by construction:
//
//   - Unfold applies G^T (via the shared unfold helper): every sliding block
//     is extracted into a column.
//   - Fold applies G itself: because G maps (block, kernel-offset) rows to
//     input positions, multiplying by G scatters the blocks back and sums
//     overlapping contributions automatically — exactly PyTorch's Fold.
//
// PyTorch invariant: Fold(Unfold(x)) == x * divisor, where divisor =
// Fold(Unfold(ones_like(x))) counts how many blocks cover each position.

// resolveFoldOpts resolves the shared conv options for Unfold/Fold, which
// accept only kernel/stride/pad/dilation.
func resolveFoldOpts(what string, kernel int, opts []ConvOpt) convOpts {
	o := resolveConvOpts(2, kernel, opts)
	if o.groups != 1 {
		panic(fmt.Sprintf("nn: %s does not support groups", what))
	}
	for d := range o.outPad {
		if o.outPad[d] != 0 {
			panic(fmt.Sprintf("nn: %s does not support output_padding", what))
		}
	}
	return o
}

// Unfold extracts sliding local blocks from a batched 4D input (im2col):
// (N, C, H, W) -> (N, C*KH*KW, L), where L is the number of block positions
// and each output column l holds one block flattened in (C, kh, kw) order —
// PyTorch torch.nn.Unfold semantics.
//
// Configure with WithKernel/WithStride/WithPad/WithDilation (one value
// broadcasts, or one per spatial dim).
type Unfold struct {
	Base
	Kernel, Stride, Pad, Dilation []int

	// Single-entry gather cache keyed by input spatial size (same pattern as
	// convNd: the matrix is a constant leaf, safe to reuse across forwards).
	cacheKey                    []int
	cachedG                     *tensor.Tensor
	cachedNumWin, cachedWinSize int
}

// NewUnfold creates an Unfold with a square kernel by default; override
// per-dim geometry with WithKernel/WithStride/WithPad/WithDilation.
func NewUnfold(kernel int, opts ...ConvOpt) *Unfold {
	o := resolveFoldOpts("Unfold", kernel, opts)
	return &Unfold{Kernel: o.kernel, Stride: o.stride, Pad: o.pad, Dilation: o.dilation}
}

func (u *Unfold) gatherFor(spatial []int) (*tensor.Tensor, int, int) {
	if u.cachedG != nil && intsEqual(u.cacheKey, spatial) {
		return u.cachedG, u.cachedNumWin, u.cachedWinSize
	}
	spec := slidingSpec{
		In: append([]int(nil), spatial...), Kernel: u.Kernel,
		Stride: u.Stride, Pad: u.Pad, Dilation: u.Dilation,
	}
	g, _, numWin, winSize := gatherMatrix(spec)
	u.cacheKey = spec.In
	u.cachedG, u.cachedNumWin, u.cachedWinSize = g, numWin, winSize
	return g, numWin, winSize
}

// Forward maps (N, C, H, W) to (N, C*KH*KW, L). The shared unfold helper
// yields (N*L, C*winSize) with (channel, kernel-offset) column order; a
// reshape + permute turns that into PyTorch's (N, C*winSize, L) layout —
// rows ordered (C, kh, kw), columns ordered by block position row-major.
func (u *Unfold) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic(fmt.Sprintf("nn: Unfold expected 4D input (N,C,H,W), got shape %v", x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	g, numWin, winSize := u.gatherFor(x.Shape[2:])
	return unfold(x, g, numWin, winSize). // (N*L, C*winSize)
						Reshape(N, numWin, C*winSize).
						Permute(0, 2, 1) // (N, C*winSize, L)
}

// Fold combines an array of sliding local blocks into a 4D output tensor
// (col2im): (N, C*KH*KW, L) -> (N, C, H, W), summing values where blocks
// overlap — PyTorch torch.nn.Fold semantics. outputSize is the target
// spatial size (H, W); L must equal the number of block positions that
// Unfold with the same geometry would produce over that size.
type Fold struct {
	Base
	OutputSize                    [2]int
	Kernel, Stride, Pad, Dilation []int

	g               *tensor.Tensor // gather for In=OutputSize, built once
	numWin, winSize int
}

// NewFold creates a Fold targeting the given output spatial size. Same
// options as NewUnfold.
func NewFold(outputSize [2]int, kernel int, opts ...ConvOpt) *Fold {
	o := resolveFoldOpts("Fold", kernel, opts)
	f := &Fold{OutputSize: outputSize, Kernel: o.kernel, Stride: o.stride, Pad: o.pad, Dilation: o.dilation}
	spec := slidingSpec{
		In: []int{outputSize[0], outputSize[1]}, Kernel: f.Kernel,
		Stride: f.Stride, Pad: f.Pad, Dilation: f.Dilation,
	}
	f.g, _, f.numWin, f.winSize = gatherMatrix(spec)
	return f
}

// Forward maps (N, C*KH*KW, L) to (N, C, H, W). It is the exact transpose of
// Unfold: the blocks are rearranged into the (N*C, L*winSize) layout and
// multiplied by the gather matrix G (rather than G^T), which scatters every
// block entry back to its source position; positions covered by several
// blocks accumulate their sum.
func (f *Fold) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic(fmt.Sprintf("nn: Fold expected 3D input (N, C*prod(kernel), L), got shape %v", x.Shape))
	}
	N, CW, L := x.Shape[0], x.Shape[1], x.Shape[2]
	if CW%f.winSize != 0 {
		panic(fmt.Sprintf("nn: Fold input dim 1 (%d) not divisible by prod(kernel)=%d", CW, f.winSize))
	}
	if L != f.numWin {
		panic(fmt.Sprintf("nn: Fold expected %d blocks for output %v with kernel=%v stride=%v pad=%v dilation=%v, got %d",
			f.numWin, f.OutputSize, f.Kernel, f.Stride, f.Pad, f.Dilation, L))
	}
	C := CW / f.winSize
	col := x.Reshape(N, C, f.winSize, L).
		Permute(0, 1, 3, 2). // (N, C, L, winSize)
		Reshape(N*C, L*f.winSize)
	out := col.MatMul(f.g) // (N*C, H*W): G sums overlapping taps
	return out.Reshape(N, C, f.OutputSize[0], f.OutputSize[1])
}
