package nn

import (
	"fmt"

	"gonn/tensor"
)

// The shared N-dimensional sliding-window machinery behind every conv,
// conv-transpose, and pooling layer: a per-channel 0/1 gather matrix G of
// shape (numWin*winSize, prod(In)) that, applied to a flattened input,
// extracts every window. Rows that tap padding are all-zero (zero-pad
// semantics). Keeping G per-channel (channels handled by unfold's reshape/
// permute) makes the matrix a factor C^2 smaller than the historical
// channel-inclusive conv gathers. Everything stays composed from
// differentiable tensor ops, so autograd works by construction, and gather
// selection is exact (no floating-point reordering).

// slidingSpec describes an N-dimensional sliding-window read pattern over
// the spatial dims of an (N, C, spatial...) tensor. All slices have length
// equal to the spatial rank (1-3). OutPad (transposed convs only; nil means
// zero) appends extra output positions at the high edge of each dim.
type slidingSpec struct {
	In, Kernel, Stride, Pad, Dilation []int
	OutPad                            []int
}

// outSizes returns the output spatial dims for a forward convolution/pool:
// out = (In + 2*Pad - Dilation*(Kernel-1) - 1)/Stride + 1.
func (s slidingSpec) outSizes() []int {
	out := make([]int, len(s.In))
	for d := range s.In {
		out[d] = (s.In[d]+2*s.Pad[d]-s.Dilation[d]*(s.Kernel[d]-1)-1)/s.Stride[d] + 1
		if out[d] <= 0 {
			panic(fmt.Sprintf("nn: non-positive output size %d for dim %d (in=%d kernel=%d stride=%d pad=%d dilation=%d)",
				out[d], d, s.In[d], s.Kernel[d], s.Stride[d], s.Pad[d], s.Dilation[d]))
		}
	}
	return out
}

// transposedOutSizes returns the output spatial dims for a transposed conv:
// out = (In-1)*Stride - 2*Pad + Dilation*(Kernel-1) + OutPad + 1.
// The OutPad positions sit at the high edge and have no source taps, so the
// gather naturally expresses them as all-zero rows.
func (s slidingSpec) transposedOutSizes() []int {
	out := make([]int, len(s.In))
	for d := range s.In {
		out[d] = (s.In[d]-1)*s.Stride[d] - 2*s.Pad[d] + s.Dilation[d]*(s.Kernel[d]-1) + 1
		if s.OutPad != nil {
			out[d] += s.OutPad[d]
		}
		if out[d] <= 0 {
			panic(fmt.Sprintf("nn: non-positive transposed output size %d for dim %d", out[d], d))
		}
	}
	return out
}

// prodInts multiplies a slice of dims.
func prodInts(dims []int) int {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return n
}

// rowMajorStrides returns row-major strides for dims.
func rowMajorStrides(dims []int) []int {
	s := make([]int, len(dims))
	if len(dims) == 0 {
		return s
	}
	s[len(dims)-1] = 1
	for d := len(dims) - 2; d >= 0; d-- {
		s[d] = s[d+1] * dims[d+1]
	}
	return s
}

// incMultiIndex advances a row-major multi-index over dims; used to iterate
// windows and kernel offsets without per-rank loops.
func incMultiIndex(idx, dims []int) {
	for d := len(idx) - 1; d >= 0; d-- {
		idx[d]++
		if idx[d] < dims[d] {
			return
		}
		idx[d] = 0
	}
}

// gatherMatrix builds the forward-conv/pool gather: row (w*winSize + kk) has
// a single 1 at the input flat index read by window w at kernel offset kk,
// where input position = winIdx*Stride + kIdx*Dilation - Pad per dim; rows
// whose position falls outside the input stay all-zero (zero padding).
// Returns (G, outSizes, numWin, winSize).
func gatherMatrix(s slidingSpec) (*tensor.Tensor, []int, int, int) {
	out := s.outSizes()
	return buildGather(s, out, func(winIdx, kIdx []int) (int, bool) {
		col := 0
		for d := range s.In {
			pos := winIdx[d]*s.Stride[d] + kIdx[d]*s.Dilation[d] - s.Pad[d]
			if pos < 0 || pos >= s.In[d] {
				return 0, false
			}
			col += pos * s.inStride(d)
		}
		return col, true
	})
}

// transposedGatherMatrix builds the transposed-conv gather with the inverted
// index math: window o at kernel offset kk reads input i where
// i = (o + Pad - Dilation*kk)/Stride, only when that division is exact and
// in range. Returns (G, outSizes, numWin, winSize).
func transposedGatherMatrix(s slidingSpec) (*tensor.Tensor, []int, int, int) {
	out := s.transposedOutSizes()
	return buildGather(s, out, func(winIdx, kIdx []int) (int, bool) {
		col := 0
		for d := range s.In {
			num := winIdx[d] + s.Pad[d] - kIdx[d]*s.Dilation[d]
			if num < 0 || num%s.Stride[d] != 0 {
				return 0, false
			}
			pos := num / s.Stride[d]
			if pos >= s.In[d] {
				return 0, false
			}
			col += pos * s.inStride(d)
		}
		return col, true
	})
}

// inStride returns the row-major stride of input spatial dim d.
func (s slidingSpec) inStride(d int) int {
	stride := 1
	for i := d + 1; i < len(s.In); i++ {
		stride *= s.In[i]
	}
	return stride
}

// buildGather iterates all (window, kernel-offset) pairs, asking source for
// the input flat index (or false for a zero row).
func buildGather(s slidingSpec, out []int, source func(winIdx, kIdx []int) (int, bool)) (*tensor.Tensor, []int, int, int) {
	numWin := prodInts(out)
	winSize := prodInts(s.Kernel)
	cols := prodInts(s.In)
	gData := make([]float64, numWin*winSize*cols)

	winIdx := make([]int, len(s.In))
	for w := 0; w < numWin; w++ {
		kIdx := make([]int, len(s.In))
		for kk := 0; kk < winSize; kk++ {
			if col, ok := source(winIdx, kIdx); ok {
				gData[(w*winSize+kk)*cols+col] = 1
			}
			incMultiIndex(kIdx, s.Kernel)
		}
		incMultiIndex(winIdx, out)
	}
	return tensor.New(gData, numWin*winSize, cols), out, numWin, winSize
}

// unfold applies the gather G to x (N, C, spatial...) and returns the im2col
// matrix (N*numWin, C*winSize). The Permute moves channels next to the
// kernel offsets, which is what lets G stay per-channel: column order within
// a row is (channel, kernel-offset) — exactly the layout of a flattened
// (OutC, InC, K...) weight.
func unfold(x *tensor.Tensor, g *tensor.Tensor, numWin, winSize int) *tensor.Tensor {
	N, C := x.Shape[0], x.Shape[1]
	S := prodInts(x.Shape[2:])
	return x.Reshape(N*C, S).
		MatMul(g.Transpose()). // (N*C, numWin*winSize)
		Reshape(N, C, numWin, winSize).
		Permute(0, 2, 1, 3). // (N, numWin, C, winSize)
		Reshape(N*numWin, C*winSize)
}

// indexMapGather builds a (len(indexMap), cols) matrix with a 1 at
// (r, indexMap[r]) for each non-negative entry — the padding/upsample
// pattern where every output cell reads at most one input cell.
func indexMapGather(indexMap []int, cols int) *tensor.Tensor {
	rows := len(indexMap)
	gData := make([]float64, rows*cols)
	for r, src := range indexMap {
		if src >= 0 && src < cols {
			gData[r*cols+src] = 1
		}
	}
	return tensor.New(gData, rows, cols)
}
