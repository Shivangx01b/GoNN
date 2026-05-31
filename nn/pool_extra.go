package nn

import "gonn/tensor"

// poolGather builds a sparse gather matrix (rows = numWindows*windowSize,
// cols = spatialSize) of 1s that, multiplied against a flattened (N*C, cols)
// input, extracts every pooling window. It mirrors the construction used by the
// 2D pooling implementation in pool.go, generalized to an arbitrary number of
// spatial dimensions.
//
// in/k/stride are slices over the spatial dims (e.g. [W] for 1d, [D,H,W] for
// 3d). It returns the gather tensor, the per-dimension output sizes, the number
// of output windows, and the window size.
func poolGather(in, k, stride []int) (*tensor.Tensor, []int, int, int) {
	nd := len(in)
	out := make([]int, nd)
	numWin := 1
	winSize := 1
	cols := 1
	for d := 0; d < nd; d++ {
		out[d] = (in[d]-k[d])/stride[d] + 1
		numWin *= out[d]
		winSize *= k[d]
		cols *= in[d]
	}
	rows := numWin * winSize
	gData := make([]float64, rows*cols)

	// Strides for the input spatial layout (row-major).
	inStride := make([]int, nd)
	inStride[nd-1] = 1
	for d := nd - 2; d >= 0; d-- {
		inStride[d] = inStride[d+1] * in[d+1]
	}

	// Iterate over every (window, kernel-offset) pair and set the gather bit.
	winIdx := make([]int, nd)
	for w := 0; w < numWin; w++ {
		kIdx := make([]int, nd)
		for ks := 0; ks < winSize; ks++ {
			col := 0
			for d := 0; d < nd; d++ {
				pos := winIdx[d]*stride[d] + kIdx[d]
				col += pos * inStride[d]
			}
			row := w*winSize + ks
			gData[row*cols+col] = 1
			// increment kernel multi-index (row-major over k dims)
			for d := nd - 1; d >= 0; d-- {
				kIdx[d]++
				if kIdx[d] < k[d] {
					break
				}
				kIdx[d] = 0
			}
		}
		// increment window multi-index (row-major over out dims)
		for d := nd - 1; d >= 0; d-- {
			winIdx[d]++
			if winIdx[d] < out[d] {
				break
			}
			winIdx[d] = 0
		}
	}
	return tensor.New(gData, rows, cols), out, numWin, winSize
}

// MaxPool1d performs 1D max pooling on (N, C, L) inputs.
type MaxPool1d struct {
	K, Stride int
}

// NewMaxPool1d creates a MaxPool1d. If stride <= 0 it defaults to the kernel.
func NewMaxPool1d(kernel, stride int) *MaxPool1d {
	if stride <= 0 {
		stride = kernel
	}
	return &MaxPool1d{K: kernel, Stride: stride}
}

// Forward applies 1D max pooling.
func (p *MaxPool1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	g, out, numWin, winSize := poolGather([]int{L}, []int{p.K}, []int{p.Stride})
	xFlat := x.Reshape(N*C, L)
	col := xFlat.MatMul(g.Transpose())
	win := col.Reshape(N*C, numWin, winSize)
	mx := win.MaxAxis(2, false)
	return mx.Reshape(N, C, out[0])
}

// Parameters returns nothing.
func (p *MaxPool1d) Parameters() []*tensor.Tensor { return nil }

// AvgPool1d performs 1D average pooling on (N, C, L) inputs.
type AvgPool1d struct {
	K, Stride int
}

// NewAvgPool1d creates an AvgPool1d. If stride <= 0 it defaults to the kernel.
func NewAvgPool1d(kernel, stride int) *AvgPool1d {
	if stride <= 0 {
		stride = kernel
	}
	return &AvgPool1d{K: kernel, Stride: stride}
}

// Forward applies 1D average pooling.
func (p *AvgPool1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	g, out, numWin, winSize := poolGather([]int{L}, []int{p.K}, []int{p.Stride})
	xFlat := x.Reshape(N*C, L)
	col := xFlat.MatMul(g.Transpose())
	win := col.Reshape(N*C, numWin, winSize)
	avg := win.MeanAxis(2, false)
	return avg.Reshape(N, C, out[0])
}

// Parameters returns nothing.
func (p *AvgPool1d) Parameters() []*tensor.Tensor { return nil }

// MaxPool3d performs 3D max pooling on (N, C, D, H, W) inputs.
type MaxPool3d struct {
	KD, KH, KW                int
	StrideD, StrideH, StrideW int
}

// NewMaxPool3d creates a cubic MaxPool3d. If stride <= 0 it defaults to kernel.
func NewMaxPool3d(kernel, stride int) *MaxPool3d {
	if stride <= 0 {
		stride = kernel
	}
	return &MaxPool3d{KD: kernel, KH: kernel, KW: kernel, StrideD: stride, StrideH: stride, StrideW: stride}
}

// Forward applies 3D max pooling.
func (p *MaxPool3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	g, out, numWin, winSize := poolGather(
		[]int{D, H, W}, []int{p.KD, p.KH, p.KW}, []int{p.StrideD, p.StrideH, p.StrideW})
	xFlat := x.Reshape(N*C, D*H*W)
	col := xFlat.MatMul(g.Transpose())
	win := col.Reshape(N*C, numWin, winSize)
	mx := win.MaxAxis(2, false)
	return mx.Reshape(N, C, out[0], out[1], out[2])
}

// Parameters returns nothing.
func (p *MaxPool3d) Parameters() []*tensor.Tensor { return nil }

// AvgPool3d performs 3D average pooling on (N, C, D, H, W) inputs.
type AvgPool3d struct {
	KD, KH, KW                int
	StrideD, StrideH, StrideW int
}

// NewAvgPool3d creates a cubic AvgPool3d. If stride <= 0 it defaults to kernel.
func NewAvgPool3d(kernel, stride int) *AvgPool3d {
	if stride <= 0 {
		stride = kernel
	}
	return &AvgPool3d{KD: kernel, KH: kernel, KW: kernel, StrideD: stride, StrideH: stride, StrideW: stride}
}

// Forward applies 3D average pooling.
func (p *AvgPool3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	g, out, numWin, winSize := poolGather(
		[]int{D, H, W}, []int{p.KD, p.KH, p.KW}, []int{p.StrideD, p.StrideH, p.StrideW})
	xFlat := x.Reshape(N*C, D*H*W)
	col := xFlat.MatMul(g.Transpose())
	win := col.Reshape(N*C, numWin, winSize)
	avg := win.MeanAxis(2, false)
	return avg.Reshape(N, C, out[0], out[1], out[2])
}

// Parameters returns nothing.
func (p *AvgPool3d) Parameters() []*tensor.Tensor { return nil }
