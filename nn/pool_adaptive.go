package nn

import (
	"fmt"

	"gonn/tensor"
)

// Adaptive pooling: output size is fixed and the window for each output cell
// is computed from the input size (PyTorch semantics: start = floor(i*in/out),
// end = ceil((i+1)*in/out)). One rank-generic core serves all six layers;
// each output cell gathers its (variably sized) window and reduces it, and
// the per-cell results are concatenated (autograd-aware).

// startEnd returns the start (inclusive) and end (exclusive) indices for the
// i-th adaptive pool window over an input of size `in` producing `out` cells.
func startEnd(i, in, out int) (int, int) {
	s := (i * in) / out
	num := (i + 1) * in
	e := num / out
	if num%out != 0 {
		e++
	}
	if s < 0 {
		s = 0
	}
	if e > in {
		e = in
	}
	if e <= s { // degenerate window: fall back to a single cell
		s = i * in / out
		if s >= in {
			s = in - 1
		}
		e = s + 1
	}
	return s, e
}

// adaptivePoolNd reduces x (N, C, spatial...) to the given output spatial
// sizes with per-cell max or mean.
func adaptivePoolNd(x *tensor.Tensor, outSizes []int, isMax bool) *tensor.Tensor {
	rank := len(outSizes)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: adaptive pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	in := x.Shape[2:]
	cols := prodInts(in)
	inStrides := rowMajorStrides(in)
	xFlat := x.Reshape(N*C, cols)

	outTotal := prodInts(outSizes)
	outs := make([]*tensor.Tensor, outTotal)
	cellIdx := make([]int, rank)
	for cell := 0; cell < outTotal; cell++ {
		// Window bounds per dim for this output cell.
		starts := make([]int, rank)
		lens := make([]int, rank)
		win := 1
		for d := 0; d < rank; d++ {
			s, e := startEnd(cellIdx[d], in[d], outSizes[d])
			starts[d] = s
			lens[d] = e - s
			win *= lens[d]
		}
		// Gather matrix (win, cols) selecting the window cells in row-major
		// window order.
		gData := make([]float64, win*cols)
		wIdx := make([]int, rank)
		for r := 0; r < win; r++ {
			col := 0
			for d := 0; d < rank; d++ {
				col += (starts[d] + wIdx[d]) * inStrides[d]
			}
			gData[r*cols+col] = 1
			incMultiIndex(wIdx, lens)
		}
		g := tensor.New(gData, win, cols)
		w := xFlat.MatMul(g.Transpose()) // (N*C, win)
		if isMax {
			outs[cell] = w.MaxAxis(1, true) // (N*C, 1)
		} else {
			outs[cell] = w.MeanAxis(1, true)
		}
		incMultiIndex(cellIdx, outSizes)
	}
	cat := tensor.Concat(1, outs...) // (N*C, outTotal)
	return cat.Reshape(append([]int{N, C}, outSizes...)...)
}

// AdaptiveAvgPool1d applies adaptive average pooling on (N, C, L) -> (N, C, OutSize).
type AdaptiveAvgPool1d struct {
	Base
	OutSize int
}

// NewAdaptiveAvgPool1d constructs the layer.
func NewAdaptiveAvgPool1d(outSize int) *AdaptiveAvgPool1d {
	return &AdaptiveAvgPool1d{OutSize: outSize}
}

// Forward computes per-cell mean over the dynamically sized window.
func (p *AdaptiveAvgPool1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return adaptivePoolNd(x, []int{p.OutSize}, false)
}

// AdaptiveMaxPool1d applies adaptive max pooling on (N, C, L).
type AdaptiveMaxPool1d struct {
	Base
	OutSize int
}

// NewAdaptiveMaxPool1d constructs the layer.
func NewAdaptiveMaxPool1d(outSize int) *AdaptiveMaxPool1d {
	return &AdaptiveMaxPool1d{OutSize: outSize}
}

// Forward computes per-cell max over the dynamically sized window.
func (p *AdaptiveMaxPool1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return adaptivePoolNd(x, []int{p.OutSize}, true)
}

// AdaptiveAvgPool2d applies adaptive average pooling on (N, C, H, W) -> (N, C, OutH, OutW).
type AdaptiveAvgPool2d struct {
	Base
	OutH, OutW int
}

// NewAdaptiveAvgPool2d constructs the layer.
func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{OutH: outH, OutW: outW}
}

// Forward computes per-cell mean over each 2D adaptive window.
func (p *AdaptiveAvgPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return adaptivePoolNd(x, []int{p.OutH, p.OutW}, false)
}

// AdaptiveMaxPool2d applies adaptive max pooling on (N, C, H, W).
type AdaptiveMaxPool2d struct {
	Base
	OutH, OutW int
}

// NewAdaptiveMaxPool2d constructs the layer.
func NewAdaptiveMaxPool2d(outH, outW int) *AdaptiveMaxPool2d {
	return &AdaptiveMaxPool2d{OutH: outH, OutW: outW}
}

// Forward computes per-cell max over each 2D adaptive window.
func (p *AdaptiveMaxPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return adaptivePoolNd(x, []int{p.OutH, p.OutW}, true)
}

// AdaptiveAvgPool3d applies adaptive average pooling on (N, C, D, H, W).
type AdaptiveAvgPool3d struct {
	Base
	OutD, OutH, OutW int
}

// NewAdaptiveAvgPool3d constructs the layer.
func NewAdaptiveAvgPool3d(outD, outH, outW int) *AdaptiveAvgPool3d {
	return &AdaptiveAvgPool3d{OutD: outD, OutH: outH, OutW: outW}
}

// Forward computes per-cell mean over each 3D adaptive window.
func (p *AdaptiveAvgPool3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return adaptivePoolNd(x, []int{p.OutD, p.OutH, p.OutW}, false)
}

// AdaptiveMaxPool3d applies adaptive max pooling on (N, C, D, H, W).
type AdaptiveMaxPool3d struct {
	Base
	OutD, OutH, OutW int
}

// NewAdaptiveMaxPool3d constructs the layer.
func NewAdaptiveMaxPool3d(outD, outH, outW int) *AdaptiveMaxPool3d {
	return &AdaptiveMaxPool3d{OutD: outD, OutH: outH, OutW: outW}
}

// Forward computes per-cell max over each 3D adaptive window.
func (p *AdaptiveMaxPool3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return adaptivePoolNd(x, []int{p.OutD, p.OutH, p.OutW}, true)
}
