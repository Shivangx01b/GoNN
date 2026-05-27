package nn

import "gonn/tensor"

// MaxPool2d performs 2D max pooling on (N, C, H, W) inputs.
type MaxPool2d struct {
	KH, KW           int
	StrideH, StrideW int
}

// NewMaxPool2d creates a square MaxPool2d.
func NewMaxPool2d(kernel, stride int) *MaxPool2d {
	if stride <= 0 {
		stride = kernel
	}
	return &MaxPool2d{KH: kernel, KW: kernel, StrideH: stride, StrideW: stride}
}

// Forward applies max pooling using a gather to build windows, then MaxAxis.
func (p *MaxPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (H-p.KH)/p.StrideH + 1
	outW := (W-p.KW)/p.StrideW + 1

	// Build gather mat (outH*outW*KH*KW, H*W) per-channel reused via reshape.
	rows := outH * outW * p.KH * p.KW
	cols := H * W
	gData := make([]float64, rows*cols)
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			for ki := 0; ki < p.KH; ki++ {
				hi := oh*p.StrideH + ki
				for kj := 0; kj < p.KW; kj++ {
					wi := ow*p.StrideW + kj
					row := ((oh*outW+ow)*p.KH+ki)*p.KW + kj
					col := hi*W + wi
					gData[row*cols+col] = 1
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	// Reshape x to (N*C, H*W), gather -> (N*C, rows)
	xFlat := x.Reshape(N*C, cols)
	col := xFlat.MatMul(g.Transpose())
	// (N*C, outH*outW, KH*KW)
	win := col.Reshape(N*C, outH*outW, p.KH*p.KW)
	// max over last axis -> (N*C, outH*outW)
	mx := win.MaxAxis(2, false)
	return mx.Reshape(N, C, outH, outW)
}

// Parameters returns nothing.
func (p *MaxPool2d) Parameters() []*tensor.Tensor { return nil }

// AvgPool2d performs 2D average pooling on (N, C, H, W) inputs.
type AvgPool2d struct {
	KH, KW           int
	StrideH, StrideW int
}

// NewAvgPool2d creates a square AvgPool2d.
func NewAvgPool2d(kernel, stride int) *AvgPool2d {
	if stride <= 0 {
		stride = kernel
	}
	return &AvgPool2d{KH: kernel, KW: kernel, StrideH: stride, StrideW: stride}
}

// Forward applies average pooling.
func (p *AvgPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := (H-p.KH)/p.StrideH + 1
	outW := (W-p.KW)/p.StrideW + 1

	rows := outH * outW * p.KH * p.KW
	cols := H * W
	gData := make([]float64, rows*cols)
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			for ki := 0; ki < p.KH; ki++ {
				hi := oh*p.StrideH + ki
				for kj := 0; kj < p.KW; kj++ {
					wi := ow*p.StrideW + kj
					row := ((oh*outW+ow)*p.KH+ki)*p.KW + kj
					col := hi*W + wi
					gData[row*cols+col] = 1
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	xFlat := x.Reshape(N*C, cols)
	col := xFlat.MatMul(g.Transpose())
	win := col.Reshape(N*C, outH*outW, p.KH*p.KW)
	avg := win.MeanAxis(2, false)
	return avg.Reshape(N, C, outH, outW)
}

// Parameters returns nothing.
func (p *AvgPool2d) Parameters() []*tensor.Tensor { return nil }
