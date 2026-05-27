package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// Conv2d performs 2D convolution on inputs of shape (N, C, H, W).
type Conv2d struct {
	InC, OutC        int
	KH, KW           int
	StrideH, StrideW int
	PadH, PadW       int
	Weight           *tensor.Tensor // (OutC, InC, KH, KW)
	Bias             *tensor.Tensor // (OutC,) or nil
}

// NewConv2d creates a Conv2d with Kaiming-uniform-initialized weights.
func NewConv2d(inC, outC, kernel, stride, padding int, bias bool) *Conv2d {
	return NewConv2dHW(inC, outC, kernel, kernel, stride, stride, padding, padding, bias)
}

// NewConv2dHW is the general constructor allowing different H/W kernel/stride/pad.
func NewConv2dHW(inC, outC, kh, kw, sh, sw, ph, pw int, bias bool) *Conv2d {
	fanIn := inC * kh * kw
	bound := math.Sqrt(1.0 / float64(fanIn))
	wData := make([]float64, outC*inC*kh*kw)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, outC, inC, kh, kw).SetRequiresGrad(true)
	c := &Conv2d{
		InC: inC, OutC: outC,
		KH: kh, KW: kw,
		StrideH: sh, StrideW: sw,
		PadH: ph, PadW: pw,
		Weight: w,
	}
	if bias {
		bData := make([]float64, outC)
		for i := range bData {
			bData[i] = -bound + rand.Float64()*(2*bound)
		}
		c.Bias = tensor.New(bData, outC).SetRequiresGrad(true)
	}
	return c
}

// Forward applies conv2d using an im2col gather matrix + matmul. The gather
// matrix encodes padding by zeroing rows that fall outside the input.
func (c *Conv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("Conv2d.Forward: expected 4D input (N,C,H,W)")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	if C != c.InC {
		panic("Conv2d.Forward: input channels mismatch")
	}
	outH := (H+2*c.PadH-c.KH)/c.StrideH + 1
	outW := (W+2*c.PadW-c.KW)/c.StrideW + 1

	// Gather matrix: rows = outH*outW*C*KH*KW, cols = C*H*W.
	rows := outH * outW * C * c.KH * c.KW
	cols := C * H * W
	gData := make([]float64, rows*cols)
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			for cc := 0; cc < C; cc++ {
				for ki := 0; ki < c.KH; ki++ {
					hi := oh*c.StrideH + ki - c.PadH
					for kj := 0; kj < c.KW; kj++ {
						wi := ow*c.StrideW + kj - c.PadW
						row := ((oh*outW+ow)*C+cc)*c.KH*c.KW + ki*c.KW + kj
						if hi >= 0 && hi < H && wi >= 0 && wi < W {
							col := (cc*H+hi)*W + wi
							gData[row*cols+col] = 1
						}
					}
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols) // no grad

	// x_flat: (N, C*H*W)
	xFlat := x.Reshape(N, cols)
	// col = x_flat @ G^T -> (N, rows)
	col := xFlat.MatMul(g.Transpose())
	// reshape to (N*outH*outW, C*KH*KW)
	col2 := col.Reshape(N*outH*outW, C*c.KH*c.KW)

	// W as (OutC, C*KH*KW)
	wMat := c.Weight.Reshape(c.OutC, c.InC*c.KH*c.KW)
	// out = col2 @ wMat^T -> (N*outH*outW, OutC)
	out := col2.MatMul(wMat.Transpose())
	if c.Bias != nil {
		out = out.Add(c.Bias)
	}
	// (N, outH, outW, OutC) -> (N, OutC, outH, outW)
	out = out.Reshape(N, outH, outW, c.OutC).Permute(0, 3, 1, 2)
	return out
}

// Parameters returns weight and (optional) bias.
func (c *Conv2d) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.Weight}
	if c.Bias != nil {
		ps = append(ps, c.Bias)
	}
	return ps
}
