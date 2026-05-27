package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// Conv3d performs 3D convolution on inputs of shape (N, C, D, H, W).
type Conv3d struct {
	InC, OutC                  int
	KD, KH, KW                 int
	StrideD, StrideH, StrideW  int
	PadD, PadH, PadW           int
	Weight                     *tensor.Tensor // (OutC, InC, KD, KH, KW)
	Bias                       *tensor.Tensor // (OutC,) or nil
}

// NewConv3d creates a Conv3d with symmetric kernel/stride/padding.
func NewConv3d(inC, outC, kernel, stride, padding int, bias bool) *Conv3d {
	return NewConv3dDHW(inC, outC, kernel, kernel, kernel, stride, stride, stride, padding, padding, padding, bias)
}

// NewConv3dDHW is the general constructor allowing different D/H/W kernel/stride/pad.
func NewConv3dDHW(inC, outC, kd, kh, kw, sd, sh, sw, pd, ph, pw int, bias bool) *Conv3d {
	if sd <= 0 {
		sd = 1
	}
	if sh <= 0 {
		sh = 1
	}
	if sw <= 0 {
		sw = 1
	}
	fanIn := inC * kd * kh * kw
	bound := math.Sqrt(1.0 / float64(fanIn))
	wData := make([]float64, outC*inC*kd*kh*kw)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, outC, inC, kd, kh, kw).SetRequiresGrad(true)
	c := &Conv3d{
		InC: inC, OutC: outC,
		KD: kd, KH: kh, KW: kw,
		StrideD: sd, StrideH: sh, StrideW: sw,
		PadD: pd, PadH: ph, PadW: pw,
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

// Forward applies conv3d using a gather matrix + matmul.
func (c *Conv3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic("Conv3d.Forward: expected 5D input (N,C,D,H,W)")
	}
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	if C != c.InC {
		panic("Conv3d.Forward: input channels mismatch")
	}
	outD := (D+2*c.PadD-c.KD)/c.StrideD + 1
	outH := (H+2*c.PadH-c.KH)/c.StrideH + 1
	outW := (W+2*c.PadW-c.KW)/c.StrideW + 1

	rows := outD * outH * outW * C * c.KD * c.KH * c.KW
	cols := C * D * H * W
	gData := make([]float64, rows*cols)
	for od := 0; od < outD; od++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for cc := 0; cc < C; cc++ {
					for kd := 0; kd < c.KD; kd++ {
						di := od*c.StrideD + kd - c.PadD
						for ki := 0; ki < c.KH; ki++ {
							hi := oh*c.StrideH + ki - c.PadH
							for kj := 0; kj < c.KW; kj++ {
								wi := ow*c.StrideW + kj - c.PadW
								row := ((((od*outH+oh)*outW+ow)*C+cc)*c.KD+kd)*c.KH*c.KW + ki*c.KW + kj
								if di >= 0 && di < D && hi >= 0 && hi < H && wi >= 0 && wi < W {
									col := ((cc*D+di)*H+hi)*W + wi
									gData[row*cols+col] = 1
								}
							}
						}
					}
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	xFlat := x.Reshape(N, cols)
	col := xFlat.MatMul(g.Transpose()) // (N, rows)
	col2 := col.Reshape(N*outD*outH*outW, C*c.KD*c.KH*c.KW)

	wMat := c.Weight.Reshape(c.OutC, c.InC*c.KD*c.KH*c.KW)
	out := col2.MatMul(wMat.Transpose()) // (N*outD*outH*outW, OutC)
	if c.Bias != nil {
		out = out.Add(c.Bias)
	}
	// (N, outD, outH, outW, OutC) -> (N, OutC, outD, outH, outW)
	out = out.Reshape(N, outD, outH, outW, c.OutC).Permute(0, 4, 1, 2, 3)
	return out
}

// Parameters returns weight and (optional) bias.
func (c *Conv3d) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.Weight}
	if c.Bias != nil {
		ps = append(ps, c.Bias)
	}
	return ps
}
