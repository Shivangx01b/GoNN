package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// ConvTranspose1d performs 1D transposed convolution ("deconvolution") on
// inputs of shape (N, InC, L). Weight shape is (InC, OutC, K).
type ConvTranspose1d struct {
	InC, OutC int
	K         int
	Stride    int
	Pad       int
	Weight    *tensor.Tensor // (InC, OutC, K)
	Bias      *tensor.Tensor // (OutC,) or nil
}

// NewConvTranspose1d creates a ConvTranspose1d.
func NewConvTranspose1d(inC, outC, kernel, stride, padding int, bias bool) *ConvTranspose1d {
	if stride <= 0 {
		stride = 1
	}
	fanIn := inC * kernel
	bound := math.Sqrt(1.0 / float64(fanIn))
	wData := make([]float64, inC*outC*kernel)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, inC, outC, kernel).SetRequiresGrad(true)
	c := &ConvTranspose1d{
		InC: inC, OutC: outC,
		K:      kernel,
		Stride: stride,
		Pad:    padding,
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

// Forward applies transposed 1D conv: out_len = (L-1)*stride - 2*pad + kernel.
func (c *ConvTranspose1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("ConvTranspose1d.Forward: expected 3D input (N,C,L)")
	}
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	if C != c.InC {
		panic("ConvTranspose1d.Forward: input channels mismatch")
	}
	outL := (L-1)*c.Stride - 2*c.Pad + c.K
	if outL <= 0 {
		panic("ConvTranspose1d.Forward: non-positive output length")
	}

	// For each (outL, k) we read input position i = (ol + pad - k) / stride
	// if (ol + pad - k) % stride == 0 and 0 <= i < L.
	// Build a gather matrix from x (shape (N, C*L)) producing (N, outL*C*K)
	// where the win at (ol, cc, k) reads x[n, cc, i] (or zero).
	rows := outL * C * c.K
	cols := C * L
	gData := make([]float64, rows*cols)
	for ol := 0; ol < outL; ol++ {
		for cc := 0; cc < C; cc++ {
			for ki := 0; ki < c.K; ki++ {
				num := ol + c.Pad - ki
				row := (ol*C+cc)*c.K + ki
				if num < 0 || num%c.Stride != 0 {
					continue
				}
				li := num / c.Stride
				if li < 0 || li >= L {
					continue
				}
				col := cc*L + li
				gData[row*cols+col] = 1
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	xFlat := x.Reshape(N, cols)
	col := xFlat.MatMul(g.Transpose())     // (N, outL*C*K)
	col2 := col.Reshape(N*outL, C*c.K)     // (N*outL, InC*K)

	// Weight is (InC, OutC, K). For matmul we want (OutC, InC*K), i.e. flatten
	// over (InC, K) so column j corresponds to flat (cc, ki).
	wMat := c.Weight.Permute(1, 0, 2).Reshape(c.OutC, c.InC*c.K)
	out := col2.MatMul(wMat.Transpose()) // (N*outL, OutC)
	if c.Bias != nil {
		out = out.Add(c.Bias)
	}
	out = out.Reshape(N, outL, c.OutC).Permute(0, 2, 1)
	return out
}

// Parameters returns weight and (optional) bias.
func (c *ConvTranspose1d) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.Weight}
	if c.Bias != nil {
		ps = append(ps, c.Bias)
	}
	return ps
}

// ConvTranspose2d performs 2D transposed convolution on (N, InC, H, W).
// Weight shape is (InC, OutC, KH, KW).
type ConvTranspose2d struct {
	InC, OutC        int
	KH, KW           int
	StrideH, StrideW int
	PadH, PadW       int
	Weight           *tensor.Tensor // (InC, OutC, KH, KW)
	Bias             *tensor.Tensor // (OutC,) or nil
}

// NewConvTranspose2d creates a ConvTranspose2d with a symmetric kernel/stride/padding.
func NewConvTranspose2d(inC, outC, kernel, stride, padding int, bias bool) *ConvTranspose2d {
	return NewConvTranspose2dHW(inC, outC, kernel, kernel, stride, stride, padding, padding, bias)
}

// NewConvTranspose2dHW is the general constructor.
func NewConvTranspose2dHW(inC, outC, kh, kw, sh, sw, ph, pw int, bias bool) *ConvTranspose2d {
	if sh <= 0 {
		sh = 1
	}
	if sw <= 0 {
		sw = 1
	}
	fanIn := inC * kh * kw
	bound := math.Sqrt(1.0 / float64(fanIn))
	wData := make([]float64, inC*outC*kh*kw)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, inC, outC, kh, kw).SetRequiresGrad(true)
	c := &ConvTranspose2d{
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

// Forward applies transposed 2D conv.
//   outH = (H-1)*strideH - 2*padH + KH
//   outW = (W-1)*strideW - 2*padW + KW
func (c *ConvTranspose2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("ConvTranspose2d.Forward: expected 4D input (N,C,H,W)")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	if C != c.InC {
		panic("ConvTranspose2d.Forward: input channels mismatch")
	}
	outH := (H-1)*c.StrideH - 2*c.PadH + c.KH
	outW := (W-1)*c.StrideW - 2*c.PadW + c.KW
	if outH <= 0 || outW <= 0 {
		panic("ConvTranspose2d.Forward: non-positive output spatial dims")
	}

	rows := outH * outW * C * c.KH * c.KW
	cols := C * H * W
	gData := make([]float64, rows*cols)
	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			for cc := 0; cc < C; cc++ {
				for ki := 0; ki < c.KH; ki++ {
					numH := oh + c.PadH - ki
					for kj := 0; kj < c.KW; kj++ {
						numW := ow + c.PadW - kj
						row := ((oh*outW+ow)*C+cc)*c.KH*c.KW + ki*c.KW + kj
						if numH < 0 || numH%c.StrideH != 0 {
							continue
						}
						if numW < 0 || numW%c.StrideW != 0 {
							continue
						}
						hi := numH / c.StrideH
						wi := numW / c.StrideW
						if hi < 0 || hi >= H || wi < 0 || wi >= W {
							continue
						}
						col := (cc*H+hi)*W + wi
						gData[row*cols+col] = 1
					}
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	xFlat := x.Reshape(N, cols)
	col := xFlat.MatMul(g.Transpose())
	col2 := col.Reshape(N*outH*outW, C*c.KH*c.KW)

	wMat := c.Weight.Permute(1, 0, 2, 3).Reshape(c.OutC, c.InC*c.KH*c.KW)
	out := col2.MatMul(wMat.Transpose())
	if c.Bias != nil {
		out = out.Add(c.Bias)
	}
	out = out.Reshape(N, outH, outW, c.OutC).Permute(0, 3, 1, 2)
	return out
}

// Parameters returns weight and (optional) bias.
func (c *ConvTranspose2d) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.Weight}
	if c.Bias != nil {
		ps = append(ps, c.Bias)
	}
	return ps
}

// ConvTranspose3d performs 3D transposed convolution on (N, InC, D, H, W).
// Weight shape is (InC, OutC, KD, KH, KW).
type ConvTranspose3d struct {
	InC, OutC                 int
	KD, KH, KW                int
	StrideD, StrideH, StrideW int
	PadD, PadH, PadW          int
	Weight                    *tensor.Tensor // (InC, OutC, KD, KH, KW)
	Bias                      *tensor.Tensor // (OutC,) or nil
}

// NewConvTranspose3d creates a ConvTranspose3d with symmetric kernel/stride/padding.
func NewConvTranspose3d(inC, outC, kernel, stride, padding int, bias bool) *ConvTranspose3d {
	return NewConvTranspose3dDHW(inC, outC, kernel, kernel, kernel, stride, stride, stride, padding, padding, padding, bias)
}

// NewConvTranspose3dDHW is the general constructor.
func NewConvTranspose3dDHW(inC, outC, kd, kh, kw, sd, sh, sw, pd, ph, pw int, bias bool) *ConvTranspose3d {
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
	wData := make([]float64, inC*outC*kd*kh*kw)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, inC, outC, kd, kh, kw).SetRequiresGrad(true)
	c := &ConvTranspose3d{
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

// Forward applies transposed 3D conv.
func (c *ConvTranspose3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic("ConvTranspose3d.Forward: expected 5D input (N,C,D,H,W)")
	}
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	if C != c.InC {
		panic("ConvTranspose3d.Forward: input channels mismatch")
	}
	outD := (D-1)*c.StrideD - 2*c.PadD + c.KD
	outH := (H-1)*c.StrideH - 2*c.PadH + c.KH
	outW := (W-1)*c.StrideW - 2*c.PadW + c.KW
	if outD <= 0 || outH <= 0 || outW <= 0 {
		panic("ConvTranspose3d.Forward: non-positive output spatial dims")
	}

	rows := outD * outH * outW * C * c.KD * c.KH * c.KW
	cols := C * D * H * W
	gData := make([]float64, rows*cols)
	for od := 0; od < outD; od++ {
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				for cc := 0; cc < C; cc++ {
					for kd := 0; kd < c.KD; kd++ {
						numD := od + c.PadD - kd
						for ki := 0; ki < c.KH; ki++ {
							numH := oh + c.PadH - ki
							for kj := 0; kj < c.KW; kj++ {
								numW := ow + c.PadW - kj
								row := ((((od*outH+oh)*outW+ow)*C+cc)*c.KD+kd)*c.KH*c.KW + ki*c.KW + kj
								if numD < 0 || numD%c.StrideD != 0 {
									continue
								}
								if numH < 0 || numH%c.StrideH != 0 {
									continue
								}
								if numW < 0 || numW%c.StrideW != 0 {
									continue
								}
								di := numD / c.StrideD
								hi := numH / c.StrideH
								wi := numW / c.StrideW
								if di < 0 || di >= D || hi < 0 || hi >= H || wi < 0 || wi >= W {
									continue
								}
								col := ((cc*D+di)*H+hi)*W + wi
								gData[row*cols+col] = 1
							}
						}
					}
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	xFlat := x.Reshape(N, cols)
	col := xFlat.MatMul(g.Transpose())
	col2 := col.Reshape(N*outD*outH*outW, C*c.KD*c.KH*c.KW)

	wMat := c.Weight.Permute(1, 0, 2, 3, 4).Reshape(c.OutC, c.InC*c.KD*c.KH*c.KW)
	out := col2.MatMul(wMat.Transpose())
	if c.Bias != nil {
		out = out.Add(c.Bias)
	}
	out = out.Reshape(N, outD, outH, outW, c.OutC).Permute(0, 4, 1, 2, 3)
	return out
}

// Parameters returns weight and (optional) bias.
func (c *ConvTranspose3d) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.Weight}
	if c.Bias != nil {
		ps = append(ps, c.Bias)
	}
	return ps
}
