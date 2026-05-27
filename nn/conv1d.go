package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// Conv1d performs 1D convolution on inputs of shape (N, C, L).
type Conv1d struct {
	InC, OutC int
	K         int
	Stride    int
	Pad       int
	Weight    *tensor.Tensor // (OutC, InC, K)
	Bias      *tensor.Tensor // (OutC,) or nil
}

// NewConv1d creates a Conv1d with Kaiming-uniform-initialized weights.
func NewConv1d(inC, outC, kernel, stride, padding int, bias bool) *Conv1d {
	if stride <= 0 {
		stride = 1
	}
	fanIn := inC * kernel
	bound := math.Sqrt(1.0 / float64(fanIn))
	wData := make([]float64, outC*inC*kernel)
	for i := range wData {
		wData[i] = -bound + rand.Float64()*(2*bound)
	}
	w := tensor.New(wData, outC, inC, kernel).SetRequiresGrad(true)
	c := &Conv1d{
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

// Forward applies conv1d using a gather matrix + matmul (im2col-style).
func (c *Conv1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("Conv1d.Forward: expected 3D input (N,C,L)")
	}
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	if C != c.InC {
		panic("Conv1d.Forward: input channels mismatch")
	}
	outL := (L+2*c.Pad-c.K)/c.Stride + 1

	rows := outL * C * c.K
	cols := C * L
	gData := make([]float64, rows*cols)
	for ol := 0; ol < outL; ol++ {
		for cc := 0; cc < C; cc++ {
			for ki := 0; ki < c.K; ki++ {
				li := ol*c.Stride + ki - c.Pad
				row := (ol*C+cc)*c.K + ki
				if li >= 0 && li < L {
					col := cc*L + li
					gData[row*cols+col] = 1
				}
			}
		}
	}
	g := tensor.New(gData, rows, cols)

	xFlat := x.Reshape(N, cols)
	col := xFlat.MatMul(g.Transpose())     // (N, rows)
	col2 := col.Reshape(N*outL, C*c.K)     // (N*outL, C*K)

	wMat := c.Weight.Reshape(c.OutC, c.InC*c.K)
	out := col2.MatMul(wMat.Transpose()) // (N*outL, OutC)
	if c.Bias != nil {
		out = out.Add(c.Bias)
	}
	// (N, outL, OutC) -> (N, OutC, outL)
	out = out.Reshape(N, outL, c.OutC).Permute(0, 2, 1)
	return out
}

// Parameters returns weight and (optional) bias.
func (c *Conv1d) Parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{c.Weight}
	if c.Bias != nil {
		ps = append(ps, c.Bias)
	}
	return ps
}
