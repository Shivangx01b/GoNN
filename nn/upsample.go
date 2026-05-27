package nn

import "gonn/tensor"

// Upsample upsamples (N, C, H, W) tensors by an integer ScaleFactor.
// Mode "nearest" repeats each cell; "bilinear" performs bilinear interpolation.
type Upsample struct {
	ScaleFactor int
	Mode        string
}

// NewUpsample constructs an Upsample. Mode defaults to "nearest" if empty.
func NewUpsample(scaleFactor int, mode string) *Upsample {
	if scaleFactor <= 0 {
		panic("Upsample: scaleFactor must be positive")
	}
	if mode == "" {
		mode = "nearest"
	}
	if mode != "nearest" && mode != "bilinear" {
		panic("Upsample: mode must be 'nearest' or 'bilinear'")
	}
	return &Upsample{ScaleFactor: scaleFactor, Mode: mode}
}

// Forward upsamples a 4D tensor.
func (u *Upsample) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("Upsample: expected 4D input (N,C,H,W)")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	r := u.ScaleFactor
	outH := H * r
	outW := W * r

	switch u.Mode {
	case "nearest":
		// Build (outH*outW, H*W) gather matrix.
		rows := outH * outW
		cols := H * W
		gData := make([]float64, rows*cols)
		for oh := 0; oh < outH; oh++ {
			ih := oh / r
			for ow := 0; ow < outW; ow++ {
				iw := ow / r
				gData[(oh*outW+ow)*cols+(ih*W+iw)] = 1
			}
		}
		g := tensor.New(gData, rows, cols)
		xFlat := x.Reshape(N*C, cols)
		out := xFlat.MatMul(g.Transpose())
		return out.Reshape(N, C, outH, outW)

	case "bilinear":
		// PyTorch-style "align_corners=False" sampling: for each output cell
		// (oh, ow), source coord is ((oh + 0.5)/scale) - 0.5 (and same for W).
		// We compute weighted sum over 4 neighbours with clamped indices.
		rows := outH * outW
		cols := H * W
		gData := make([]float64, rows*cols)
		for oh := 0; oh < outH; oh++ {
			y := (float64(oh)+0.5)/float64(r) - 0.5
			y0 := int(floorInt(y))
			y1 := y0 + 1
			wy := y - float64(y0)
			// Clamp.
			y0c := clampInt(y0, 0, H-1)
			y1c := clampInt(y1, 0, H-1)
			for ow := 0; ow < outW; ow++ {
				x_ := (float64(ow)+0.5)/float64(r) - 0.5
				x0 := int(floorInt(x_))
				x1 := x0 + 1
				wx := x_ - float64(x0)
				x0c := clampInt(x0, 0, W-1)
				x1c := clampInt(x1, 0, W-1)
				row := oh*outW + ow
				// 4 contributions
				gData[row*cols+(y0c*W+x0c)] += (1 - wy) * (1 - wx)
				gData[row*cols+(y0c*W+x1c)] += (1 - wy) * wx
				gData[row*cols+(y1c*W+x0c)] += wy * (1 - wx)
				gData[row*cols+(y1c*W+x1c)] += wy * wx
			}
		}
		g := tensor.New(gData, rows, cols)
		xFlat := x.Reshape(N*C, cols)
		out := xFlat.MatMul(g.Transpose())
		return out.Reshape(N, C, outH, outW)
	}
	panic("Upsample: unsupported mode")
}

// Parameters returns nothing.
func (u *Upsample) Parameters() []*tensor.Tensor { return nil }

func floorInt(v float64) float64 {
	// math.Floor equivalent without importing math everywhere (avoid clash).
	i := int(v)
	if v < 0 && float64(i) != v {
		i--
	}
	return float64(i)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
