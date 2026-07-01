package nn

import (
	"fmt"

	"gonn/tensor"
)

// Upsample upsamples (N, C, spatial...) tensors by an integer ScaleFactor,
// matching torch.nn.Upsample for integer scale factors:
//
//   - 3D input (N, C, L):        modes "nearest", "linear"
//   - 4D input (N, C, H, W):     modes "nearest", "bilinear", "bicubic"
//   - 5D input (N, C, D, H, W):  modes "nearest", "trilinear"
//
// AlignCorners (default false, like PyTorch) affects only the interpolating
// modes: with align_corners=false source coordinates are (dst+0.5)/scale -
// 0.5 with edge clamping; with align_corners=true they are dst*(in-1)/
// (out-1), so the corner samples of input and output coincide. Nearest mode
// ignores it.
//
// Every mode lowers to a constant gather/weight matrix applied with MatMul,
// so gradients flow by construction and the layer dispatches to cuBLAS
// automatically on -tags cuda builds.
type Upsample struct {
	Base
	ScaleFactor  int
	Mode         string
	AlignCorners bool
}

// UpsampleOpt configures an Upsample.
type UpsampleOpt func(*Upsample)

// WithAlignCorners sets align_corners for the interpolation modes ("linear",
// "bilinear", "bicubic", "trilinear"). Default false (PyTorch parity).
func WithAlignCorners(on bool) UpsampleOpt {
	return func(u *Upsample) { u.AlignCorners = on }
}

// NewUpsample constructs an Upsample. Mode defaults to "nearest" if empty.
func NewUpsample(scaleFactor int, mode string, opts ...UpsampleOpt) *Upsample {
	if scaleFactor <= 0 {
		panic("Upsample: scaleFactor must be positive")
	}
	if mode == "" {
		mode = "nearest"
	}
	switch mode {
	case "nearest", "linear", "bilinear", "bicubic", "trilinear":
	default:
		panic("Upsample: mode must be 'nearest', 'linear', 'bilinear', 'bicubic' or 'trilinear'")
	}
	u := &Upsample{ScaleFactor: scaleFactor, Mode: mode}
	for _, fn := range opts {
		fn(u)
	}
	return u
}

// NewUpsamplingNearest2d is the torch.nn.UpsamplingNearest2d alias:
// equivalent to NewUpsample(scale, "nearest") applied to 4D input.
func NewUpsamplingNearest2d(scale int) *Upsample {
	return NewUpsample(scale, "nearest")
}

// NewUpsamplingBilinear2d is the torch.nn.UpsamplingBilinear2d alias:
// equivalent to NewUpsample(scale, "bilinear", WithAlignCorners(true)) —
// PyTorch defines this legacy module with align_corners=true.
func NewUpsamplingBilinear2d(scale int) *Upsample {
	return NewUpsample(scale, "bilinear", WithAlignCorners(true))
}

// Forward upsamples a 3D, 4D or 5D tensor according to Mode.
func (u *Upsample) Forward(x *tensor.Tensor) *tensor.Tensor {
	switch len(x.Shape) {
	case 3:
		return u.forward3D(x)
	case 4:
		return u.forward4D(x)
	case 5:
		return u.forward5D(x)
	default:
		panic(fmt.Sprintf("Upsample: expected 3D, 4D or 5D input, got shape %v", x.Shape))
	}
}

// srcCoord maps output index o to a (fractional) source coordinate along one
// axis of length in upsampled to out = in*r.
func srcCoord(o, in, out, r int, alignCorners bool) float64 {
	if alignCorners {
		if out <= 1 {
			return 0
		}
		return float64(o) * float64(in-1) / float64(out-1)
	}
	return (float64(o)+0.5)/float64(r) - 0.5
}

// linWeights returns, for each output index along one axis, the two source
// indices (clamped) and the weight of the upper one.
func linWeights(in, out, r int, alignCorners bool) (i0, i1 []int, w []float64) {
	i0 = make([]int, out)
	i1 = make([]int, out)
	w = make([]float64, out)
	for o := 0; o < out; o++ {
		s := srcCoord(o, in, out, r, alignCorners)
		lo := int(floorInt(s))
		w[o] = s - float64(lo)
		i0[o] = clampInt(lo, 0, in-1)
		i1[o] = clampInt(lo+1, 0, in-1)
	}
	return i0, i1, w
}

// cubicKernel evaluates the Keys (1981) cubic convolution kernel with
// a = -0.75, the coefficient PyTorch uses for its "bicubic" mode:
//
//	W(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1       for |t| <= 1
//	W(t) = a|t|^3 - 5a|t|^2 + 8a|t| - 4a     for 1 < |t| < 2
//	W(t) = 0                                 otherwise
func cubicKernel(t float64) float64 {
	const a = -0.75
	if t < 0 {
		t = -t
	}
	if t <= 1 {
		return ((a+2)*t-(a+3))*t*t + 1
	}
	if t < 2 {
		return ((a*t-5*a)*t+8*a)*t - 4*a
	}
	return 0
}

// cubicWeights returns, for each output index along one axis, the four source
// tap indices floor(s)-1 .. floor(s)+2 (clamped to [0, in-1], PyTorch border
// handling) and their cubic kernel weights W(dist), where s follows the same
// srcCoord convention as the linear modes. The four weights always sum to 1
// (the kernel is a partition of unity), so clamping never denormalizes a row.
func cubicWeights(in, out, r int, alignCorners bool) (idx [][4]int, w [][4]float64) {
	idx = make([][4]int, out)
	w = make([][4]float64, out)
	for o := 0; o < out; o++ {
		s := srcCoord(o, in, out, r, alignCorners)
		lo := int(floorInt(s))
		f := s - float64(lo)
		for k := 0; k < 4; k++ {
			idx[o][k] = clampInt(lo-1+k, 0, in-1)
			w[o][k] = cubicKernel(f - float64(k-1))
		}
	}
	return idx, w
}

// applyGather multiplies the flattened spatial dims of x by the transpose of
// the (outCols, inCols) gather/weight matrix g.
func applyGather(x *tensor.Tensor, gData []float64, rows, cols int, outShape ...int) *tensor.Tensor {
	g := tensor.New(gData, rows, cols)
	NC := x.Shape[0] * x.Shape[1]
	out := x.Reshape(NC, cols).MatMul(g.Transpose())
	return out.Reshape(outShape...)
}

// forward3D upsamples (N, C, L) with mode "nearest" or "linear".
func (u *Upsample) forward3D(x *tensor.Tensor) *tensor.Tensor {
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	r := u.ScaleFactor
	outL := L * r

	switch u.Mode {
	case "nearest":
		gData := make([]float64, outL*L)
		for ol := 0; ol < outL; ol++ {
			gData[ol*L+ol/r] = 1
		}
		return applyGather(x, gData, outL, L, N, C, outL)

	case "linear":
		i0, i1, w := linWeights(L, outL, r, u.AlignCorners)
		gData := make([]float64, outL*L)
		for ol := 0; ol < outL; ol++ {
			gData[ol*L+i0[ol]] += 1 - w[ol]
			gData[ol*L+i1[ol]] += w[ol]
		}
		return applyGather(x, gData, outL, L, N, C, outL)
	}
	panic(fmt.Sprintf("Upsample: mode %q not supported for 3D input (want 'nearest' or 'linear')", u.Mode))
}

// forward4D upsamples (N, C, H, W) with mode "nearest" or "bilinear".
func (u *Upsample) forward4D(x *tensor.Tensor) *tensor.Tensor {
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
		return applyGather(x, gData, rows, cols, N, C, outH, outW)

	case "bilinear":
		// PyTorch-style sampling: with align_corners=false (default), for each
		// output cell (oh, ow) the source coord is ((oh + 0.5)/scale) - 0.5
		// (and same for W); with align_corners=true it is oh*(H-1)/(outH-1).
		// We compute weighted sums over 4 neighbours with clamped indices.
		rows := outH * outW
		cols := H * W
		gData := make([]float64, rows*cols)
		for oh := 0; oh < outH; oh++ {
			y := srcCoord(oh, H, outH, r, u.AlignCorners)
			y0 := int(floorInt(y))
			y1 := y0 + 1
			wy := y - float64(y0)
			// Clamp.
			y0c := clampInt(y0, 0, H-1)
			y1c := clampInt(y1, 0, H-1)
			for ow := 0; ow < outW; ow++ {
				x_ := srcCoord(ow, W, outW, r, u.AlignCorners)
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
		return applyGather(x, gData, rows, cols, N, C, outH, outW)

	case "bicubic":
		// Keys (1981) cubic convolution with a = -0.75 (PyTorch parity).
		// Each output sample reads 4 taps per axis — 16 in 2D — at
		// floor(s)-1 .. floor(s)+2 with border-clamped indices, using the
		// same srcCoord source-coordinate convention as the linear modes.
		rows := outH * outW
		cols := H * W
		yi, wy := cubicWeights(H, outH, r, u.AlignCorners)
		xi, wx := cubicWeights(W, outW, r, u.AlignCorners)
		gData := make([]float64, rows*cols)
		for oh := 0; oh < outH; oh++ {
			for ow := 0; ow < outW; ow++ {
				row := oh*outW + ow
				// 16 contributions: separable products of per-axis weights,
				// accumulated because clamped taps can coincide at borders.
				for a := 0; a < 4; a++ {
					for b := 0; b < 4; b++ {
						gData[row*cols+(yi[oh][a]*W+xi[ow][b])] += wy[oh][a] * wx[ow][b]
					}
				}
			}
		}
		return applyGather(x, gData, rows, cols, N, C, outH, outW)
	}
	panic(fmt.Sprintf("Upsample: mode %q not supported for 4D input (want 'nearest', 'bilinear' or 'bicubic')", u.Mode))
}

// forward5D upsamples (N, C, D, H, W) with mode "nearest" or "trilinear".
func (u *Upsample) forward5D(x *tensor.Tensor) *tensor.Tensor {
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	r := u.ScaleFactor
	outD, outH, outW := D*r, H*r, W*r
	rows := outD * outH * outW
	cols := D * H * W

	switch u.Mode {
	case "nearest":
		gData := make([]float64, rows*cols)
		for od := 0; od < outD; od++ {
			id := od / r
			for oh := 0; oh < outH; oh++ {
				ih := oh / r
				for ow := 0; ow < outW; ow++ {
					iw := ow / r
					gData[((od*outH+oh)*outW+ow)*cols+((id*H+ih)*W+iw)] = 1
				}
			}
		}
		return applyGather(x, gData, rows, cols, N, C, outD, outH, outW)

	case "trilinear":
		d0, d1, wd := linWeights(D, outD, r, u.AlignCorners)
		h0, h1, wh := linWeights(H, outH, r, u.AlignCorners)
		w0, w1, ww := linWeights(W, outW, r, u.AlignCorners)
		gData := make([]float64, rows*cols)
		for od := 0; od < outD; od++ {
			di := [2]int{d0[od], d1[od]}
			dw := [2]float64{1 - wd[od], wd[od]}
			for oh := 0; oh < outH; oh++ {
				hi := [2]int{h0[oh], h1[oh]}
				hw := [2]float64{1 - wh[oh], wh[oh]}
				for ow := 0; ow < outW; ow++ {
					wi := [2]int{w0[ow], w1[ow]}
					wgt := [2]float64{1 - ww[ow], ww[ow]}
					row := (od*outH+oh)*outW + ow
					// 8 corner contributions: product of per-axis weights.
					for a := 0; a < 2; a++ {
						for b := 0; b < 2; b++ {
							for c := 0; c < 2; c++ {
								gData[row*cols+((di[a]*H+hi[b])*W+wi[c])] += dw[a] * hw[b] * wgt[c]
							}
						}
					}
				}
			}
		}
		return applyGather(x, gData, rows, cols, N, C, outD, outH, outW)
	}
	panic(fmt.Sprintf("Upsample: mode %q not supported for 5D input (want 'nearest' or 'trilinear')", u.Mode))
}

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
