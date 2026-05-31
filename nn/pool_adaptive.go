package nn

import "gonn/tensor"

// startEnd returns the start (inclusive) and end (exclusive) indices for the
// i-th adaptive pool window over an input of size `in` producing `out` cells.
//   start = floor(i*in/out)
//   end   = ceil((i+1)*in/out)
func startEnd(i, in, out int) (int, int) {
	s := (i * in) / out
	// ceil((i+1)*in/out)
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
	return s, e
}

// AdaptiveAvgPool1d applies adaptive average pooling on (N, C, L) -> (N, C, OutSize).
type AdaptiveAvgPool1d struct{ OutSize int }

// NewAdaptiveAvgPool1d constructs the layer.
func NewAdaptiveAvgPool1d(outSize int) *AdaptiveAvgPool1d {
	return &AdaptiveAvgPool1d{OutSize: outSize}
}

// Forward computes per-cell mean over the dynamically sized window.
func (p *AdaptiveAvgPool1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("AdaptiveAvgPool1d: expected 3D input (N,C,L)")
	}
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	return adaptivePool1D(x, N, C, L, p.OutSize, false)
}

// Parameters returns nothing.
func (p *AdaptiveAvgPool1d) Parameters() []*tensor.Tensor { return nil }

// AdaptiveMaxPool1d applies adaptive max pooling on (N, C, L).
type AdaptiveMaxPool1d struct{ OutSize int }

// NewAdaptiveMaxPool1d constructs the layer.
func NewAdaptiveMaxPool1d(outSize int) *AdaptiveMaxPool1d {
	return &AdaptiveMaxPool1d{OutSize: outSize}
}

// Forward computes per-cell max over the dynamically sized window.
func (p *AdaptiveMaxPool1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("AdaptiveMaxPool1d: expected 3D input (N,C,L)")
	}
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	return adaptivePool1D(x, N, C, L, p.OutSize, true)
}

// Parameters returns nothing.
func (p *AdaptiveMaxPool1d) Parameters() []*tensor.Tensor { return nil }

// adaptivePool1D splits each output cell's window into a separate pass to
// support varying window sizes. We average/max each window into a (N*C, 1)
// tensor then concatenate. Built on existing autograd-aware ops.
func adaptivePool1D(x *tensor.Tensor, N, C, L, outL int, isMax bool) *tensor.Tensor {
	// Reshape to (N*C, L).
	xFlat := x.Reshape(N*C, L)
	// Build, per output cell, a gather column selecting [s:e] of length L.
	// We compute one output cell at a time and stack along a new axis.
	outs := make([]*tensor.Tensor, outL)
	for i := 0; i < outL; i++ {
		s, e := startEnd(i, L, outL)
		win := e - s
		if win <= 0 {
			// degenerate: just use position s
			s = i * L / outL
			if s >= L {
				s = L - 1
			}
			e = s + 1
			win = 1
		}
		// Gather matrix of shape (win, L): G[r, s+r] = 1.
		gData := make([]float64, win*L)
		for r := 0; r < win; r++ {
			gData[r*L+(s+r)] = 1
		}
		g := tensor.New(gData, win, L)
		// (N*C, win) = xFlat @ G^T
		w := xFlat.MatMul(g.Transpose())
		var v *tensor.Tensor
		if isMax {
			v = w.MaxAxis(1, true) // (N*C, 1)
		} else {
			v = w.MeanAxis(1, true)
		}
		outs[i] = v
	}
	// Concatenate the per-cell results (each (N*C, 1)) along axis 1 -> (N*C, outL).
	// tensor.Concat is autograd-aware, so gradients flow back to each window.
	cat := tensor.Concat(1, outs...)
	return cat.Reshape(N, C, outL)
}

// AdaptiveAvgPool2d applies adaptive average pooling on (N, C, H, W) -> (N, C, OutH, OutW).
type AdaptiveAvgPool2d struct{ OutH, OutW int }

// NewAdaptiveAvgPool2d constructs the layer.
func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d {
	return &AdaptiveAvgPool2d{OutH: outH, OutW: outW}
}

// Forward computes per-cell mean over each 2D adaptive window.
func (p *AdaptiveAvgPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("AdaptiveAvgPool2d: expected 4D input (N,C,H,W)")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	return adaptivePool2D(x, N, C, H, W, p.OutH, p.OutW, false)
}

// Parameters returns nothing.
func (p *AdaptiveAvgPool2d) Parameters() []*tensor.Tensor { return nil }

// AdaptiveMaxPool2d applies adaptive max pooling on (N, C, H, W).
type AdaptiveMaxPool2d struct{ OutH, OutW int }

// NewAdaptiveMaxPool2d constructs the layer.
func NewAdaptiveMaxPool2d(outH, outW int) *AdaptiveMaxPool2d {
	return &AdaptiveMaxPool2d{OutH: outH, OutW: outW}
}

// Forward computes per-cell max over each 2D adaptive window.
func (p *AdaptiveMaxPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("AdaptiveMaxPool2d: expected 4D input (N,C,H,W)")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	return adaptivePool2D(x, N, C, H, W, p.OutH, p.OutW, true)
}

// Parameters returns nothing.
func (p *AdaptiveMaxPool2d) Parameters() []*tensor.Tensor { return nil }

func adaptivePool2D(x *tensor.Tensor, N, C, H, W, outH, outW int, isMax bool) *tensor.Tensor {
	// Reshape to (N*C, H*W) for gather. Per (oh, ow) cell, build a gather
	// matrix that selects the window of size winH*winW from H*W.
	xFlat := x.Reshape(N*C, H*W)
	var acc *tensor.Tensor
	for oh := 0; oh < outH; oh++ {
		sH, eH := startEnd(oh, H, outH)
		winH := eH - sH
		if winH <= 0 {
			sH = oh * H / outH
			if sH >= H {
				sH = H - 1
			}
			winH = 1
		}
		for ow := 0; ow < outW; ow++ {
			sW, eW := startEnd(ow, W, outW)
			winW := eW - sW
			if winW <= 0 {
				sW = ow * W / outW
				if sW >= W {
					sW = W - 1
				}
				winW = 1
			}
			// Build (winH*winW, H*W) gather mat for this window.
			win := winH * winW
			gData := make([]float64, win*H*W)
			r := 0
			for hi := sH; hi < sH+winH; hi++ {
				for wi := sW; wi < sW+winW; wi++ {
					gData[r*H*W+(hi*W+wi)] = 1
					r++
				}
			}
			g := tensor.New(gData, win, H*W)
			w := xFlat.MatMul(g.Transpose()) // (N*C, win)
			var v *tensor.Tensor
			if isMax {
				v = w.MaxAxis(1, true) // (N*C, 1)
			} else {
				v = w.MeanAxis(1, true)
			}
			// Place into (N*C, outH*outW) at column oh*outW+ow using indicator.
			ind := make([]float64, outH*outW)
			ind[oh*outW+ow] = 1
			e := tensor.New(ind, 1, outH*outW)
			piece := v.Mul(e)
			if acc == nil {
				acc = piece
			} else {
				acc = acc.Add(piece)
			}
		}
	}
	return acc.Reshape(N, C, outH, outW)
}

// AdaptiveAvgPool3d applies adaptive average pooling on (N, C, D, H, W).
type AdaptiveAvgPool3d struct{ OutD, OutH, OutW int }

// NewAdaptiveAvgPool3d constructs the layer.
func NewAdaptiveAvgPool3d(outD, outH, outW int) *AdaptiveAvgPool3d {
	return &AdaptiveAvgPool3d{OutD: outD, OutH: outH, OutW: outW}
}

// Forward computes per-cell mean over each 3D adaptive window.
func (p *AdaptiveAvgPool3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic("AdaptiveAvgPool3d: expected 5D input (N,C,D,H,W)")
	}
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	return adaptivePool3D(x, N, C, D, H, W, p.OutD, p.OutH, p.OutW, false)
}

// Parameters returns nothing.
func (p *AdaptiveAvgPool3d) Parameters() []*tensor.Tensor { return nil }

// AdaptiveMaxPool3d applies adaptive max pooling on (N, C, D, H, W).
type AdaptiveMaxPool3d struct{ OutD, OutH, OutW int }

// NewAdaptiveMaxPool3d constructs the layer.
func NewAdaptiveMaxPool3d(outD, outH, outW int) *AdaptiveMaxPool3d {
	return &AdaptiveMaxPool3d{OutD: outD, OutH: outH, OutW: outW}
}

// Forward computes per-cell max over each 3D adaptive window.
func (p *AdaptiveMaxPool3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic("AdaptiveMaxPool3d: expected 5D input (N,C,D,H,W)")
	}
	N, C, D, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]
	return adaptivePool3D(x, N, C, D, H, W, p.OutD, p.OutH, p.OutW, true)
}

// Parameters returns nothing.
func (p *AdaptiveMaxPool3d) Parameters() []*tensor.Tensor { return nil }

func adaptivePool3D(x *tensor.Tensor, N, C, D, H, W, outD, outH, outW int, isMax bool) *tensor.Tensor {
	xFlat := x.Reshape(N*C, D*H*W)
	var acc *tensor.Tensor
	outTotal := outD * outH * outW
	for od := 0; od < outD; od++ {
		sD, eD := startEnd(od, D, outD)
		winD := eD - sD
		if winD <= 0 {
			sD = od * D / outD
			if sD >= D {
				sD = D - 1
			}
			winD = 1
		}
		for oh := 0; oh < outH; oh++ {
			sH, eH := startEnd(oh, H, outH)
			winH := eH - sH
			if winH <= 0 {
				sH = oh * H / outH
				if sH >= H {
					sH = H - 1
				}
				winH = 1
			}
			for ow := 0; ow < outW; ow++ {
				sW, eW := startEnd(ow, W, outW)
				winW := eW - sW
				if winW <= 0 {
					sW = ow * W / outW
					if sW >= W {
						sW = W - 1
					}
					winW = 1
				}
				win := winD * winH * winW
				gData := make([]float64, win*D*H*W)
				r := 0
				for di := sD; di < sD+winD; di++ {
					for hi := sH; hi < sH+winH; hi++ {
						for wi := sW; wi < sW+winW; wi++ {
							gData[r*D*H*W+((di*H+hi)*W+wi)] = 1
							r++
						}
					}
				}
				g := tensor.New(gData, win, D*H*W)
				w := xFlat.MatMul(g.Transpose()) // (N*C, win)
				var v *tensor.Tensor
				if isMax {
					v = w.MaxAxis(1, true)
				} else {
					v = w.MeanAxis(1, true)
				}
				ind := make([]float64, outTotal)
				ind[(od*outH+oh)*outW+ow] = 1
				e := tensor.New(ind, 1, outTotal)
				piece := v.Mul(e)
				if acc == nil {
					acc = piece
				} else {
					acc = acc.Add(piece)
				}
			}
		}
	}
	return acc.Reshape(N, C, outD, outH, outW)
}
