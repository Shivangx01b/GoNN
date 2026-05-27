package tensor

import "math"

// Sum returns the sum of all elements as a scalar tensor.
func (t *Tensor) Sum() *Tensor {
	var s float64
	for _, v := range t.Data {
		s += v
	}
	out := New([]float64{s}, 1)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		shape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Sum",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(shape...)
				v := grad.Data[0]
				for i := range g.Data {
					g.Data[i] = v
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Mean returns the mean of all elements as a scalar tensor.
func (t *Tensor) Mean() *Tensor {
	n := float64(len(t.Data))
	return t.Sum().DivScalar(n)
}

// Max returns the max of all elements as a scalar tensor.
func (t *Tensor) Max() *Tensor {
	if len(t.Data) == 0 {
		panic("Max: empty tensor")
	}
	m := t.Data[0]
	idx := 0
	for i, v := range t.Data {
		if v > m {
			m = v
			idx = i
		}
	}
	out := New([]float64{m}, 1)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		shape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Max",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(shape...)
				g.Data[idx] = grad.Data[0]
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Min returns the min of all elements as a scalar tensor.
func (t *Tensor) Min() *Tensor {
	if len(t.Data) == 0 {
		panic("Min: empty tensor")
	}
	m := t.Data[0]
	idx := 0
	for i, v := range t.Data {
		if v < m {
			m = v
			idx = i
		}
	}
	out := New([]float64{m}, 1)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		shape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Min",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(shape...)
				g.Data[idx] = grad.Data[0]
				return []*Tensor{g}
			},
		}
	}
	return out
}

// SumAxis returns the sum along axis. If keepDim, the reduced axis stays as size 1.
func (t *Tensor) SumAxis(axis int, keepDim bool) *Tensor {
	return sumAxis(t, axis, keepDim)
}

// MeanAxis returns the mean along axis.
func (t *Tensor) MeanAxis(axis int, keepDim bool) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	n := float64(t.Shape[axis])
	return sumAxis(t, axis, keepDim).DivScalar(n)
}

// MaxAxis returns the max along axis.
func (t *Tensor) MaxAxis(axis int, keepDim bool) *Tensor {
	return reduceAxis(t, axis, keepDim, "Max")
}

// MinAxis returns the min along axis.
func (t *Tensor) MinAxis(axis int, keepDim bool) *Tensor {
	return reduceAxis(t, axis, keepDim, "Min")
}

// ArgMax returns indices of max along axis (no grad).
func (t *Tensor) ArgMax(axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	outShape := append([]int(nil), t.Shape...)
	outShape = append(outShape[:axis], outShape[axis+1:]...)
	out := Zeros(outShape...)
	dim := t.Shape[axis]
	stride := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}
	outer := numel(t.Shape) / (dim * stride)
	outIdx := 0
	for o := 0; o < outer; o++ {
		for s := 0; s < stride; s++ {
			best := math.Inf(-1)
			bestIdx := 0
			for d := 0; d < dim; d++ {
				v := t.Data[o*dim*stride+d*stride+s]
				if v > best {
					best = v
					bestIdx = d
				}
			}
			out.Data[outIdx] = float64(bestIdx)
			outIdx++
		}
	}
	return out
}

// ArgMin returns indices of min along axis (no grad).
func (t *Tensor) ArgMin(axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	outShape := append([]int(nil), t.Shape...)
	outShape = append(outShape[:axis], outShape[axis+1:]...)
	out := Zeros(outShape...)
	dim := t.Shape[axis]
	stride := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}
	outer := numel(t.Shape) / (dim * stride)
	outIdx := 0
	for o := 0; o < outer; o++ {
		for s := 0; s < stride; s++ {
			best := math.Inf(1)
			bestIdx := 0
			for d := 0; d < dim; d++ {
				v := t.Data[o*dim*stride+d*stride+s]
				if v < best {
					best = v
					bestIdx = d
				}
			}
			out.Data[outIdx] = float64(bestIdx)
			outIdx++
		}
	}
	return out
}

// sumAxis reduces along one axis.
func sumAxis(t *Tensor, axis int, keepDim bool) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	dim := t.Shape[axis]
	stride := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}
	outer := numel(t.Shape) / (dim * stride)
	outShape := append([]int(nil), t.Shape...)
	if keepDim {
		outShape[axis] = 1
	} else {
		outShape = append(outShape[:axis], outShape[axis+1:]...)
	}
	out := Zeros(outShape...)
	outIdx := 0
	for o := 0; o < outer; o++ {
		for s := 0; s < stride; s++ {
			var sum float64
			for d := 0; d < dim; d++ {
				sum += t.Data[o*dim*stride+d*stride+s]
			}
			out.Data[outIdx] = sum
			outIdx++
		}
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		inShape := append([]int(nil), t.Shape...)
		ax := axis
		out.creator = &Function{
			Name:   "SumAxis",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(inShape...)
				// broadcast grad along reduced axis
				gDim := inShape[ax]
				gStride := 1
				for i := ax + 1; i < len(inShape); i++ {
					gStride *= inShape[i]
				}
				gOuter := numel(inShape) / (gDim * gStride)
				idx := 0
				for o := 0; o < gOuter; o++ {
					for s := 0; s < gStride; s++ {
						v := grad.Data[idx]
						for d := 0; d < gDim; d++ {
							g.Data[o*gDim*gStride+d*gStride+s] = v
						}
						idx++
					}
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// reduceAxis reduces along one axis with max/min, including correct backward.
func reduceAxis(t *Tensor, axis int, keepDim bool, op string) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	dim := t.Shape[axis]
	stride := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}
	outer := numel(t.Shape) / (dim * stride)
	outShape := append([]int(nil), t.Shape...)
	if keepDim {
		outShape[axis] = 1
	} else {
		outShape = append(outShape[:axis], outShape[axis+1:]...)
	}
	out := Zeros(outShape...)
	bestIdx := make([]int, len(out.Data))
	outIdx := 0
	for o := 0; o < outer; o++ {
		for s := 0; s < stride; s++ {
			var best float64
			var bidx int
			if op == "Max" {
				best = math.Inf(-1)
			} else {
				best = math.Inf(1)
			}
			for d := 0; d < dim; d++ {
				v := t.Data[o*dim*stride+d*stride+s]
				if (op == "Max" && v > best) || (op == "Min" && v < best) {
					best = v
					bidx = d
				}
			}
			out.Data[outIdx] = best
			bestIdx[outIdx] = o*dim*stride + bidx*stride + s
			outIdx++
		}
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		inShape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   op + "Axis",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(inShape...)
				for i, idx := range bestIdx {
					g.Data[idx] += grad.Data[i]
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}
