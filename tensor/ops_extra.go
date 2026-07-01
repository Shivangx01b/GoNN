package tensor

// triMask builds a 0/1 mask over the last two dims of shape. keepLower==true
// keeps entries on/below the k-th diagonal (Tril); otherwise on/above (Triu).
// An element at (row,col) of the last 2 dims is kept when, for Tril,
// col <= row+k, and for Triu, col >= row+k.
func triMask(shape []int, k int, keepLower bool) []float64 {
	if len(shape) < 2 {
		opError("Tril/Triu", "need at least 2 dims, got shape %v", shape)
	}
	rows := shape[len(shape)-2]
	cols := shape[len(shape)-1]
	n := numel(shape)
	mask := make([]float64, n)
	mat := rows * cols
	for base := 0; base < n; base += mat {
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				keep := false
				if keepLower {
					keep = c <= r+k
				} else {
					keep = c >= r+k
				}
				if keep {
					mask[base+r*cols+c] = 1
				}
			}
		}
	}
	return mask
}

// applyMask returns t * mask (elementwise) with autograd routed through mask.
func applyMaskedKeep(t *Tensor, mask []float64, name string) *Tensor {
	out := Zeros(t.Shape...)
	for i := range out.Data {
		out.Data[i] = t.Data[i] * mask[i]
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		out.creator = &Function{
			Name:   name,
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(grad.Shape...)
				for i := range g.Data {
					g.Data[i] = grad.Data[i] * mask[i]
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Tril zeros out everything above the k-th diagonal of the last two dims.
// Differentiable (grad masked the same way).
func (t *Tensor) Tril(k int) *Tensor {
	return applyMaskedKeep(t, triMask(t.Shape, k, true), "Tril")
}

// Triu zeros out everything below the k-th diagonal of the last two dims.
// Differentiable (grad masked the same way).
func (t *Tensor) Triu(k int) *Tensor {
	return applyMaskedKeep(t, triMask(t.Shape, k, false), "Triu")
}

// Where selects elementwise: out = cond!=0 ? a : b. cond, a, b must share shape
// (no broadcasting). Differentiable wrt a and b (grad routed by the mask); cond
// is treated as a constant.
func Where(cond, a, b *Tensor) *Tensor {
	if !shapesEqual(a.Shape, b.Shape) || !shapesEqual(cond.Shape, a.Shape) {
		opError("Where", "shape mismatch cond=%v a=%v b=%v", cond.Shape, a.Shape, b.Shape)
	}
	out := Zeros(a.Shape...)
	for i := range out.Data {
		if cond.Data[i] != 0 {
			out.Data[i] = a.Data[i]
		} else {
			out.Data[i] = b.Data[i]
		}
	}
	if a.RequiresGrad || b.RequiresGrad || a.creator != nil || b.creator != nil {
		out.RequiresGrad = true
		condData := append([]float64(nil), cond.Data...)
		out.creator = &Function{
			Name:   "Where",
			Inputs: []*Tensor{a, b},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				ga := Zeros(grad.Shape...)
				gb := Zeros(grad.Shape...)
				for i := range grad.Data {
					if condData[i] != 0 {
						ga.Data[i] = grad.Data[i]
					} else {
						gb.Data[i] = grad.Data[i]
					}
				}
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

// MaskedFill returns a copy of t with elements set to value wherever mask!=0.
// mask must share t's shape. Differentiable wrt t (grad zeroed where filled).
func (t *Tensor) MaskedFill(mask *Tensor, value float64) *Tensor {
	if !shapesEqual(mask.Shape, t.Shape) {
		opError("MaskedFill", "mask shape %v != tensor shape %v", mask.Shape, t.Shape)
	}
	out := Zeros(t.Shape...)
	for i := range out.Data {
		if mask.Data[i] != 0 {
			out.Data[i] = value
		} else {
			out.Data[i] = t.Data[i]
		}
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		maskData := append([]float64(nil), mask.Data...)
		out.creator = &Function{
			Name:   "MaskedFill",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(grad.Shape...)
				for i := range g.Data {
					if maskData[i] == 0 {
						g.Data[i] = grad.Data[i]
					}
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Cumsum returns the cumulative sum along axis. Differentiable; the backward of
// a forward cumulative sum is a reverse cumulative sum of the gradient.
func (t *Tensor) Cumsum(axis int) *Tensor {
	axis = normalizeAxis("Cumsum", axis, len(t.Shape))
	dim, stride, outer := axisStrideOuter(t.Shape, axis)
	out := Zeros(t.Shape...)
	for o := 0; o < outer; o++ {
		for s := 0; s < stride; s++ {
			var acc float64
			for d := 0; d < dim; d++ {
				acc += t.Data[o*dim*stride+d*stride+s]
				out.Data[o*dim*stride+d*stride+s] = acc
			}
		}
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		out.creator = &Function{
			Name:   "Cumsum",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(grad.Shape...)
				for o := 0; o < outer; o++ {
					for s := 0; s < stride; s++ {
						var acc float64
						for d := dim - 1; d >= 0; d-- {
							acc += grad.Data[o*dim*stride+d*stride+s]
							g.Data[o*dim*stride+d*stride+s] = acc
						}
					}
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Flip reverses t along each of the given axes. Differentiable (the gradient is
// flipped back along the same axes).
func (t *Tensor) Flip(axes ...int) *Tensor {
	flip := make([]bool, len(t.Shape))
	for _, a := range axes {
		flip[normalizeAxis("Flip", a, len(t.Shape))] = true
	}
	out := flipData(t, flip)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		out.creator = &Function{
			Name:   "Flip",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				return []*Tensor{flipData(grad, flip)}
			},
		}
	}
	return out
}

// flipData reverses data along every axis marked true in flip (no autograd).
func flipData(t *Tensor, flip []bool) *Tensor {
	out := Zeros(t.Shape...)
	strides := contiguousStrides(t.Shape)
	idx := make([]int, len(t.Shape))
	for k := range out.Data {
		src := 0
		for d := range idx {
			coord := idx[d]
			if flip[d] {
				coord = t.Shape[d] - 1 - coord
			}
			src += coord * strides[d]
		}
		out.Data[k] = t.Data[src]
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < t.Shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

// Repeat tiles the tensor reps[i] times along dimension i (PyTorch Tensor.repeat
// semantics). len(reps) must equal the tensor rank. Differentiable (the gradient
// sums over the tiled copies). For NumPy-style tiling use Tile.
func (t *Tensor) Repeat(reps ...int) *Tensor {
	if len(reps) != len(t.Shape) {
		opError("Repeat", "reps len %d != tensor rank %d", len(reps), len(t.Shape))
	}
	outShape := make([]int, len(t.Shape))
	for i := range t.Shape {
		if reps[i] < 1 {
			opError("Repeat", "reps must be >= 1, got %d", reps[i])
		}
		outShape[i] = t.Shape[i] * reps[i]
	}
	out := Zeros(outShape...)
	inStrides := contiguousStrides(t.Shape)
	idx := make([]int, len(outShape))
	for k := range out.Data {
		src := 0
		for d := range idx {
			src += (idx[d] % t.Shape[d]) * inStrides[d]
		}
		out.Data[k] = t.Data[src]
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < outShape[d] {
				break
			}
			idx[d] = 0
		}
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		inShape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Repeat",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(inShape...)
				gStrides := contiguousStrides(inShape)
				oi := make([]int, len(outShape))
				for k := range grad.Data {
					dst := 0
					for d := range oi {
						dst += (oi[d] % inShape[d]) * gStrides[d]
					}
					g.Data[dst] += grad.Data[k]
					for d := len(oi) - 1; d >= 0; d-- {
						oi[d]++
						if oi[d] < outShape[d] {
							break
						}
						oi[d] = 0
					}
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Tile is an alias for Repeat (right-aligned reps not supported; len(reps) must
// equal rank). Differentiable.
func (t *Tensor) Tile(reps ...int) *Tensor { return t.Repeat(reps...) }

// Split splits t along axis into chunks of the given size (the final chunk may
// be smaller if axis is not divisible by size). Each returned piece is
// differentiable and routes its gradient back into the corresponding slice of t.
func (t *Tensor) Split(axis, size int) []*Tensor {
	axis = normalizeAxis("Split", axis, len(t.Shape))
	if size < 1 {
		opError("Split", "size must be >= 1, got %d", size)
	}
	dim := t.Shape[axis]
	var sizes []int
	for off := 0; off < dim; off += size {
		s := size
		if off+s > dim {
			s = dim - off
		}
		sizes = append(sizes, s)
	}
	return sliceAlong(t, axis, sizes)
}

// Chunk splits t along axis into n chunks as evenly as possible (PyTorch
// semantics: the first chunks get ceil(dim/n); the remainder may be smaller or
// the number of chunks fewer than n). Differentiable, like Split.
func (t *Tensor) Chunk(axis, n int) []*Tensor {
	axis = normalizeAxis("Chunk", axis, len(t.Shape))
	if n < 1 {
		opError("Chunk", "n must be >= 1, got %d", n)
	}
	dim := t.Shape[axis]
	size := (dim + n - 1) / n // ceil
	return t.Split(axis, size)
}

// sliceAlong cuts t along axis into contiguous pieces of the given sizes,
// each autograd-aware (scatter its grad back into t's region).
func sliceAlong(t *Tensor, axis int, sizes []int) []*Tensor {
	dim, stride, outer := axisStrideOuter(t.Shape, axis)
	pieces := make([]*Tensor, len(sizes))
	offset := 0
	for pi, sz := range sizes {
		pShape := append([]int(nil), t.Shape...)
		pShape[axis] = sz
		p := Zeros(pShape...)
		for o := 0; o < outer; o++ {
			for d := 0; d < sz; d++ {
				for s := 0; s < stride; s++ {
					p.Data[o*sz*stride+d*stride+s] = t.Data[o*dim*stride+(offset+d)*stride+s]
				}
			}
		}
		if t.RequiresGrad || t.creator != nil {
			p.RequiresGrad = true
			inShape := append([]int(nil), t.Shape...)
			off := offset
			pieceSz := sz
			p.creator = &Function{
				Name:   "Split",
				Inputs: []*Tensor{t},
				Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
					g := Zeros(inShape...)
					gDim, gStride, gOuter := axisStrideOuter(inShape, axis)
					for o := 0; o < gOuter; o++ {
						for d := 0; d < pieceSz; d++ {
							for s := 0; s < gStride; s++ {
								g.Data[o*gDim*gStride+(off+d)*gStride+s] = grad.Data[o*pieceSz*gStride+d*gStride+s]
							}
						}
					}
					return []*Tensor{g}
				},
			}
		}
		pieces[pi] = p
		offset += sz
	}
	return pieces
}
