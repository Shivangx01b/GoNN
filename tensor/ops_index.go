package tensor

// Gather gathers values along axis using integer indices stored as float64 in
// index. index must have the same rank as t, and the same shape on every axis
// except (optionally) axis. out[i...] = t[..., index[i...], ...] where the
// indexed coordinate is along axis. Differentiable wrt t (scatter-add grad).
func (t *Tensor) Gather(axis int, index *Tensor) *Tensor {
	axis = normalizeAxis("Gather", axis, len(t.Shape))
	if len(index.Shape) != len(t.Shape) {
		opError("Gather", "index rank %d != tensor rank %d", len(index.Shape), len(t.Shape))
	}
	for d := range t.Shape {
		if d != axis && index.Shape[d] != t.Shape[d] {
			opError("Gather", "index shape %v incompatible with %v on axis %d", index.Shape, t.Shape, d)
		}
	}
	outShape := append([]int(nil), index.Shape...)
	out := Zeros(outShape...)

	tDim := t.Shape[axis]
	_, tStride, _ := axisStrideOuter(t.Shape, axis)
	oDim, oStride, oOuter := axisStrideOuter(outShape, axis)

	// Map a flat index in `out` to the corresponding flat index in `t` for a
	// given gathered coordinate along axis.
	srcIndex := make([]int, len(out.Data)) // saved for backward
	for o := 0; o < oOuter; o++ {
		for d := 0; d < oDim; d++ {
			for s := 0; s < oStride; s++ {
				flat := o*oDim*oStride + d*oStride + s
				gi := int(index.Data[flat])
				if gi < 0 {
					gi += tDim
				}
				if gi < 0 || gi >= tDim {
					opError("Gather", "index %d out of range [0,%d)", gi, tDim)
				}
				src := o*tDim*tStride + gi*tStride + s
				out.Data[flat] = t.Data[src]
				srcIndex[flat] = src
			}
		}
	}

	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		inShape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Gather",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(inShape...)
				for flat, src := range srcIndex {
					g.Data[src] += grad.Data[flat]
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// IndexSelect selects slices along axis at the integer positions given in index
// (a 1-D tensor of float64-encoded indices). The output has the same shape as t
// except that axis becomes len(index). Differentiable wrt t (scatter-add grad).
func (t *Tensor) IndexSelect(axis int, index *Tensor) *Tensor {
	axis = normalizeAxis("IndexSelect", axis, len(t.Shape))
	idx := make([]int, len(index.Data))
	for i, v := range index.Data {
		j := int(v)
		if j < 0 {
			j += t.Shape[axis]
		}
		if j < 0 || j >= t.Shape[axis] {
			opError("IndexSelect", "index %d out of range [0,%d)", j, t.Shape[axis])
		}
		idx[i] = j
	}
	outShape := append([]int(nil), t.Shape...)
	outShape[axis] = len(idx)
	out := Zeros(outShape...)

	tDim, tStride, outer := axisStrideOuter(t.Shape, axis)
	for o := 0; o < outer; o++ {
		for d, src := range idx {
			for s := 0; s < tStride; s++ {
				out.Data[o*len(idx)*tStride+d*tStride+s] = t.Data[o*tDim*tStride+src*tStride+s]
			}
		}
	}

	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		inShape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "IndexSelect",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(inShape...)
				gDim, gStride, gOuter := axisStrideOuter(inShape, axis)
				for o := 0; o < gOuter; o++ {
					for d, src := range idx {
						for s := 0; s < gStride; s++ {
							g.Data[o*gDim*gStride+src*gStride+s] += grad.Data[o*len(idx)*gStride+d*gStride+s]
						}
					}
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// ArgWhere returns the flat indices (as float64) of all non-zero elements of t.
// Non-differentiable helper (integer output).
func (t *Tensor) ArgWhere() *Tensor {
	var idx []float64
	for i, v := range t.Data {
		if v != 0 {
			idx = append(idx, float64(i))
		}
	}
	return New(idx, len(idx))
}
