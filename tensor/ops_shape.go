package tensor

import "fmt"

// Reshape returns a tensor with the same data and a new shape.
// At most one dim may be -1; it is inferred.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	// Resolve -1
	negIdx := -1
	known := 1
	for i, d := range shape {
		if d == -1 {
			if negIdx >= 0 {
				panic("Reshape: only one -1 allowed")
			}
			negIdx = i
		} else {
			known *= d
		}
	}
	resolved := append([]int(nil), shape...)
	if negIdx >= 0 {
		if len(t.Data)%known != 0 {
			panic(fmt.Sprintf("Reshape: cannot infer -1, numel=%d known=%d", len(t.Data), known))
		}
		resolved[negIdx] = len(t.Data) / known
	}
	if numel(resolved) != len(t.Data) {
		panic(fmt.Sprintf("Reshape: numel mismatch, %d -> %v", len(t.Data), resolved))
	}
	out := &Tensor{Data: t.Data, Shape: resolved, Strides: contiguousStrides(resolved)}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		origShape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Reshape",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := &Tensor{Data: append([]float64(nil), grad.Data...), Shape: origShape, Strides: contiguousStrides(origShape)}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// View is an alias for Reshape (PyTorch parity).
func (t *Tensor) View(shape ...int) *Tensor { return t.Reshape(shape...) }

// Flatten flattens the tensor to 1D.
func (t *Tensor) Flatten() *Tensor { return t.Reshape(len(t.Data)) }

// Transpose swaps the last two dimensions (matrix transpose).
func (t *Tensor) Transpose() *Tensor {
	if len(t.Shape) < 2 {
		panic("Transpose: need at least 2 dims")
	}
	return t.Permute(swapLastTwo(len(t.Shape))...)
}

// T is shorthand for Transpose (2D only).
func (t *Tensor) T() *Tensor { return t.Transpose() }

// Permute reorders dimensions according to dims.
func (t *Tensor) Permute(dims ...int) *Tensor {
	if len(dims) != len(t.Shape) {
		panic("Permute: dims length must match tensor rank")
	}
	newShape := make([]int, len(dims))
	for i, d := range dims {
		newShape[i] = t.Shape[d]
	}
	out := Zeros(newShape...)
	// Copy data with permuted strides.
	srcStrides := contiguousStrides(t.Shape)
	idx := make([]int, len(newShape))
	for k := 0; k < len(out.Data); k++ {
		// Compute source offset.
		off := 0
		for i, ii := range idx {
			off += ii * srcStrides[dims[i]]
		}
		out.Data[k] = t.Data[off]
		for i := len(idx) - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < newShape[i] {
				break
			}
			idx[i] = 0
		}
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		// inverse permutation for backward
		inv := make([]int, len(dims))
		for i, d := range dims {
			inv[d] = i
		}
		out.creator = &Function{
			Name:   "Permute",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				return []*Tensor{grad.Permute(inv...)}
			},
		}
	}
	return out
}

// Squeeze removes size-1 dimensions. If axis < 0, all 1-dims are removed.
func (t *Tensor) Squeeze(axis ...int) *Tensor {
	newShape := []int{}
	if len(axis) == 0 {
		for _, d := range t.Shape {
			if d != 1 {
				newShape = append(newShape, d)
			}
		}
	} else {
		ax := axis[0]
		if ax < 0 {
			ax += len(t.Shape)
		}
		for i, d := range t.Shape {
			if i == ax && d == 1 {
				continue
			}
			newShape = append(newShape, d)
		}
	}
	if len(newShape) == 0 {
		newShape = []int{1}
	}
	return t.Reshape(newShape...)
}

// Unsqueeze inserts a size-1 dim at axis.
func (t *Tensor) Unsqueeze(axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape) + 1
	}
	newShape := make([]int, 0, len(t.Shape)+1)
	newShape = append(newShape, t.Shape[:axis]...)
	newShape = append(newShape, 1)
	newShape = append(newShape, t.Shape[axis:]...)
	return t.Reshape(newShape...)
}

// Expand broadcasts t to the target shape (returns contiguous copy).
func (t *Tensor) Expand(shape ...int) *Tensor {
	out := expandTo(t, shape)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		origShape := append([]int(nil), t.Shape...)
		out.creator = &Function{
			Name:   "Expand",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				return []*Tensor{unbroadcast(grad, origShape)}
			},
		}
	}
	return out
}

// Concat concatenates tensors along axis. No autograd for now.
func Concat(axis int, tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		panic("Concat: empty input")
	}
	if axis < 0 {
		axis += len(tensors[0].Shape)
	}
	totalDim := 0
	for _, x := range tensors {
		totalDim += x.Shape[axis]
	}
	outShape := append([]int(nil), tensors[0].Shape...)
	outShape[axis] = totalDim
	out := Zeros(outShape...)
	// outer iteration
	stride := 1
	for i := axis + 1; i < len(outShape); i++ {
		stride *= outShape[i]
	}
	outer := numel(outShape) / (totalDim * stride)
	for o := 0; o < outer; o++ {
		d := 0
		for _, x := range tensors {
			xDim := x.Shape[axis]
			for k := 0; k < xDim*stride; k++ {
				out.Data[o*totalDim*stride+d*stride+k] = x.Data[o*xDim*stride+k]
			}
			d += xDim
		}
	}
	return out
}

// Stack stacks tensors along a new axis.
func Stack(axis int, tensors ...*Tensor) *Tensor {
	expanded := make([]*Tensor, len(tensors))
	for i, x := range tensors {
		expanded[i] = x.Unsqueeze(axis)
	}
	return Concat(axis, expanded...)
}

func swapLastTwo(rank int) []int {
	dims := make([]int, rank)
	for i := range dims {
		dims[i] = i
	}
	dims[rank-1], dims[rank-2] = dims[rank-2], dims[rank-1]
	return dims
}
