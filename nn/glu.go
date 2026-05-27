package nn

import (
	"fmt"

	"gonn/tensor"
)

// GLU is the gated linear unit: split input into halves a, b along Dim and
// return a * sigmoid(b). The size of x along Dim must be even.
type GLU struct {
	Dim int
}

// Forward implements GLU(x) = a * sigmoid(b).
func (g GLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	dim := g.Dim
	if dim < 0 {
		dim += len(x.Shape)
	}
	if dim < 0 || dim >= len(x.Shape) {
		panic(fmt.Sprintf("GLU: dim %d out of range for shape %v", g.Dim, x.Shape))
	}
	if x.Shape[dim]%2 != 0 {
		panic(fmt.Sprintf("GLU: size along dim %d must be even, got %d", dim, x.Shape[dim]))
	}
	half := x.Shape[dim] / 2
	a := sliceAxis(x, dim, 0, half)
	b := sliceAxis(x, dim, half, x.Shape[dim])
	return a.Mul(b.Sigmoid())
}

// Parameters returns nothing (GLU has no learnable params).
func (GLU) Parameters() []*tensor.Tensor { return nil }

// sliceAxis returns t[..., start:end, ...] along the given axis with autograd
// preserved. It is implemented via permute + matmul with a 0/1 selector matrix
// so that gradients flow through the built-in autograd machinery.
func sliceAxis(t *tensor.Tensor, axis, start, end int) *tensor.Tensor {
	if start < 0 || end > t.Shape[axis] || start >= end {
		panic(fmt.Sprintf("sliceAxis: invalid range [%d,%d) for dim of size %d", start, end, t.Shape[axis]))
	}
	dimIn := t.Shape[axis]
	dimOut := end - start

	// Compute outer / inner extents around the slicing axis.
	inner := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		inner *= t.Shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= t.Shape[i]
	}

	// Permute axis to the end via reshape+permute so we can do a 2D matmul.
	// View t as (outer, dimIn, inner) -> permute to (outer, inner, dimIn).
	t3 := t.Reshape(outer, dimIn, inner)
	perm := t3.Permute(0, 2, 1)              // (outer, inner, dimIn)
	flat := perm.Reshape(outer*inner, dimIn) // (outer*inner, dimIn)

	// Selector matrix S of shape (dimIn, dimOut), S[i,j] = 1 if i == start+j.
	sel := tensor.Zeros(dimIn, dimOut)
	for j := 0; j < dimOut; j++ {
		sel.Data[(start+j)*dimOut+j] = 1
	}
	picked := flat.MatMul(sel) // (outer*inner, dimOut)

	// Unflatten and undo the permutation.
	pickedR := picked.Reshape(outer, inner, dimOut).Permute(0, 2, 1) // (outer, dimOut, inner)

	outShape := append([]int(nil), t.Shape...)
	outShape[axis] = dimOut
	return pickedR.Reshape(outShape...)
}
