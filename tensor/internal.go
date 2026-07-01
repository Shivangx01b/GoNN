package tensor

import "fmt"

// Shared internal helpers: consistent panics, axis normalization, dtype
// finishing, and shape/stride arithmetic used across the op files.

// opError panics with a consistent "tensor.<Op>: <message>" prefix. All op
// precondition failures route through this so error text is uniform.
func opError(op, format string, args ...interface{}) {
	panic(fmt.Sprintf("tensor.%s: %s", op, fmt.Sprintf(format, args...)))
}

// normalizeAxis resolves a possibly-negative axis for a tensor of the given
// rank and validates the result. Unlike the historical inline
// `if axis < 0 { axis += rank }` snippets, this also rejects out-of-range
// axes instead of silently corrupting downstream index math.
func normalizeAxis(op string, axis, rank int) int {
	orig := axis
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		opError(op, "axis %d out of range for rank %d", orig, rank)
	}
	return axis
}

// finishOp stamps the result dtype on out and rounds its values to that
// dtype's precision. Every op that produces a new tensor from typed inputs
// funnels through this instead of duplicating the promote/cast pair.
func finishOp(out *Tensor, dt DType) {
	out.Dtype = dt
	castInPlace(out)
}

// numel returns the product of dims (1 for a 0-d shape).
func numel(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// contiguousStrides returns row-major strides for shape.
func contiguousStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	s := make([]int, len(shape))
	s[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		s[i] = s[i+1] * shape[i+1]
	}
	return s
}

// shapesEqual reports whether two shapes are identical.
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// axisStrideOuter returns (dim, stride, outer) for iterating a tensor along
// axis: data index = o*dim*stride + d*stride + s with o in [0,outer),
// d in [0,dim), s in [0,stride). Shared by the reduce, index, and slice ops.
func axisStrideOuter(shape []int, axis int) (dim, stride, outer int) {
	dim = shape[axis]
	stride = 1
	for i := axis + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	outer = numel(shape) / (dim * stride)
	return
}
