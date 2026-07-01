package tensor

// broadcastShapes returns the broadcasted shape of a and b, NumPy-style.
func broadcastShapes(a, b []int) []int {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if i < len(a) {
			da = a[len(a)-1-i]
		}
		if i < len(b) {
			db = b[len(b)-1-i]
		}
		switch {
		case da == db:
			out[n-1-i] = da
		case da == 1:
			out[n-1-i] = db
		case db == 1:
			out[n-1-i] = da
		default:
			opError("broadcast", "incompatible shapes %v vs %v", a, b)
		}
	}
	return out
}

// expandTo returns a new tensor (contiguous) broadcast to target shape.
// Cheap implementation: enumerate target indices and pull from source via
// broadcasting offset arithmetic.
func expandTo(t *Tensor, target []int) *Tensor {
	if shapesEqual(t.Shape, target) {
		return t
	}
	out := Zeros(target...)
	// pad source shape on the left with 1s
	pad := len(target) - len(t.Shape)
	srcShape := make([]int, len(target))
	for i := range srcShape {
		srcShape[i] = 1
	}
	for i, d := range t.Shape {
		srcShape[i+pad] = d
	}
	// Validate compatibility.
	for i := range target {
		if srcShape[i] != target[i] && srcShape[i] != 1 {
			opError("expandTo", "cannot broadcast %v to %v", t.Shape, target)
		}
	}
	srcStrides := contiguousStrides(t.Shape)
	// pad source strides
	paddedStrides := make([]int, len(target))
	for i := range paddedStrides {
		if i < pad {
			paddedStrides[i] = 0
		} else {
			paddedStrides[i] = srcStrides[i-pad]
		}
	}
	// For broadcast dims (size 1), set stride to 0.
	for i := range target {
		if srcShape[i] == 1 && target[i] != 1 {
			paddedStrides[i] = 0
		}
	}
	idx := make([]int, len(target))
	for k := 0; k < len(out.Data); k++ {
		off := 0
		for i, ii := range idx {
			off += ii * paddedStrides[i]
		}
		out.Data[k] = t.Data[off]
		// increment idx
		for i := len(idx) - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < target[i] {
				break
			}
			idx[i] = 0
		}
	}
	return out
}

// unbroadcast reduces grad back to target shape by summing over broadcast dims.
// Used during backward when an op broadcasted inputs.
func unbroadcast(grad *Tensor, target []int) *Tensor {
	if shapesEqual(grad.Shape, target) {
		return grad
	}
	// Pad target on the left with 1s.
	pad := len(grad.Shape) - len(target)
	paddedTarget := make([]int, len(grad.Shape))
	for i := range paddedTarget {
		paddedTarget[i] = 1
	}
	for i, d := range target {
		paddedTarget[i+pad] = d
	}
	// Sum over any axis where padded target == 1 but grad shape != 1.
	out := grad
	for axis := 0; axis < len(grad.Shape); axis++ {
		if paddedTarget[axis] == 1 && out.Shape[axis] != 1 {
			out = sumAxis(out, axis, true)
		}
	}
	// Drop the leading padded dims to reach target rank.
	if len(out.Shape) > len(target) {
		newShape := append([]int(nil), out.Shape[len(out.Shape)-len(target):]...)
		out = &Tensor{Data: out.Data, Shape: newShape, Strides: contiguousStrides(newShape)}
	}
	return out
}
