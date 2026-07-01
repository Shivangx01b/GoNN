package tensor

// Matrix multiplication: plain 2D GEMM, and batched matmul over the last two
// dims with NumPy-style broadcasting of the leading (batch) dims — a subset
// of torch.matmul semantics. Both forward and backward run through the
// backend GEMM (dispatchGemm), with the backward using trans flags instead of
// materializing transposes.

// MatMul performs matrix multiplication.
//
//   - 2D @ 2D: the classic C(m,n) = A(m,k) @ B(k,n).
//   - rank >= 3: batched matmul over the last two dims. Leading dims are
//     broadcast NumPy-style, so (B,H,M,K) @ (K,N), (1,H,M,K) @ (B,1,K,N),
//     etc. all work.
//
// Rank < 2 panics; use Reshape to add explicit dims first.
func (t *Tensor) MatMul(o *Tensor) *Tensor {
	if len(t.Shape) < 2 || len(o.Shape) < 2 {
		opError("MatMul", "both tensors must be at least 2D, got %v and %v", t.Shape, o.Shape)
	}
	m, k := t.Shape[len(t.Shape)-2], t.Shape[len(t.Shape)-1]
	k2, n := o.Shape[len(o.Shape)-2], o.Shape[len(o.Shape)-1]
	if k != k2 {
		opError("MatMul", "inner dims do not match: %v @ %v", t.Shape, o.Shape)
	}

	// Fast path: plain 2D GEMM (identical to the historical behavior).
	if len(t.Shape) == 2 && len(o.Shape) == 2 {
		out := New(dispatchGemm(t.Data, o.Data, 1, m, k, n, false, false), m, n)
		finishOp(out, promote(t.Dtype, o.Dtype))
		if t.RequiresGrad || o.RequiresGrad || t.creator != nil || o.creator != nil {
			out.RequiresGrad = true
			out.creator = &Function{
				Name:   "MatMul",
				Inputs: []*Tensor{t, o},
				Backward: func(grad *Tensor, _ []interface{}, inputs []*Tensor) []*Tensor {
					a, b := inputs[0], inputs[1]
					// dA = grad @ B^T, dB = A^T @ grad — via GEMM trans flags,
					// so no explicit transpose copies are materialized.
					ga := dispatchGemm(grad.Data, b.Data, 1, m, n, k, false, true)
					gb := dispatchGemm(a.Data, grad.Data, 1, k, m, n, true, false)
					return []*Tensor{
						New(ga, a.Shape...),
						New(gb, b.Shape...),
					}
				},
			}
		}
		return out
	}

	return batchedMatMul(t, o, m, k, n)
}

// BMM is the strict batched form: t is (B, M, K) and o is (B, K, N) with
// identical batch sizes (no broadcasting). Returns (B, M, N).
func (t *Tensor) BMM(o *Tensor) *Tensor {
	if len(t.Shape) != 3 || len(o.Shape) != 3 {
		opError("BMM", "both tensors must be 3D, got %v and %v", t.Shape, o.Shape)
	}
	if t.Shape[0] != o.Shape[0] {
		opError("BMM", "batch dims do not match: %v @ %v", t.Shape, o.Shape)
	}
	if t.Shape[2] != o.Shape[1] {
		opError("BMM", "inner dims do not match: %v @ %v", t.Shape, o.Shape)
	}
	return batchedMatMul(t, o, t.Shape[1], t.Shape[2], o.Shape[2])
}

// batchedMatMul implements the rank>=3 path: broadcast the leading dims,
// flatten to (B, m, k) @ (B, k, n), one strided-batched GEMM, reshape back.
func batchedMatMul(t, o *Tensor, m, k, n int) *Tensor {
	batchShape := broadcastShapes(t.Shape[:len(t.Shape)-2], o.Shape[:len(o.Shape)-2])
	B := numel(batchShape)

	// Materialize broadcast copies (same pattern binOp uses via expandTo).
	ax := expandTo(t, append(append([]int{}, batchShape...), m, k))
	bx := expandTo(o, append(append([]int{}, batchShape...), k, n))

	outShape := append(append([]int{}, batchShape...), m, n)
	out := New(dispatchGemm(ax.Data, bx.Data, B, m, k, n, false, false), outShape...)
	finishOp(out, promote(t.Dtype, o.Dtype))

	if t.RequiresGrad || o.RequiresGrad || t.creator != nil || o.creator != nil {
		out.RequiresGrad = true
		axShape := append([]int(nil), ax.Shape...)
		bxShape := append([]int(nil), bx.Shape...)
		out.creator = &Function{
			Name:   "MatMul",
			Inputs: []*Tensor{t, o},
			Saved:  []interface{}{ax, bx},
			Backward: func(grad *Tensor, saved []interface{}, _ []*Tensor) []*Tensor {
				ax := saved[0].(*Tensor)
				bx := saved[1].(*Tensor)
				// Per batch: dA = grad @ B^T (m,k), dB = A^T @ grad (k,n).
				ga := New(dispatchGemm(grad.Data, bx.Data, B, m, n, k, false, true), axShape...)
				gb := New(dispatchGemm(ax.Data, grad.Data, B, k, m, n, true, false), bxShape...)
				// The autograd engine unbroadcasts these back to the original
				// (pre-broadcast) input shapes, summing over expanded dims.
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}
