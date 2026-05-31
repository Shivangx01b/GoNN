package nn

import (
	"math"

	"gonn/tensor"
)

// MultiHeadAttention implements scaled dot-product multi-head attention.
type MultiHeadAttention struct {
	EmbedDim int
	NumHeads int
	HeadDim  int
	QProj    *Linear
	KProj    *Linear
	VProj    *Linear
	OutProj  *Linear
}

// NewMultiHeadAttention builds an MHA with embed_dim split across num_heads.
func NewMultiHeadAttention(embedDim, numHeads int) *MultiHeadAttention {
	if embedDim%numHeads != 0 {
		panic("MultiHeadAttention: embedDim must be divisible by numHeads")
	}
	return &MultiHeadAttention{
		EmbedDim: embedDim,
		NumHeads: numHeads,
		HeadDim:  embedDim / numHeads,
		QProj:    NewLinear(embedDim, embedDim, true),
		KProj:    NewLinear(embedDim, embedDim, true),
		VProj:    NewLinear(embedDim, embedDim, true),
		OutProj:  NewLinear(embedDim, embedDim, true),
	}
}

// Forward computes MHA. q,k,v shapes: (batch, seq, embed). If causal is true,
// applies a causal mask (upper-triangular set to -inf before softmax).
func (m *MultiHeadAttention) Forward(q, k, v *tensor.Tensor, causal bool) *tensor.Tensor {
	B, Tq, _ := q.Shape[0], q.Shape[1], q.Shape[2]
	Tk := k.Shape[1]
	H := m.NumHeads
	D := m.HeadDim

	// project
	qp := m.QProj.Forward(q) // (B, Tq, E)
	kp := m.KProj.Forward(k) // (B, Tk, E)
	vp := m.VProj.Forward(v) // (B, Tk, E)

	// (B, T, H, D) -> (B, H, T, D)
	qp = qp.Reshape(B, Tq, H, D).Permute(0, 2, 1, 3)
	kp = kp.Reshape(B, Tk, H, D).Permute(0, 2, 1, 3)
	vp = vp.Reshape(B, Tk, H, D).Permute(0, 2, 1, 3)

	scale := 1.0 / math.Sqrt(float64(D))

	// Fully-differentiable fused GPU path: when the CUDA flash-attention kernel
	// is available and applicable, run the whole scaled-dot-product core (incl.
	// its backward) through it. This is what makes training use the kernel.
	if fusedAttnAvailable() && D == 64 && Tq == Tk {
		ctx := fusedAttnDiff(qp, kp, vp, B*H, Tq, D, scale, causal).
			Permute(0, 2, 1, 3).Reshape(B, Tq, m.EmbedDim)
		return m.OutProj.Forward(ctx)
	}

	// scores = qp @ kp^T -> (B, H, Tq, Tk)
	scores := batchedMatMul(qp, transposeLastTwo(kp))
	scores = scores.MulScalar(scale)

	if causal {
		mask := tensor.Zeros(1, 1, Tq, Tk)
		neg := -1e30
		for i := 0; i < Tq; i++ {
			for j := 0; j < Tk; j++ {
				if j > i {
					mask.Data[i*Tk+j] = neg
				}
			}
		}
		scores = scores.Add(mask)
	}

	attn := scores.Softmax(3) // softmax over Tk
	// ctx = attn @ vp -> (B, H, Tq, D)
	ctx := batchedMatMul(attn, vp)
	// (B, H, Tq, D) -> (B, Tq, H, D) -> (B, Tq, E)
	ctx = ctx.Permute(0, 2, 1, 3).Reshape(B, Tq, m.EmbedDim)
	return m.OutProj.Forward(ctx)
}

// ForwardFused is a forward-only (inference) attention path that runs the
// scaled-dot-product core on the GPU via the fused flash-attention kernel,
// replacing the projection -> batched-matmul -> softmax -> batched-matmul core
// with a single kernel launch (online softmax, no S*S materialization).
//
// It is NOT autograd-differentiable through the attention core, so use it for
// inference/generation only. It transparently falls back to the regular
// (differentiable) Forward when the fused kernel is unavailable (non-cuda
// build), when HeadDim != 64 (kernel specialization), or when Tq != Tk.
func (m *MultiHeadAttention) ForwardFused(q, k, v *tensor.Tensor, causal bool) *tensor.Tensor {
	B, Tq, _ := q.Shape[0], q.Shape[1], q.Shape[2]
	Tk := k.Shape[1]
	H, D := m.NumHeads, m.HeadDim

	if !fusedAttnAvailable() || D != 64 || Tq != Tk {
		return m.Forward(q, k, v, causal)
	}

	// Project and split heads -> (B, H, T, D), contiguous after Permute.
	qp := m.QProj.Forward(q).Reshape(B, Tq, H, D).Permute(0, 2, 1, 3)
	kp := m.KProj.Forward(k).Reshape(B, Tk, H, D).Permute(0, 2, 1, 3)
	vp := m.VProj.Forward(v).Reshape(B, Tk, H, D).Permute(0, 2, 1, 3)

	scale := 1.0 / math.Sqrt(float64(D))
	// (B*H, T, D) row-major matches the kernel's (BH, S, d) layout exactly.
	O := fusedAttnF64(qp.Data, kp.Data, vp.Data, B*H, Tq, D, scale, causal)

	ctx := tensor.New(O, B, H, Tq, D).Permute(0, 2, 1, 3).Reshape(B, Tq, m.EmbedDim)
	return m.OutProj.Forward(ctx)
}

// Parameters returns all linear projection parameters.
func (m *MultiHeadAttention) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	ps = append(ps, m.QProj.Parameters()...)
	ps = append(ps, m.KProj.Parameters()...)
	ps = append(ps, m.VProj.Parameters()...)
	ps = append(ps, m.OutProj.Parameters()...)
	return ps
}

// batchedMatMul performs matmul on the last two dims of 4D tensors
// a: (B, H, M, K), b: (B, H, K, N) -> (B, H, M, N).
// Implemented via reshape to (B*H, M, K) and pair-wise 2D matmul; we flatten
// further to a single big matmul by folding B*H into M when shapes allow.
func batchedMatMul(a, b *tensor.Tensor) *tensor.Tensor {
	// Expect 4D inputs.
	B, H, M, K := a.Shape[0], a.Shape[1], a.Shape[2], a.Shape[3]
	K2, N := b.Shape[2], b.Shape[3]
	if K != K2 || B != b.Shape[0] || H != b.Shape[1] {
		panic("batchedMatMul: shape mismatch")
	}
	// Build a block-diagonal of b matrices so a single 2D matmul covers all batches.
	// a flat: (B*H*M, K) by reshape. b block: (B*H*K, B*H*N) by zero-padding.
	// Then result[i, b*H*N + j] picks the right block. But this is sparse and
	// memory-expensive. Instead, just loop and stack results.
	out := tensor.Zeros(B, H, M, N)
	// Per-batch 2D matmul, then "place" into out via broadcasted multiply-add.
	// To keep autograd, do all ops in tensor space.
	var summed *tensor.Tensor
	for bi := 0; bi < B; bi++ {
		for hi := 0; hi < H; hi++ {
			aSlice := slice4DTo2D(a, bi, hi) // (M, K)
			bSlice := slice4DTo2D(b, bi, hi) // (K, N)
			prod := aSlice.MatMul(bSlice)    // (M, N)
			placed := placeIn4D(prod, B, H, M, N, bi, hi)
			if summed == nil {
				summed = placed
			} else {
				summed = summed.Add(placed)
			}
		}
	}
	if summed == nil {
		return out
	}
	return summed
}

// slice4DTo2D returns t[b, h, :, :] as (M, N) via a permuted reshape + masked matmul.
// We can't slice directly so we use a one-hot selector.
func slice4DTo2D(t *tensor.Tensor, bi, hi int) *tensor.Tensor {
	B, H, M, N := t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3]
	// reshape t to (B*H, M*N), then select row (bi*H+hi).
	flat := t.Reshape(B*H, M*N)
	sel := tensor.Zeros(1, B*H)
	sel.Data[bi*H+hi] = 1
	row := sel.MatMul(flat) // (1, M*N)
	return row.Reshape(M, N)
}

// placeIn4D embeds a (M, N) tensor at position (bi, hi) of a (B, H, M, N) zero tensor.
func placeIn4D(x *tensor.Tensor, B, H, M, N, bi, hi int) *tensor.Tensor {
	// Build selector e of shape (B, H, 1, 1) with 1 at (bi, hi).
	sel := tensor.Zeros(B, H, 1, 1)
	sel.Data[bi*H+hi] = 1
	xExp := x.Reshape(1, 1, M, N)
	return xExp.Mul(sel) // broadcasts to (B, H, M, N)
}

// transposeLastTwo swaps the last two dims of a 4D tensor.
func transposeLastTwo(t *tensor.Tensor) *tensor.Tensor {
	return t.Permute(0, 1, 3, 2)
}
