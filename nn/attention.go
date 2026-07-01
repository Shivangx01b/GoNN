package nn

import (
	"math"

	"gonn/tensor"
)

// MultiHeadAttention implements scaled dot-product multi-head attention.
// It takes three inputs (q, k, v), so it satisfies Child but not Module.
type MultiHeadAttention struct {
	Base
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
	m := &MultiHeadAttention{
		EmbedDim: embedDim,
		NumHeads: numHeads,
		HeadDim:  embedDim / numHeads,
		QProj:    NewLinear(embedDim, embedDim, true),
		KProj:    NewLinear(embedDim, embedDim, true),
		VProj:    NewLinear(embedDim, embedDim, true),
		OutProj:  NewLinear(embedDim, embedDim, true),
	}
	m.regChild("qproj", m.QProj)
	m.regChild("kproj", m.KProj)
	m.regChild("vproj", m.VProj)
	m.regChild("outproj", m.OutProj)
	return m
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

	// scores = qp @ kp^T -> (B, H, Tq, Tk); batched matmul over the head dims.
	scores := qp.MatMul(kp.Transpose()).MulScalar(scale)

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
	ctx := attn.MatMul(vp)
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
