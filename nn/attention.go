package nn

import (
	"fmt"
	"math"
	"math/rand"

	"gonn/tensor"
)

// MultiHeadAttention implements scaled dot-product multi-head attention.
// It takes three inputs (q, k, v), so it satisfies Child but not Module.
type MultiHeadAttention struct {
	Base
	EmbedDim int
	NumHeads int
	HeadDim  int
	// KDim/VDim are the key/value input feature sizes (PyTorch kdim/vdim).
	// Zero means EmbedDim (the default self-attention configuration).
	KDim    int
	VDim    int
	QProj   *Linear
	KProj   *Linear
	VProj   *Linear
	OutProj *Linear
}

// MHAOpt configures NewMultiHeadAttention.
type MHAOpt func(*mhaOpts)

type mhaOpts struct {
	kdim int
	vdim int
}

// WithKDim sets the key input feature size (PyTorch kdim): KProj becomes
// Linear(kdim, embedDim). Default is embedDim.
func WithKDim(k int) MHAOpt { return func(o *mhaOpts) { o.kdim = k } }

// WithVDim sets the value input feature size (PyTorch vdim): VProj becomes
// Linear(vdim, embedDim). Default is embedDim.
func WithVDim(v int) MHAOpt { return func(o *mhaOpts) { o.vdim = v } }

// NewMultiHeadAttention builds an MHA with embed_dim split across num_heads.
// Optional WithKDim/WithVDim set distinct key/value input feature sizes
// (PyTorch kdim/vdim); by default both equal embedDim, which draws the exact
// RNG sequence of the historical two-argument constructor.
func NewMultiHeadAttention(embedDim, numHeads int, opts ...MHAOpt) *MultiHeadAttention {
	if embedDim%numHeads != 0 {
		panic("MultiHeadAttention: embedDim must be divisible by numHeads")
	}
	o := mhaOpts{kdim: embedDim, vdim: embedDim}
	for _, fn := range opts {
		fn(&o)
	}
	if o.kdim < 1 {
		panic(fmt.Sprintf("MultiHeadAttention: kdim must be >= 1, got %d", o.kdim))
	}
	if o.vdim < 1 {
		panic(fmt.Sprintf("MultiHeadAttention: vdim must be >= 1, got %d", o.vdim))
	}
	m := &MultiHeadAttention{
		EmbedDim: embedDim,
		NumHeads: numHeads,
		HeadDim:  embedDim / numHeads,
		KDim:     o.kdim,
		VDim:     o.vdim,
		QProj:    NewLinear(embedDim, embedDim, true),
		KProj:    NewLinear(o.kdim, embedDim, true),
		VProj:    NewLinear(o.vdim, embedDim, true),
		OutProj:  NewLinear(embedDim, embedDim, true),
	}
	m.regChild("qproj", m.QProj)
	m.regChild("kproj", m.KProj)
	m.regChild("vproj", m.VProj)
	m.regChild("outproj", m.OutProj)
	return m
}

// checkQKVDims validates the q/k/v input feature dims against EmbedDim and
// KDim/VDim. Zero KDim/VDim (struct-literal construction) fall back to
// EmbedDim, preserving historical behavior.
func (m *MultiHeadAttention) checkQKVDims(q, k, v *tensor.Tensor) {
	kd, vd := m.KDim, m.VDim
	if kd == 0 {
		kd = m.EmbedDim
	}
	if vd == 0 {
		vd = m.EmbedDim
	}
	if len(q.Shape) != 3 || q.Shape[2] != m.EmbedDim {
		panic(fmt.Sprintf("MultiHeadAttention: q shape %v, want (B, Tq, %d)", q.Shape, m.EmbedDim))
	}
	if len(k.Shape) != 3 || k.Shape[2] != kd {
		panic(fmt.Sprintf("MultiHeadAttention: k shape %v, want (B, Tk, kdim=%d)", k.Shape, kd))
	}
	if len(v.Shape) != 3 || v.Shape[2] != vd {
		panic(fmt.Sprintf("MultiHeadAttention: v shape %v, want (B, Tk, vdim=%d)", v.Shape, vd))
	}
	if k.Shape[1] != v.Shape[1] {
		panic(fmt.Sprintf("MultiHeadAttention: k and v sequence lengths differ (%d vs %d)", k.Shape[1], v.Shape[1]))
	}
}

// Forward computes MHA. q,k,v shapes: (batch, seq, embed). If causal is true,
// applies a causal mask (upper-triangular set to -inf before softmax).
func (m *MultiHeadAttention) Forward(q, k, v *tensor.Tensor, causal bool) *tensor.Tensor {
	m.checkQKVDims(q, k, v)
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
		scores = scores.Add(causalMaskAdd(Tq, Tk))
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
	m.checkQKVDims(q, k, v)
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

// causalMaskAdd builds the additive causal mask (1, 1, Tq, Tk): 0 on and
// below the diagonal, -1e30 above (future positions). Shared by Forward and
// ForwardMasked so both produce bit-identical numerics.
func causalMaskAdd(Tq, Tk int) *tensor.Tensor {
	mask := tensor.Zeros(1, 1, Tq, Tk)
	neg := -1e30
	for i := 0; i < Tq; i++ {
		for j := 0; j < Tk; j++ {
			if j > i {
				mask.Data[i*Tk+j] = neg
			}
		}
	}
	return mask
}

// AttnOpt configures MultiHeadAttention.ForwardMasked.
type AttnOpt func(*attnOpts)

type attnOpts struct {
	causal   bool
	attnMask *tensor.Tensor
	keyPad   *tensor.Tensor
	dropoutP float64
}

// WithCausal applies the causal (upper-triangular -inf) mask, exactly like
// Forward(q, k, v, true).
func WithCausal() AttnOpt { return func(o *attnOpts) { o.causal = true } }

// WithAttnMask supplies an additive float mask on the pre-softmax attention
// scores, PyTorch attn_mask style. Accepted shapes (broadcast to
// (B, H, Tq, Tk)):
//
//	(Tq, Tk)        — shared across batch and heads
//	(B, Tq, Tk)     — per-batch, shared across heads
//	(B*H, Tq, Tk)   — per batch-and-head (PyTorch's 3-D form)
//	(B, H, Tq, Tk)  — fully explicit
//
// Use large negative values (e.g. -1e30) to block positions.
func WithAttnMask(mask *tensor.Tensor) AttnOpt {
	return func(o *attnOpts) { o.attnMask = mask }
}

// WithKeyPaddingMask supplies a (B, Tk) mask where a nonzero entry means
// "ignore this key": -1e30 is added to that key's scores for every query and
// head (PyTorch key_padding_mask with True = ignore).
func WithKeyPaddingMask(mask *tensor.Tensor) AttnOpt {
	return func(o *attnOpts) { o.keyPad = mask }
}

// WithAttnDropout applies dropout with probability p to the softmaxed
// attention weights (PyTorch MultiheadAttention dropout). Active only while
// the module is in Training() mode; identity in Eval().
func WithAttnDropout(p float64) AttnOpt {
	return func(o *attnOpts) { o.dropoutP = p }
}

// expandAttnMask reshapes a user attn mask to broadcast against
// (B, H, Tq, Tk) scores. See WithAttnMask for the accepted shapes.
func expandAttnMask(mask *tensor.Tensor, B, H, Tq, Tk int) *tensor.Tensor {
	s := mask.Shape
	switch {
	case len(s) == 2 && s[0] == Tq && s[1] == Tk:
		return mask.Reshape(1, 1, Tq, Tk)
	case len(s) == 3 && s[0] == B*H && s[1] == Tq && s[2] == Tk:
		return mask.Reshape(B, H, Tq, Tk)
	case len(s) == 3 && s[0] == B && s[1] == Tq && s[2] == Tk:
		return mask.Reshape(B, 1, Tq, Tk)
	case len(s) == 4 && s[0] == B && s[1] == H && s[2] == Tq && s[3] == Tk:
		return mask
	}
	panic(fmt.Sprintf(
		"MultiHeadAttention: attn mask shape %v incompatible with (Tq,Tk)=(%d,%d), (B,Tq,Tk) B=%d, (B*H,Tq,Tk) B*H=%d, or (B,H,Tq,Tk)",
		s, Tq, Tk, B, B*H))
}

// keyPaddingMaskAdd converts a (B, Tk) nonzero-means-ignore mask into an
// additive (B, 1, 1, Tk) score mask of 0 / -1e30 constants.
func keyPaddingMaskAdd(mask *tensor.Tensor, B, Tk int) *tensor.Tensor {
	if len(mask.Shape) != 2 || mask.Shape[0] != B || mask.Shape[1] != Tk {
		panic(fmt.Sprintf("MultiHeadAttention: key padding mask shape %v, want (B, Tk) = (%d, %d)", mask.Shape, B, Tk))
	}
	add := tensor.Zeros(B, 1, 1, Tk)
	for i, v := range mask.Data {
		if v != 0 {
			add.Data[i] = -1e30
		}
	}
	return add
}

// dropoutAttn applies inverted dropout with probability p to x (used on
// attention weights). Caller guarantees training mode and p > 0.
func dropoutAttn(x *tensor.Tensor, p float64) *tensor.Tensor {
	if p >= 1 {
		return x.MulScalar(0)
	}
	keep := 1.0 - p
	scale := 1.0 / keep
	mask := tensor.Zeros(x.Shape...)
	for i := range mask.Data {
		if rand.Float64() < keep {
			mask.Data[i] = scale
		}
	}
	return x.Mul(mask)
}

// ForwardMasked computes MHA with PyTorch-style masking and attention
// dropout, configured via options:
//
//	WithCausal()               — causal mask, identical to Forward(..., true)
//	WithAttnMask(mask)         — additive float mask (see accepted shapes)
//	WithKeyPaddingMask(mask)   — (B, Tk), nonzero = ignore that key
//	WithAttnDropout(p)         — dropout on softmaxed weights (Training only)
//
// q, k, v shapes: (batch, seq, embed), like Forward. With only WithCausal()
// (or no options) it reproduces the CPU Forward path exactly.
//
// ForwardMasked always uses the non-fused (pure autograd) path: the fused
// CUDA flash-attention kernel supports at most a causal mask and no dropout,
// so it is never taken here. Use Forward for the fused fast path.
func (m *MultiHeadAttention) ForwardMasked(q, k, v *tensor.Tensor, opts ...AttnOpt) *tensor.Tensor {
	var o attnOpts
	for _, fn := range opts {
		fn(&o)
	}

	m.checkQKVDims(q, k, v)
	B, Tq, _ := q.Shape[0], q.Shape[1], q.Shape[2]
	Tk := k.Shape[1]
	H := m.NumHeads
	D := m.HeadDim

	qp := m.QProj.Forward(q).Reshape(B, Tq, H, D).Permute(0, 2, 1, 3)
	kp := m.KProj.Forward(k).Reshape(B, Tk, H, D).Permute(0, 2, 1, 3)
	vp := m.VProj.Forward(v).Reshape(B, Tk, H, D).Permute(0, 2, 1, 3)

	scale := 1.0 / math.Sqrt(float64(D))
	scores := qp.MatMul(kp.Transpose()).MulScalar(scale) // (B, H, Tq, Tk)

	if o.causal {
		scores = scores.Add(causalMaskAdd(Tq, Tk))
	}
	if o.attnMask != nil {
		scores = scores.Add(expandAttnMask(o.attnMask, B, H, Tq, Tk))
	}
	if o.keyPad != nil {
		scores = scores.Add(keyPaddingMaskAdd(o.keyPad, B, Tk))
	}

	attn := scores.Softmax(3)
	if o.dropoutP > 0 && m.Training() {
		attn = dropoutAttn(attn, o.dropoutP)
	}

	ctx := attn.MatMul(vp).Permute(0, 2, 1, 3).Reshape(B, Tq, m.EmbedDim)
	return m.OutProj.Forward(ctx)
}
