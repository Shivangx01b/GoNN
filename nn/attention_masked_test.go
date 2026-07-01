package nn

import (
	"testing"

	"gonn/tensor"
)

func tensorsEqualExact(t *testing.T, name string, a, b *tensor.Tensor) {
	t.Helper()
	if len(a.Data) != len(b.Data) {
		t.Fatalf("%s: size mismatch %v vs %v", name, a.Shape, b.Shape)
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("%s: data[%d]: %.17g != %.17g", name, i, a.Data[i], b.Data[i])
		}
	}
}

// ForwardMasked(WithCausal()) must reproduce Forward(causal=true) exactly on
// the CPU (non-fused) path: same op sequence, bit-identical numerics.
func TestForwardMaskedCausalEquivalence(t *testing.T) {
	m := NewMultiHeadAttention(8, 2) // HeadDim=4: fused kernel never applies
	q := seededRandn(301, 2, 3, 8)
	k := seededRandn(302, 2, 5, 8)
	v := seededRandn(303, 2, 5, 8)

	// Self-attention, causal.
	a := m.Forward(q, q, q, true)
	b := m.ForwardMasked(q, q, q, WithCausal())
	tensorsEqualExact(t, "causal self-attn", a, b)

	// Cross-attention, no mask at all.
	a = m.Forward(q, k, v, false)
	b = m.ForwardMasked(q, k, v)
	tensorsEqualExact(t, "unmasked cross-attn", a, b)
}

// A key flagged by the key-padding mask must contribute nothing: changing its
// key/value content must not change the output at all.
func TestForwardMaskedKeyPaddingIgnoresKeys(t *testing.T) {
	m := NewMultiHeadAttention(8, 2)
	B, Tq, Tk, E := 2, 3, 4, 8
	q := seededRandn(311, B, Tq, E)
	k := seededRandn(312, B, Tk, E)
	v := seededRandn(313, B, Tk, E)

	// mask[b][t] != 0 => ignore key t of batch b.
	mask := tensor.Zeros(B, Tk)
	mask.Data[0*Tk+3] = 1
	mask.Data[1*Tk+1] = 1

	out1 := m.ForwardMasked(q, k, v, WithKeyPaddingMask(mask))

	// Scramble k and v at exactly the masked (b, t) positions.
	k2 := tensor.New(append([]float64(nil), k.Data...), B, Tk, E)
	v2 := tensor.New(append([]float64(nil), v.Data...), B, Tk, E)
	for e := 0; e < E; e++ {
		k2.Data[(0*Tk+3)*E+e] += 7.5
		v2.Data[(0*Tk+3)*E+e] -= 3.25
		k2.Data[(1*Tk+1)*E+e] *= -2
		v2.Data[(1*Tk+1)*E+e] += 11
	}
	out2 := m.ForwardMasked(q, k2, v2, WithKeyPaddingMask(mask))
	tensorsEqualExact(t, "key padding independence", out1, out2)

	// Sanity: without the mask the scrambled keys do change the output.
	u1 := m.ForwardMasked(q, k, v)
	u2 := m.ForwardMasked(q, k2, v2)
	same := true
	for i := range u1.Data {
		if u1.Data[i] != u2.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("unmasked outputs unexpectedly identical after scrambling keys")
	}
}

// The (Tq,Tk), (B,Tq,Tk), (B*H,Tq,Tk) and (B,H,Tq,Tk) attn-mask shapes must
// agree when they encode the same mask.
func TestForwardMaskedAttnMaskBroadcast(t *testing.T) {
	m := NewMultiHeadAttention(8, 2)
	B, H, Tq, Tk, E := 2, 2, 3, 4, 8
	q := seededRandn(321, B, Tq, E)
	k := seededRandn(322, B, Tk, E)
	v := seededRandn(323, B, Tk, E)

	base := seededRandn(324, Tq, Tk) // additive float mask
	want := m.ForwardMasked(q, k, v, WithAttnMask(base))

	tile := func(reps int) []float64 {
		out := make([]float64, 0, reps*len(base.Data))
		for r := 0; r < reps; r++ {
			out = append(out, base.Data...)
		}
		return out
	}
	m3b := tensor.New(tile(B), B, Tq, Tk)
	m3bh := tensor.New(tile(B*H), B*H, Tq, Tk)
	m4 := tensor.New(tile(B*H), B, H, Tq, Tk)

	tensorsEqualExact(t, "(B,Tq,Tk) mask", want, m.ForwardMasked(q, k, v, WithAttnMask(m3b)))
	tensorsEqualExact(t, "(B*H,Tq,Tk) mask", want, m.ForwardMasked(q, k, v, WithAttnMask(m3bh)))
	tensorsEqualExact(t, "(B,H,Tq,Tk) mask", want, m.ForwardMasked(q, k, v, WithAttnMask(m4)))

	// Wrong shape must panic.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic for bad attn mask shape")
			}
		}()
		m.ForwardMasked(q, k, v, WithAttnMask(tensor.Zeros(Tq+1, Tk)))
	}()
}

// Attention dropout must be active only in training mode.
func TestForwardMaskedDropoutTrainEval(t *testing.T) {
	m := NewMultiHeadAttention(8, 2)
	q := seededRandn(331, 2, 4, 8)

	ref := m.ForwardMasked(q, q, q, WithCausal())

	m.Eval()
	e1 := m.ForwardMasked(q, q, q, WithCausal(), WithAttnDropout(0.5))
	e2 := m.ForwardMasked(q, q, q, WithCausal(), WithAttnDropout(0.5))
	tensorsEqualExact(t, "eval dropout == no dropout", ref, e1)
	tensorsEqualExact(t, "eval dropout deterministic", e1, e2)

	m.Train()
	t1 := m.ForwardMasked(q, q, q, WithCausal(), WithAttnDropout(0.5))
	diff := false
	for i := range t1.Data {
		if t1.Data[i] != ref.Data[i] {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("training-mode attention dropout had no effect (p=0.5)")
	}
}

func TestGradCheckForwardMasked(t *testing.T) {
	m := NewMultiHeadAttention(8, 2)
	B, Tq, Tk := 2, 3, 4
	q := seededRandn(341, B, Tq, 8).SetRequiresGrad(true)
	k := seededRandn(342, B, Tk, 8).SetRequiresGrad(true)
	v := seededRandn(343, B, Tk, 8).SetRequiresGrad(true)

	attnMask := seededRandn(344, Tq, Tk) // small additive float mask
	// Note: chosen so no (causal-)row ends up fully masked.
	keyPad := tensor.Zeros(B, Tk)
	keyPad.Data[0*Tk+2] = 1
	keyPad.Data[1*Tk+1] = 1

	loss := func() *tensor.Tensor {
		return m.ForwardMasked(q, k, v,
			WithCausal(), WithAttnMask(attnMask), WithKeyPaddingMask(keyPad)).
			Square().Mean()
	}
	gradCheck(t, "ForwardMasked", loss, append(m.Parameters(), q, k, v), gcEps, gcTol, 25)
}
