package nn

import (
	"math/rand"
	"strings"
	"testing"

	"gonn/tensor"
)

// WithKDim/WithVDim must size the projections like PyTorch kdim/vdim:
// KProj: (kdim -> embed), VProj: (vdim -> embed); output stays (B, Tq, embed).
func TestMHAKVDimShapes(t *testing.T) {
	m := NewMultiHeadAttention(8, 2, WithKDim(5), WithVDim(6))
	if m.KDim != 5 || m.VDim != 6 {
		t.Fatalf("KDim/VDim: got %d/%d want 5/6", m.KDim, m.VDim)
	}
	if m.KProj.InFeatures != 5 || m.KProj.OutFeatures != 8 {
		t.Fatalf("KProj: got (%d -> %d) want (5 -> 8)", m.KProj.InFeatures, m.KProj.OutFeatures)
	}
	if m.VProj.InFeatures != 6 || m.VProj.OutFeatures != 8 {
		t.Fatalf("VProj: got (%d -> %d) want (6 -> 8)", m.VProj.InFeatures, m.VProj.OutFeatures)
	}

	q := seededRandn(601, 2, 3, 8)
	k := seededRandn(602, 2, 4, 5)
	v := seededRandn(603, 2, 4, 6)

	out := m.Forward(q, k, v, false)
	if out.Shape[0] != 2 || out.Shape[1] != 3 || out.Shape[2] != 8 {
		t.Fatalf("Forward output shape: got %v want [2 3 8]", out.Shape)
	}

	// ForwardMasked accepts the same cross-dim inputs.
	pad := tensor.Zeros(2, 4)
	pad.Data[3] = 1 // ignore last key of batch 0
	out2 := m.ForwardMasked(q, k, v, WithKeyPaddingMask(pad))
	if out2.Shape[0] != 2 || out2.Shape[1] != 3 || out2.Shape[2] != 8 {
		t.Fatalf("ForwardMasked output shape: got %v want [2 3 8]", out2.Shape)
	}
}

// The default construction (no options) must draw the identical RNG sequence
// as the historical two-argument constructor: same layer sizes, same
// construction order — parameters and forward output bit-identical.
func TestMHAKVDimDefaultRNGParity(t *testing.T) {
	rand.Seed(9301)
	m1 := NewMultiHeadAttention(8, 2)
	rand.Seed(9301)
	m2 := NewMultiHeadAttention(8, 2, WithKDim(8), WithVDim(8))

	p1, p2 := m1.Parameters(), m2.Parameters()
	if len(p1) != len(p2) {
		t.Fatalf("param count: %d vs %d", len(p1), len(p2))
	}
	for i := range p1 {
		tensorsEqualExact(t, "param", p1[i], p2[i])
	}

	x := seededRandn(604, 2, 3, 8)
	tensorsEqualExact(t, "forward", m1.Forward(x, x, x, true), m2.Forward(x, x, x, true))
	tensorsEqualExact(t, "forwardmasked",
		m1.ForwardMasked(x, x, x, WithCausal()),
		m2.ForwardMasked(x, x, x, WithCausal()))
}

// Full gradcheck through cross-attention where the memory has different key
// and value feature sizes.
func TestGradCheckMHAKVDim(t *testing.T) {
	m := NewMultiHeadAttention(6, 2, WithKDim(4), WithVDim(5))
	q := seededRandn(611, 2, 3, 6).SetRequiresGrad(true)
	k := seededRandn(612, 2, 4, 4).SetRequiresGrad(true)
	v := seededRandn(613, 2, 4, 5).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return m.Forward(q, k, v, false).Square().Mean() }
	gradCheck(t, "MHA-kdim-vdim", loss, append(m.Parameters(), q, k, v), gcEps, gcTol, 30)
}

// Cross-attention with a memory of different dim: an encoder-decoder shape,
// decoder queries (embed) attending over encoder memory (kdim = vdim = mem).
func TestMHAKVDimCrossAttention(t *testing.T) {
	const embed, mem = 8, 12
	m := NewMultiHeadAttention(embed, 4, WithKDim(mem), WithVDim(mem))
	q := seededRandn(621, 2, 5, embed)
	memory := seededRandn(622, 2, 7, mem)

	out := m.Forward(q, memory, memory, false)
	if out.Shape[0] != 2 || out.Shape[1] != 5 || out.Shape[2] != embed {
		t.Fatalf("cross-attention output shape: got %v want [2 5 %d]", out.Shape, embed)
	}

	// Gradients flow into the memory.
	memory.SetRequiresGrad(true)
	m.Forward(q, memory, memory, false).Sum().Backward()
	if memory.Grad == nil {
		t.Fatal("memory received no gradient")
	}
	nonzero := false
	for _, g := range memory.Grad.Data {
		if g != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("memory gradient is all zeros")
	}
}

// Wrong input feature dims must panic with a clear message on every path.
func TestMHAKVDimValidation(t *testing.T) {
	m := NewMultiHeadAttention(8, 2, WithKDim(5), WithVDim(6))
	q := seededRandn(631, 2, 3, 8)
	k := seededRandn(632, 2, 4, 5)
	v := seededRandn(633, 2, 4, 6)

	mustPanic := func(name, wantSub string, f func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: expected panic", name)
			}
			if msg, ok := r.(string); !ok || !strings.Contains(msg, wantSub) {
				t.Fatalf("%s: panic %v does not mention %q", name, r, wantSub)
			}
		}()
		f()
	}

	badK := seededRandn(634, 2, 4, 8) // embed-dim keys against kdim=5
	badV := seededRandn(635, 2, 4, 8)
	mustPanic("bad k", "kdim", func() { m.Forward(q, badK, v, false) })
	mustPanic("bad v", "vdim", func() { m.Forward(q, k, badV, false) })
	mustPanic("bad q", "q shape", func() { m.Forward(k, k, v, false) })
	mustPanic("masked bad k", "kdim", func() { m.ForwardMasked(q, badK, v) })

	kShort := seededRandn(636, 2, 3, 5) // Tk mismatch vs v's Tk=4
	mustPanic("k/v length mismatch", "sequence lengths differ", func() { m.Forward(q, kShort, v, false) })

	mustPanic("bad kdim ctor", "kdim", func() { NewMultiHeadAttention(8, 2, WithKDim(0)) })
	mustPanic("bad vdim ctor", "vdim", func() { NewMultiHeadAttention(8, 2, WithVDim(-1)) })
}
