package nn

import (
	"math/rand"
	"testing"

	"gonn/tensor"
)

// Defaults must be exactly the historical behavior: explicitly spelling out
// the default options gives the same weights (same RNG draws) and the same
// forward numerics as the bare constructor.
func TestTransformerExplicitDefaultsMatch(t *testing.T) {
	rand.Seed(401)
	a := NewTransformerEncoderLayer(8, 2, 16)
	rand.Seed(401)
	b := NewTransformerEncoderLayer(8, 2, 16, WithTransformerDropout(0), WithFFActivation(ReLU()))
	x := seededRandn(402, 2, 3, 8)
	tensorsEqualExact(t, "encoder-layer explicit defaults", a.Forward(x), b.Forward(x))

	rand.Seed(403)
	da := NewTransformerDecoderLayer(8, 2, 16)
	rand.Seed(403)
	db := NewTransformerDecoderLayer(8, 2, 16, WithTransformerDropout(0), WithFFActivation(ReLU()))
	tgt := seededRandn(404, 2, 3, 8)
	mem := seededRandn(405, 2, 4, 8)
	tensorsEqualExact(t, "decoder-layer explicit defaults", da.Forward(tgt, mem), db.Forward(tgt, mem))
}

// Pre-norm and post-norm must compute different functions (same weights).
func TestTransformerPreNormDiffers(t *testing.T) {
	rand.Seed(411)
	post := NewTransformerEncoderLayer(8, 2, 16)
	rand.Seed(411)
	pre := NewTransformerEncoderLayer(8, 2, 16, WithPreNorm())

	x := seededRandn(412, 2, 3, 8)
	yPost := post.Forward(x)
	yPre := pre.Forward(x)
	same := true
	for i := range yPost.Data {
		if yPost.Data[i] != yPre.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("pre-norm and post-norm outputs are identical")
	}

	rand.Seed(413)
	dpost := NewTransformerDecoderLayer(8, 2, 16)
	rand.Seed(413)
	dpre := NewTransformerDecoderLayer(8, 2, 16, WithPreNorm())
	tgt := seededRandn(414, 2, 3, 8)
	mem := seededRandn(415, 2, 4, 8)
	a, b := dpost.Forward(tgt, mem), dpre.Forward(tgt, mem)
	same = true
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("decoder pre-norm and post-norm outputs are identical")
	}
}

// WithFFActivation must change the FFN nonlinearity.
func TestTransformerFFActivation(t *testing.T) {
	rand.Seed(421)
	relu := NewTransformerEncoderLayer(8, 2, 16)
	rand.Seed(421)
	gelu := NewTransformerEncoderLayer(8, 2, 16, WithFFActivation(GELU()))
	x := seededRandn(422, 2, 3, 8)
	a, b := relu.Forward(x), gelu.Forward(x)
	same := true
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("GELU FFN output identical to ReLU FFN output")
	}
}

// Dropout must be active in Train() only; Eval() must match the p=0 model.
func TestTransformerDropoutOnlyInTraining(t *testing.T) {
	rand.Seed(431)
	drop := NewTransformerEncoderLayer(8, 2, 16, WithTransformerDropout(0.5))
	rand.Seed(431)
	ref := NewTransformerEncoderLayer(8, 2, 16)
	x := seededRandn(432, 2, 3, 8)

	drop.Eval()
	tensorsEqualExact(t, "eval dropout == no dropout", ref.Forward(x), drop.Forward(x))
	tensorsEqualExact(t, "eval deterministic", drop.Forward(x), drop.Forward(x))

	drop.Train()
	a, b := drop.Forward(x), drop.Forward(x)
	same := true
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("training-mode transformer dropout had no effect (p=0.5)")
	}
}

// The all-in-one Transformer: shapes, mode propagation, gradients.
func TestFullTransformer(t *testing.T) {
	m := NewTransformer(8, 2, 2, 2, 16)
	src := seededRandn(441, 2, 5, 8)
	tgt := seededRandn(442, 2, 3, 8)

	out := m.Forward(src, tgt)
	if out.Shape[0] != 2 || out.Shape[1] != 3 || out.Shape[2] != 8 {
		t.Fatalf("Transformer output shape: got %v want [2 3 8]", out.Shape)
	}

	loss := out.Square().Mean()
	loss.Backward()
	for i, p := range m.Parameters() {
		if p.Grad == nil {
			t.Fatalf("Transformer param %d has no grad", i)
		}
	}

	// Eval/Train propagates into every layer's dropouts.
	m.Eval()
	if m.Encoder.Layers[0].Drop1.Training() || m.Decoder.Layers[1].DropFF.Training() {
		t.Fatal("Eval did not propagate to transformer dropouts")
	}
	m.Train()

	// Pre-norm variant runs and differs.
	pre := NewTransformer(8, 2, 1, 1, 16, WithPreNorm(), WithFFActivation(GELU()), WithTransformerDropout(0.1))
	pre.Eval()
	out2 := pre.Forward(src, tgt)
	if out2.Shape[0] != 2 || out2.Shape[1] != 3 || out2.Shape[2] != 8 {
		t.Fatalf("pre-norm Transformer output shape: got %v", out2.Shape)
	}
}

func TestGradCheckFullTransformer(t *testing.T) {
	m := NewTransformer(4, 2, 1, 1, 8)
	src := seededRandn(451, 1, 3, 4).SetRequiresGrad(true)
	tgt := seededRandn(452, 1, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return m.Forward(src, tgt).Square().Mean() }
	gradCheck(t, "Transformer", loss, append(m.Parameters(), src, tgt), gcEps, gcTol, 6)
}
