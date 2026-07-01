package nn

import (
	"math/rand"
	"strings"
	"testing"

	"gonn/tensor"
)

// ============================================================================
// WithRNNDropout
// ============================================================================

// p = 0 must construct a module tree identical to the default: no Dropout
// children, same parameters, bit-identical forward.
func TestRNNDropoutZeroIsDefault(t *testing.T) {
	x := seededRandn(701, 2, 4, 3)

	rand.Seed(9401)
	m1 := NewLSTM(3, 4, WithLayers(2))
	rand.Seed(9401)
	m2 := NewLSTM(3, 4, WithLayers(2), WithRNNDropout(0))

	if m2.Dropouts != nil {
		t.Fatal("WithRNNDropout(0) must not construct Dropout modules")
	}
	p1, p2 := m1.Parameters(), m2.Parameters()
	if len(p1) != len(p2) {
		t.Fatalf("param count: %d vs %d", len(p1), len(p2))
	}
	for i := range p1 {
		tensorsEqualExact(t, "param", p1[i], p2[i])
	}
	tensorsEqualExact(t, "forward", m1.Forward(x), m2.Forward(x))

	// A single layer never gets inter-layer dropout (PyTorch: dropout applies
	// to all layers except the last).
	if m := NewGRU(3, 4, WithRNNDropout(0.5)); m.Dropouts != nil {
		t.Fatal("single-layer stack must not construct Dropout modules")
	}
}

// Training mode must be stochastic; Eval must be deterministic and equal to
// the same-weights no-dropout model (dropout is identity at eval).
func TestRNNDropoutTrainEval(t *testing.T) {
	x := seededRandn(702, 2, 4, 3)

	rand.Seed(9402)
	drop := NewGRU(3, 4, WithLayers(3), WithRNNDropout(0.5))
	rand.Seed(9402)
	plain := NewGRU(3, 4, WithLayers(3))

	if len(drop.Dropouts) != 2 {
		t.Fatalf("Dropouts: got %d want layers-1 = 2", len(drop.Dropouts))
	}
	// Dropout has no parameters, so the parameter lists still match.
	if len(drop.Parameters()) != len(plain.Parameters()) {
		t.Fatalf("param count: %d vs %d", len(drop.Parameters()), len(plain.Parameters()))
	}

	// Training: two runs differ (P(identical masks) is astronomically small).
	y1 := drop.Forward(x)
	y2 := drop.Forward(x)
	same := true
	for i := range y1.Data {
		if y1.Data[i] != y2.Data[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("training-mode dropout produced identical outputs on two runs")
	}

	// Eval propagates to the registered Dropout children and is deterministic.
	drop.Eval()
	for i, d := range drop.Dropouts {
		if d.Training() {
			t.Fatalf("Dropouts[%d] still in training mode after Eval()", i)
		}
	}
	e1 := drop.Forward(x)
	e2 := drop.Forward(x)
	tensorsEqualExact(t, "eval deterministic", e1, e2)
	tensorsEqualExact(t, "eval == no-dropout", e1, plain.Forward(x))
}

func TestRNNDropoutValidation(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for dropout out of [0, 1]")
		}
	}()
	NewRNN(3, 4, WithLayers(2), WithRNNDropout(1.5))
}

// ============================================================================
// WithProjSize (LSTM proj_size)
// ============================================================================

// Output feature dim == proj, cell state dim == hidden; ForwardWithState
// shapes follow PyTorch: h is (LD, B, proj), c is (LD, B, hidden).
func TestLSTMProjSizeShapes(t *testing.T) {
	const B, T, F, H, P = 2, 4, 3, 6, 2
	x := seededRandn(711, B, T, F)

	l := NewLSTM(F, H, WithProjSize(P))
	if got := l.Forward(x); got.Shape[0] != B || got.Shape[1] != T || got.Shape[2] != P {
		t.Fatalf("Forward shape: got %v want [%d %d %d]", got.Shape, B, T, P)
	}
	out, hN, cN := l.ForwardWithState(x, nil, nil)
	if out.Shape[2] != P {
		t.Fatalf("output feature dim: got %d want proj=%d", out.Shape[2], P)
	}
	if hN.Shape[0] != 1 || hN.Shape[1] != B || hN.Shape[2] != P {
		t.Fatalf("hN shape: got %v want [1 %d %d]", hN.Shape, B, P)
	}
	if cN.Shape[0] != 1 || cN.Shape[1] != B || cN.Shape[2] != H {
		t.Fatalf("cN shape: got %v want [1 %d %d]", cN.Shape, B, H)
	}

	// Round-trip: feeding (hN, cN) back in must be accepted (h is proj-sized,
	// c is hidden-sized).
	out2, _, _ := l.ForwardWithState(x, hN, cN)
	if out2.Shape[2] != P {
		t.Fatalf("state round-trip output dim: got %d want %d", out2.Shape[2], P)
	}

	// Multi-layer bidirectional: features are 2*proj, states (L*2, B, proj/hidden).
	lb := NewLSTM(F, H, WithLayers(2), WithBidirectional(), WithProjSize(P))
	outB, hB, cB := lb.ForwardWithState(x, nil, nil)
	if outB.Shape[2] != 2*P {
		t.Fatalf("bidir output feature dim: got %d want %d", outB.Shape[2], 2*P)
	}
	if hB.Shape[0] != 4 || hB.Shape[2] != P {
		t.Fatalf("bidir hN shape: got %v want [4 %d %d]", hB.Shape, B, P)
	}
	if cB.Shape[0] != 4 || cB.Shape[2] != H {
		t.Fatalf("bidir cN shape: got %v want [4 %d %d]", cB.Shape, B, H)
	}
}

// The cell-level projection: h is (B, proj), c stays (B, hidden), Whh
// consumes proj-sized hidden state and Whr is Linear(hidden, proj, no bias).
func TestLSTMCellProjSize(t *testing.T) {
	c := NewLSTMCell(3, 6, WithProjSize(2))
	if c.Whr == nil || c.Whr.InFeatures != 6 || c.Whr.OutFeatures != 2 || c.Whr.Bias != nil {
		t.Fatalf("Whr: want Linear(6, 2, no bias), got %+v", c.Whr)
	}
	if c.Whh.InFeatures != 2 {
		t.Fatalf("Whh input dim: got %d want proj=2", c.Whh.InFeatures)
	}
	x := seededRandn(721, 2, 3)
	st := c.Forward(x, nil)
	if st.H.Shape[0] != 2 || st.H.Shape[1] != 2 {
		t.Fatalf("h shape: got %v want [2 2]", st.H.Shape)
	}
	if st.C.Shape[0] != 2 || st.C.Shape[1] != 6 {
		t.Fatalf("c shape: got %v want [2 6]", st.C.Shape)
	}
	// Replicate the cell's exact op sequence by hand: h_t must be exactly Whr
	// applied to the unprojected hidden output o*tanh(c), and c_t unprojected.
	const H = 6
	gates := c.Wih.Forward(x).Add(c.Whh.Forward(tensor.Zeros(2, 2)))
	i := sliceCol(gates, 0, H).Sigmoid()
	f := sliceCol(gates, H, 2*H).Sigmoid()
	g := sliceCol(gates, 2*H, 3*H).Tanh()
	o := sliceCol(gates, 3*H, 4*H).Sigmoid()
	cNew := f.Mul(tensor.Zeros(2, H)).Add(i.Mul(g))
	tensorsEqualExact(t, "h == Whr(o*tanh(c))", st.H, c.Whr.Forward(o.Mul(cNew.Tanh())))
	tensorsEqualExact(t, "c unchanged by projection", st.C, cNew)
}

func TestLSTMProjSizeValidation(t *testing.T) {
	mustPanicSub := func(name, wantSub string, f func()) {
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
	mustPanicSub("proj >= hidden", "proj", func() { NewLSTM(3, 4, WithProjSize(4)) })
	mustPanicSub("proj < 0", "proj", func() { NewLSTM(3, 4, WithProjSize(-1)) })
	mustPanicSub("proj on RNN", "LSTM", func() { NewRNN(3, 4, WithProjSize(2)) })
	mustPanicSub("proj on GRU", "LSTM", func() { NewGRU(3, 4, WithProjSize(2)) })
}

// Full gradcheck through a small projected LSTM (parameters incl. Whr, and x).
func TestGradCheckLSTMProjSize(t *testing.T) {
	l := NewLSTM(2, 4, WithProjSize(2))
	x := seededRandn(731, 2, 3, 2).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return l.Forward(x).Square().Mean() }
	gradCheck(t, "LSTM-proj", loss, append(l.Parameters(), x), gcEps, gcTol, 20)
}

// ============================================================================
// PackedSequence
// ============================================================================

// Pack/unpack round-trip: PackSequence pads exactly like PadSequence and
// PadPackedSequence + UnpadSequence recover every original sequence.
func TestPackedSequenceRoundTrip(t *testing.T) {
	seqs := []*tensor.Tensor{
		seededRandn(741, 4, 3),
		seededRandn(742, 2, 3),
		seededRandn(743, 3, 3),
	}
	ps := PackSequence(seqs)
	if !ps.BatchFirst {
		t.Fatal("PackSequence must produce a batch-first PackedSequence")
	}
	tensorsEqualExact(t, "pack == PadSequence", ps.Padded, PadSequence(seqs, true, 0))

	padded, lengths := PadPackedSequence(ps)
	if len(lengths) != 3 || lengths[0] != 4 || lengths[1] != 2 || lengths[2] != 3 {
		t.Fatalf("lengths: got %v want [4 2 3]", lengths)
	}
	back := UnpadSequence(padded, lengths)
	for i := range seqs {
		tensorsEqualExact(t, "unpad", back[i], seqs[i])
	}

	// PackPaddedSequence accepts UNSORTED lengths (enforce_sorted=false
	// semantics) and validates the range.
	ps2 := PackPaddedSequence(padded, []int{4, 2, 3}, true)
	tensorsEqualExact(t, "packpadded", ps2.Padded, padded)
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic for length > Tmax")
			}
		}()
		PackPaddedSequence(padded, []int{5, 2, 3}, true)
	}()
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic for wrong lengths count")
			}
		}()
		PackPaddedSequence(padded, []int{4, 2}, true)
	}()
}

// The key packed guarantee: for a batch with unequal lengths, each sequence's
// ForwardPacked output equals running the module on that sequence ALONE
// (trimmed to its true length), exactly; hN matches the per-sequence final
// state; outputs at t >= length are zero.
func TestForwardPackedPerSequenceEquivalence(t *testing.T) {
	const F, H, L = 3, 4, 2
	seqs := []*tensor.Tensor{
		seededRandn(751, 4, F),
		seededRandn(752, 2, F),
		seededRandn(753, 3, F),
	}
	ps := PackSequence(seqs)
	B, T := 3, 4

	checkOne := func(name string, outPS PackedSequence, hN *tensor.Tensor, hOut int,
		solo func(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor)) {
		t.Helper()
		padded := outPS.Padded
		if padded.Shape[0] != B || padded.Shape[1] != T || padded.Shape[2] != hOut {
			t.Fatalf("%s: packed output shape %v want [%d %d %d]", name, padded.Shape, B, T, hOut)
		}
		for b, s := range seqs {
			Lb := s.Shape[0]
			x := s.Reshape(1, Lb, F)
			soloOut, soloH := solo(x) // (1, Lb, hOut), (L, 1, hOut)
			for tt := 0; tt < T; tt++ {
				for j := 0; j < hOut; j++ {
					got := padded.Data[(b*T+tt)*hOut+j]
					if tt < Lb {
						want := soloOut.Data[(tt)*hOut+j]
						if got != want {
							t.Fatalf("%s: seq %d out[t=%d,j=%d]: got %.17g want %.17g", name, b, tt, j, got, want)
						}
					} else if got != 0 {
						t.Fatalf("%s: seq %d out[t=%d,j=%d]: padded position not zeroed (%g)", name, b, tt, j, got)
					}
				}
			}
			for li := 0; li < L; li++ {
				for j := 0; j < hOut; j++ {
					got := hN.Data[(li*B+b)*hOut+j]
					want := soloH.Data[li*hOut+j]
					if got != want {
						t.Fatalf("%s: seq %d hN[layer=%d,j=%d]: got %.17g want %.17g", name, b, li, j, got, want)
					}
				}
			}
		}
	}

	t.Run("RNN", func(t *testing.T) {
		r := NewRNN(F, H, WithLayers(L))
		outPS, hN := r.ForwardPacked(ps)
		checkOne("RNN", outPS, hN, H, func(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
			return r.ForwardWithState(x, nil)
		})
	})

	t.Run("GRU", func(t *testing.T) {
		g := NewGRU(F, H, WithLayers(L))
		outPS, hN := g.ForwardPacked(ps)
		checkOne("GRU", outPS, hN, H, func(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
			return g.ForwardWithState(x, nil)
		})
	})

	t.Run("LSTM", func(t *testing.T) {
		l := NewLSTM(F, H, WithLayers(L))
		outPS, hN, cN := l.ForwardPacked(ps)
		var soloCs []*tensor.Tensor
		checkOne("LSTM", outPS, hN, H, func(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
			out, h, c := l.ForwardWithState(x, nil, nil)
			soloCs = append(soloCs, c)
			return out, h
		})
		// cN gathered the same way as hN.
		if cN.Shape[0] != L || cN.Shape[1] != B || cN.Shape[2] != H {
			t.Fatalf("cN shape: got %v want [%d %d %d]", cN.Shape, L, B, H)
		}
		for b := range seqs {
			soloC := soloCs[b] // (L, 1, H)
			for li := 0; li < L; li++ {
				for j := 0; j < H; j++ {
					got := cN.Data[(li*B+b)*H+j]
					want := soloC.Data[li*H+j]
					if got != want {
						t.Fatalf("LSTM: seq %d cN[layer=%d,j=%d]: got %.17g want %.17g", b, li, j, got, want)
					}
				}
			}
		}
	})

	t.Run("LSTM-proj", func(t *testing.T) {
		const P = 2
		l := NewLSTM(F, H, WithLayers(L), WithProjSize(P))
		outPS, hN, cN := l.ForwardPacked(ps)
		if cN.Shape[2] != H {
			t.Fatalf("proj cN feature dim: got %d want hidden=%d", cN.Shape[2], H)
		}
		checkOne("LSTM-proj", outPS, hN, P, func(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
			out, h, _ := l.ForwardWithState(x, nil, nil)
			return out, h
		})
	})
}

// A time-first PackedSequence must round-trip through ForwardPacked in the
// time-first layout and match the batch-first result exactly.
func TestForwardPackedTimeFirstLayout(t *testing.T) {
	seqs := []*tensor.Tensor{seededRandn(761, 3, 2), seededRandn(762, 2, 2)}
	bf := PackSequence(seqs)                                                // (B, T, F)
	tf := PackPaddedSequence(bf.Padded.Permute(1, 0, 2), bf.Lengths, false) // (T, B, F)

	g := NewGRU(2, 3)
	outBF, hBF := g.ForwardPacked(bf)
	outTF, hTF := g.ForwardPacked(tf)
	if outTF.BatchFirst {
		t.Fatal("time-first input must produce time-first output")
	}
	tensorsEqualExact(t, "layouts agree", outBF.Padded, outTF.Padded.Permute(1, 0, 2))
	tensorsEqualExact(t, "hN agrees", hBF, hTF)
}

// Gradients must flow through the packed forward back to the (padded) input.
func TestForwardPackedGradFlow(t *testing.T) {
	x := seededRandn(771, 2, 3, 2).SetRequiresGrad(true)
	ps := PackPaddedSequence(x, []int{3, 2}, true)

	l := NewLSTM(2, 3)
	outPS, hN, cN := l.ForwardPacked(ps)
	outPS.Padded.Sum().Add(hN.Sum()).Add(cN.Sum()).Backward()
	if x.Grad == nil {
		t.Fatal("packed input received no gradient")
	}
	// The steps at t >= length of sequence 1 must get zero gradient from the
	// masked output; but t=2 of sequence 0 (valid) must get gradient.
	nonzero := false
	for j := 0; j < 2; j++ {
		if x.Grad.Data[(0*3+2)*2+j] != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("valid final step of sequence 0 received no gradient")
	}
}

// Bidirectional packed input must panic with a clear message (padded storage
// cannot reproduce PyTorch's packed reverse pass).
func TestForwardPackedBidirectionalPanics(t *testing.T) {
	ps := PackSequence([]*tensor.Tensor{seededRandn(781, 3, 2), seededRandn(782, 2, 2)})

	mustPanicBidir := func(name string, f func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: expected panic for bidirectional packed input", name)
			}
			msg, ok := r.(string)
			if !ok || !strings.Contains(msg, "bidirectional packed input is not supported") {
				t.Fatalf("%s: panic %v lacks the bidirectional explanation", name, r)
			}
		}()
		f()
	}

	r := NewRNN(2, 3, WithBidirectional())
	mustPanicBidir("RNN", func() { r.ForwardPacked(ps) })
	l := NewLSTM(2, 3, WithBidirectional())
	mustPanicBidir("LSTM", func() { l.ForwardPacked(ps) })
	g := NewGRU(2, 3, WithBidirectional())
	mustPanicBidir("GRU", func() { g.ForwardPacked(ps) })
}
