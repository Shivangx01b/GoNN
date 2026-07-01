package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// WithReLU must select h = relu(Wih x + Whh h) instead of tanh. Hand-check
// one step against the cell's own linear sublayers.
func TestRNNCellWithReLUHandCheck(t *testing.T) {
	cell := NewRNNCell(3, 4, WithReLU())
	x := seededRandn(501, 2, 3)
	h := seededRandn(502, 2, 4)

	pre := cell.Wih.Forward(x).Add(cell.Whh.Forward(h))
	got := cell.Forward(x, h)
	for i, p := range pre.Data {
		want := math.Max(0, p)
		if got.Data[i] != want {
			t.Fatalf("relu cell data[%d]: got %.17g want %.17g (pre=%g)", i, got.Data[i], want, p)
		}
	}

	// Default stays tanh.
	tc := NewRNNCell(3, 4)
	pre = tc.Wih.Forward(x).Add(tc.Whh.Forward(h))
	got = tc.Forward(x, h)
	for i, p := range pre.Data {
		if got.Data[i] != math.Tanh(p) {
			t.Fatalf("tanh cell data[%d]: got %.17g want %.17g", i, got.Data[i], math.Tanh(p))
		}
	}
}

// WithReLU on the stacked RNN must propagate to every cell and change the
// output (given identical weights).
func TestRNNWithReLU(t *testing.T) {
	relu := NewRNN(3, 4, WithLayers(2), WithBidirectional(), WithReLU())
	for i := range relu.Cells {
		if !relu.Cells[i].ReLUAct || !relu.BackCells[i].ReLUAct {
			t.Fatalf("layer %d cells did not receive WithReLU", i)
		}
	}
	x := seededRandn(511, 2, 3, 3)
	y := relu.Forward(x)
	for i, v := range y.Data {
		if v < 0 {
			t.Fatalf("ReLU RNN output has negative value %g at %d", v, i)
		}
	}
}

// sliceTimeRange extracts x[:, lo:hi, :] as a fresh constant tensor.
func sliceTimeRange(x *tensor.Tensor, lo, hi int) *tensor.Tensor {
	idx := make([]float64, hi-lo)
	for i := range idx {
		idx[i] = float64(lo + i)
	}
	return x.IndexSelect(1, tensor.New(idx, len(idx)))
}

// Feeding hN of the first half as h0 of the second half must reproduce the
// full-sequence Forward exactly (single layer, unidirectional).
func TestRNNForwardWithStateRoundTrip(t *testing.T) {
	x := seededRandn(521, 2, 6, 3)
	x1 := sliceTimeRange(x, 0, 3)
	x2 := sliceTimeRange(x, 3, 6)

	t.Run("RNN", func(t *testing.T) {
		r := NewRNN(3, 4)
		full := r.Forward(x)
		viaState, _ := r.ForwardWithState(x, nil)
		tensorsEqualExact(t, "nil h0 == Forward", full, viaState)

		seq1, h1 := r.ForwardWithState(x1, nil)
		seq2, _ := r.ForwardWithState(x2, h1)
		tensorsEqualExact(t, "RNN split == full", full, tensor.Concat(1, seq1, seq2))
	})

	t.Run("GRU", func(t *testing.T) {
		g := NewGRU(3, 4)
		full := g.Forward(x)
		viaState, _ := g.ForwardWithState(x, nil)
		tensorsEqualExact(t, "nil h0 == Forward", full, viaState)

		seq1, h1 := g.ForwardWithState(x1, nil)
		seq2, _ := g.ForwardWithState(x2, h1)
		tensorsEqualExact(t, "GRU split == full", full, tensor.Concat(1, seq1, seq2))
	})

	t.Run("LSTM", func(t *testing.T) {
		l := NewLSTM(3, 4)
		full := l.Forward(x)
		viaState, _, _ := l.ForwardWithState(x, nil, nil)
		tensorsEqualExact(t, "nil state == Forward", full, viaState)

		seq1, h1, c1 := l.ForwardWithState(x1, nil, nil)
		seq2, _, _ := l.ForwardWithState(x2, h1, c1)
		tensorsEqualExact(t, "LSTM split == full", full, tensor.Concat(1, seq1, seq2))
	})
}

// Bidirectional multi-layer state: shapes are (numLayers*2, B, H) and the
// final states line up with the output sequence of the last layer — forward
// state equals out[:, T-1, :H], backward state equals out[:, 0, H:].
func TestForwardWithStateBidirectionalShapes(t *testing.T) {
	B, T, F, H, L := 2, 5, 3, 4, 2
	x := seededRandn(531, B, T, F)

	r := NewRNN(F, H, WithLayers(L), WithBidirectional())
	seq, hN := r.ForwardWithState(x, nil)
	if hN.Shape[0] != L*2 || hN.Shape[1] != B || hN.Shape[2] != H {
		t.Fatalf("RNN hN shape: got %v want [%d %d %d]", hN.Shape, L*2, B, H)
	}
	for b := 0; b < B; b++ {
		for j := 0; j < H; j++ {
			// forward direction of last layer: index (L-1)*2
			gotF := hN.Data[(((L-1)*2)*B+b)*H+j]
			wantF := seq.Data[(b*T+(T-1))*(2*H)+j]
			if gotF != wantF {
				t.Fatalf("fwd state[%d,%d]: got %.17g want %.17g", b, j, gotF, wantF)
			}
			// backward direction of last layer: index (L-1)*2+1
			gotB := hN.Data[(((L-1)*2+1)*B+b)*H+j]
			wantB := seq.Data[(b*T+0)*(2*H)+H+j]
			if gotB != wantB {
				t.Fatalf("bwd state[%d,%d]: got %.17g want %.17g", b, j, gotB, wantB)
			}
		}
	}

	l := NewLSTM(F, H, WithLayers(L), WithBidirectional())
	_, hL, cL := l.ForwardWithState(x, nil, nil)
	if hL.Shape[0] != L*2 || hL.Shape[1] != B || hL.Shape[2] != H {
		t.Fatalf("LSTM hN shape: got %v", hL.Shape)
	}
	if cL.Shape[0] != L*2 || cL.Shape[1] != B || cL.Shape[2] != H {
		t.Fatalf("LSTM cN shape: got %v", cL.Shape)
	}

	g := NewGRU(F, H, WithBidirectional())
	_, hG := g.ForwardWithState(x, nil)
	if hG.Shape[0] != 2 || hG.Shape[1] != B || hG.Shape[2] != H {
		t.Fatalf("GRU hN shape: got %v", hG.Shape)
	}

	// Wrong h0 shape panics.
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected panic for bad h0 shape")
			}
		}()
		r.ForwardWithState(x, tensor.Zeros(1, B, H))
	}()
}

// Gradients must flow into an explicit h0 (and through hN).
func TestForwardWithStateGradFlow(t *testing.T) {
	x := seededRandn(541, 2, 3, 3)
	h0 := seededRandn(542, 1, 2, 4).SetRequiresGrad(true)

	r := NewRNN(3, 4)
	seq, hN := r.ForwardWithState(x, h0)
	seq.Sum().Add(hN.Sum()).Backward()
	if h0.Grad == nil {
		t.Fatal("h0 received no gradient")
	}
	nonzero := false
	for _, v := range h0.Grad.Data {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("h0 gradient is all zeros")
	}
}

// Full gradcheck through the explicit-state path, including h0.
func TestGradCheckForwardWithState(t *testing.T) {
	x := seededRandn(551, 2, 3, 3).SetRequiresGrad(true)
	h0 := seededRandn(552, 2, 2, 4).SetRequiresGrad(true) // 1 layer * 2 dirs

	r := NewRNN(3, 4, WithBidirectional())
	loss := func() *tensor.Tensor {
		seq, hN := r.ForwardWithState(x, h0)
		return seq.Square().Mean().Add(hN.Square().Mean())
	}
	gradCheck(t, "RNN.ForwardWithState", loss, append(r.Parameters(), x, h0), gcEps, gcTol, 20)
}
