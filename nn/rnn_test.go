package nn

import (
	"testing"

	"gonn/tensor"
)

// ----- single-step cells -----

func TestRNNCellForwardShape(t *testing.T) {
	cell := NewRNNCell(4, 5)
	x := tensor.Randn(2, 4)
	h := cell.Forward(x, nil)
	if h.Shape[0] != 2 || h.Shape[1] != 5 {
		t.Fatalf("RNNCell shape: got %v want [2 5]", h.Shape)
	}
}

func TestLSTMCellForwardShape(t *testing.T) {
	cell := NewLSTMCell(4, 5)
	x := tensor.Randn(2, 4)
	st := cell.Forward(x, nil)
	if st.H.Shape[0] != 2 || st.H.Shape[1] != 5 {
		t.Fatalf("LSTMCell H shape: got %v want [2 5]", st.H.Shape)
	}
	if st.C.Shape[0] != 2 || st.C.Shape[1] != 5 {
		t.Fatalf("LSTMCell C shape: got %v want [2 5]", st.C.Shape)
	}
	// step it again with the returned state.
	st2 := cell.Forward(x, st)
	if st2.H.Shape[0] != 2 || st2.H.Shape[1] != 5 {
		t.Fatalf("LSTMCell step2 shape: got %v", st2.H.Shape)
	}
}

func TestGRUCellForwardShape(t *testing.T) {
	cell := NewGRUCell(4, 5)
	x := tensor.Randn(2, 4)
	h := cell.Forward(x, nil)
	if h.Shape[0] != 2 || h.Shape[1] != 5 {
		t.Fatalf("GRUCell shape: got %v want [2 5]", h.Shape)
	}
}

func TestRNNCellBackpropPopulatesGrads(t *testing.T) {
	cell := NewRNNCell(3, 4)
	x := tensor.Randn(2, 3)
	h := cell.Forward(x, nil)
	loss := h.Sum()
	loss.Backward()
	for i, p := range cell.Parameters() {
		if p.Grad == nil {
			t.Fatalf("RNNCell param %d has no grad", i)
		}
	}
}

// ----- multi-layer / bidirectional -----

func TestMultiLayerRNNShape(t *testing.T) {
	for _, bidir := range []bool{false, true} {
		opts := []RNNOpt{WithLayers(2)}
		if bidir {
			opts = append(opts, WithBidirectional())
		}
		m := NewRNN(4, 6, opts...)
		x := tensor.Randn(3, 5, 4)
		y := m.Forward(x)
		wantH := 6
		if bidir {
			wantH = 12
		}
		if y.Shape[0] != 3 || y.Shape[1] != 5 || y.Shape[2] != wantH {
			t.Fatalf("MultiLayerRNN bidir=%v shape: got %v want [3 5 %d]", bidir, y.Shape, wantH)
		}
	}
}

func TestMultiLayerLSTMShape(t *testing.T) {
	for _, bidir := range []bool{false, true} {
		opts := []RNNOpt{WithLayers(2)}
		if bidir {
			opts = append(opts, WithBidirectional())
		}
		m := NewLSTM(4, 6, opts...)
		x := tensor.Randn(3, 5, 4)
		y := m.Forward(x)
		wantH := 6
		if bidir {
			wantH = 12
		}
		if y.Shape[0] != 3 || y.Shape[1] != 5 || y.Shape[2] != wantH {
			t.Fatalf("MultiLayerLSTM bidir=%v shape: got %v want [3 5 %d]", bidir, y.Shape, wantH)
		}
	}
}

func TestMultiLayerGRUShape(t *testing.T) {
	for _, bidir := range []bool{false, true} {
		opts := []RNNOpt{WithLayers(2)}
		if bidir {
			opts = append(opts, WithBidirectional())
		}
		m := NewGRU(4, 6, opts...)
		x := tensor.Randn(3, 5, 4)
		y := m.Forward(x)
		wantH := 6
		if bidir {
			wantH = 12
		}
		if y.Shape[0] != 3 || y.Shape[1] != 5 || y.Shape[2] != wantH {
			t.Fatalf("MultiLayerGRU bidir=%v shape: got %v want [3 5 %d]", bidir, y.Shape, wantH)
		}
	}
}

func TestMultiLayerLSTMBackpropPopulatesGrads(t *testing.T) {
	m := NewLSTM(3, 4, WithLayers(2), WithBidirectional())
	x := tensor.Randn(2, 3, 3)
	y := m.Forward(x)
	loss := y.Sum()
	loss.Backward()
	for i, p := range m.Parameters() {
		if p.Grad == nil {
			t.Fatalf("MultiLayerLSTM param %d has no grad", i)
		}
	}
}

// ----- seq2seq -----

func TestSeq2SeqForwardShape(t *testing.T) {
	srcVocab, tgtVocab, embed, hidden := 7, 9, 4, 5
	m := NewSeq2Seq(srcVocab, tgtVocab, embed, hidden)
	// (B=2, T_src=3) of token ids
	srcIdx := tensor.New([]float64{1, 2, 3, 0, 4, 5}, 2, 3)
	tgtIdx := tensor.New([]float64{6, 1, 0, 2, 3, 4, 5, 7}, 2, 4) // T_tgt=4
	logits := m.Forward(srcIdx, tgtIdx)
	if logits.Shape[0] != 2 || logits.Shape[1] != 4 || logits.Shape[2] != tgtVocab {
		t.Fatalf("Seq2Seq logits shape: got %v want [2 4 %d]", logits.Shape, tgtVocab)
	}
}

func TestSeq2SeqBackpropPopulatesGrads(t *testing.T) {
	m := NewSeq2Seq(5, 6, 3, 4)
	srcIdx := tensor.New([]float64{1, 2, 0, 3, 4, 1}, 2, 3)
	tgtIdx := tensor.New([]float64{0, 1, 2, 3, 4, 5}, 2, 3)
	logits := m.Forward(srcIdx, tgtIdx)
	// Flatten (B*T_tgt, V) and use CE against a dummy target.
	B, T, V := logits.Shape[0], logits.Shape[1], logits.Shape[2]
	flat := logits.Reshape(B*T, V)
	tgt := tensor.New([]float64{0, 1, 2, 3, 4, 5}, B*T)
	loss := CrossEntropyLoss(flat, tgt)
	loss.Backward()
	for i, p := range m.Parameters() {
		if p.Grad == nil {
			t.Fatalf("Seq2Seq param %d has no grad", i)
		}
	}
}
