package nn

import (
	"gonn/tensor"
)

// ============================================================================
// Multi-layer + (optionally) bidirectional recurrent stacks.
//
// All variants take input of shape (B, T, F) and return (B, T, H_out) where
// H_out = HiddenSize if unidirectional, else 2*HiddenSize. The per-direction
// hidden size is the same.
//
// Bidirectional: a second set of cells processes the time-reversed sequence
// and the two streams are concatenated along the feature axis at each step.
// Stacking is "all layers see the full sequence of the previous layer".
// ============================================================================

// runRNNDir runs a stack of RNNCells over x (B, T, F) in either forward or
// reverse time order. Returns (B, T, H).
func runRNNDir(cells []*RNNCell, x *tensor.Tensor, reverse bool) *tensor.Tensor {
	B, T, _ := x.Shape[0], x.Shape[1], x.Shape[2]
	H := cells[0].HiddenSize
	// Drive the lowest layer over the input; each subsequent layer consumes
	// the previous layer's hidden-state sequence.
	cur := x
	for li, cell := range cells {
		outs := make([]*tensor.Tensor, T)
		var h *tensor.Tensor
		for step := 0; step < T; step++ {
			t := step
			if reverse {
				t = T - 1 - step
			}
			xt := sliceTime(cur, t)
			h = cell.Forward(xt, h)
			outs[t] = h
		}
		cur = stackTime(outs, B, T, H)
		_ = li
	}
	return cur
}

// MultiLayerRNN is a stack of Elman RNN layers, optionally bidirectional.
type MultiLayerRNN struct {
	Cells         []*RNNCell // forward direction, one per layer
	BackCells     []*RNNCell // reverse direction, one per layer (nil if !Bidirectional)
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewMultiLayerRNN builds a stack. The first layer takes input of size `in`;
// subsequent layers take the previous layer's output (which is `hidden`
// unidirectional or `2*hidden` bidirectional).
func NewMultiLayerRNN(in, hidden, numLayers int, bidir bool) *MultiLayerRNN {
	cells := make([]*RNNCell, numLayers)
	var back []*RNNCell
	if bidir {
		back = make([]*RNNCell, numLayers)
	}
	for i := 0; i < numLayers; i++ {
		layerIn := in
		if i > 0 {
			layerIn = hidden
			if bidir {
				layerIn = 2 * hidden
			}
		}
		cells[i] = NewRNNCell(layerIn, hidden)
		if bidir {
			back[i] = NewRNNCell(layerIn, hidden)
		}
	}
	return &MultiLayerRNN{
		Cells: cells, BackCells: back,
		NumLayers: numLayers, HiddenSize: hidden, Bidirectional: bidir,
	}
}

// Forward runs the stack. x: (B, T, F). Returns (B, T, H) or (B, T, 2H).
func (r *MultiLayerRNN) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("MultiLayerRNN.Forward: expected (B, T, F)")
	}
	B, T := x.Shape[0], x.Shape[1]
	cur := x
	for li := 0; li < r.NumLayers; li++ {
		fwd := runRNNDir(r.Cells[li:li+1], cur, false) // (B, T, H)
		if !r.Bidirectional {
			cur = fwd
			continue
		}
		bwd := runRNNDir(r.BackCells[li:li+1], cur, true) // (B, T, H)
		cur = concatFeature(fwd, bwd, B, T, r.HiddenSize, r.HiddenSize)
	}
	return cur
}

// Parameters returns all cell parameters.
func (r *MultiLayerRNN) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, c := range r.Cells {
		ps = append(ps, c.Parameters()...)
	}
	for _, c := range r.BackCells {
		ps = append(ps, c.Parameters()...)
	}
	return ps
}

// ----------------------------------------------------------------------------
// MultiLayerLSTM
// ----------------------------------------------------------------------------

// runLSTMDir runs a single LSTM layer over x in either direction. Returns (B,T,H).
func runLSTMDir(cell *LSTMCell, x *tensor.Tensor, reverse bool) *tensor.Tensor {
	B, T, _ := x.Shape[0], x.Shape[1], x.Shape[2]
	H := cell.HiddenSize
	outs := make([]*tensor.Tensor, T)
	var state *LSTMState
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		xt := sliceTime(x, t)
		state = cell.Forward(xt, state)
		outs[t] = state.H
	}
	return stackTime(outs, B, T, H)
}

// MultiLayerLSTM is a stack of LSTM layers, optionally bidirectional.
type MultiLayerLSTM struct {
	Cells         []*LSTMCell
	BackCells     []*LSTMCell
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewMultiLayerLSTM builds a stack.
func NewMultiLayerLSTM(in, hidden, numLayers int, bidir bool) *MultiLayerLSTM {
	cells := make([]*LSTMCell, numLayers)
	var back []*LSTMCell
	if bidir {
		back = make([]*LSTMCell, numLayers)
	}
	for i := 0; i < numLayers; i++ {
		layerIn := in
		if i > 0 {
			layerIn = hidden
			if bidir {
				layerIn = 2 * hidden
			}
		}
		cells[i] = NewLSTMCell(layerIn, hidden)
		if bidir {
			back[i] = NewLSTMCell(layerIn, hidden)
		}
	}
	return &MultiLayerLSTM{
		Cells: cells, BackCells: back,
		NumLayers: numLayers, HiddenSize: hidden, Bidirectional: bidir,
	}
}

// Forward runs the stack. x: (B, T, F).
func (l *MultiLayerLSTM) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("MultiLayerLSTM.Forward: expected (B, T, F)")
	}
	B, T := x.Shape[0], x.Shape[1]
	cur := x
	for li := 0; li < l.NumLayers; li++ {
		fwd := runLSTMDir(l.Cells[li], cur, false)
		if !l.Bidirectional {
			cur = fwd
			continue
		}
		bwd := runLSTMDir(l.BackCells[li], cur, true)
		cur = concatFeature(fwd, bwd, B, T, l.HiddenSize, l.HiddenSize)
	}
	return cur
}

// Parameters returns all cell parameters.
func (l *MultiLayerLSTM) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, c := range l.Cells {
		ps = append(ps, c.Parameters()...)
	}
	for _, c := range l.BackCells {
		ps = append(ps, c.Parameters()...)
	}
	return ps
}

// ----------------------------------------------------------------------------
// MultiLayerGRU
// ----------------------------------------------------------------------------

// runGRUDir runs a single GRU layer over x in either direction.
func runGRUDir(cell *GRUCell, x *tensor.Tensor, reverse bool) *tensor.Tensor {
	B, T, _ := x.Shape[0], x.Shape[1], x.Shape[2]
	H := cell.HiddenSize
	outs := make([]*tensor.Tensor, T)
	var h *tensor.Tensor
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		xt := sliceTime(x, t)
		h = cell.Forward(xt, h)
		outs[t] = h
	}
	return stackTime(outs, B, T, H)
}

// MultiLayerGRU is a stack of GRU layers, optionally bidirectional.
type MultiLayerGRU struct {
	Cells         []*GRUCell
	BackCells     []*GRUCell
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewMultiLayerGRU builds a stack.
func NewMultiLayerGRU(in, hidden, numLayers int, bidir bool) *MultiLayerGRU {
	cells := make([]*GRUCell, numLayers)
	var back []*GRUCell
	if bidir {
		back = make([]*GRUCell, numLayers)
	}
	for i := 0; i < numLayers; i++ {
		layerIn := in
		if i > 0 {
			layerIn = hidden
			if bidir {
				layerIn = 2 * hidden
			}
		}
		cells[i] = NewGRUCell(layerIn, hidden)
		if bidir {
			back[i] = NewGRUCell(layerIn, hidden)
		}
	}
	return &MultiLayerGRU{
		Cells: cells, BackCells: back,
		NumLayers: numLayers, HiddenSize: hidden, Bidirectional: bidir,
	}
}

// Forward runs the stack. x: (B, T, F).
func (g *MultiLayerGRU) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("MultiLayerGRU.Forward: expected (B, T, F)")
	}
	B, T := x.Shape[0], x.Shape[1]
	cur := x
	for li := 0; li < g.NumLayers; li++ {
		fwd := runGRUDir(g.Cells[li], cur, false)
		if !g.Bidirectional {
			cur = fwd
			continue
		}
		bwd := runGRUDir(g.BackCells[li], cur, true)
		cur = concatFeature(fwd, bwd, B, T, g.HiddenSize, g.HiddenSize)
	}
	return cur
}

// Parameters returns all cell parameters.
func (g *MultiLayerGRU) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, c := range g.Cells {
		ps = append(ps, c.Parameters()...)
	}
	for _, c := range g.BackCells {
		ps = append(ps, c.Parameters()...)
	}
	return ps
}

// ----------------------------------------------------------------------------
// concatFeature: concatenate two (B, T, H1) and (B, T, H2) tensors along the
// feature axis, preserving autograd. tensor.Concat is autograd-aware, so the
// forward/backward directions both receive gradients through the join.
// ----------------------------------------------------------------------------

func concatFeature(a, b *tensor.Tensor, B, T, H1, H2 int) *tensor.Tensor {
	return tensor.Concat(2, a, b)
}
