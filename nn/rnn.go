package nn

import (
	"gonn/tensor"
)

// RNN is a single-layer Elman RNN with tanh nonlinearity.
// Input: (batch, seq_len, input_size). Output: (batch, seq_len, hidden_size).
type RNN struct {
	InputSize  int
	HiddenSize int
	Wih        *Linear // x_t -> hidden
	Whh        *Linear // h_{t-1} -> hidden
}

// NewRNN builds a single-layer RNN.
func NewRNN(inputSize, hiddenSize int) *RNN {
	return &RNN{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wih:        NewLinear(inputSize, hiddenSize, true),
		Whh:        NewLinear(hiddenSize, hiddenSize, false),
	}
}

// Forward unrolls the RNN over the sequence dim.
func (r *RNN) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("RNN.Forward: expected (batch, seq, input_size)")
	}
	B, T, _ := x.Shape[0], x.Shape[1], x.Shape[2]
	h := tensor.Zeros(B, r.HiddenSize)
	outs := make([]*tensor.Tensor, T)
	for t := 0; t < T; t++ {
		xt := sliceTime(x, t) // (B, input_size)
		h = r.Wih.Forward(xt).Add(r.Whh.Forward(h)).Tanh()
		outs[t] = h
	}
	return stackTime(outs, B, T, r.HiddenSize)
}

// Parameters returns parameters of Wih and Whh.
func (r *RNN) Parameters() []*tensor.Tensor {
	return append(r.Wih.Parameters(), r.Whh.Parameters()...)
}

// LSTM is a single-layer LSTM.
type LSTM struct {
	InputSize  int
	HiddenSize int
	Wih        *Linear // x -> 4H (i, f, g, o)
	Whh        *Linear // h -> 4H
}

// NewLSTM builds a single-layer LSTM.
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	return &LSTM{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wih:        NewLinear(inputSize, 4*hiddenSize, true),
		Whh:        NewLinear(hiddenSize, 4*hiddenSize, false),
	}
}

// Forward unrolls the LSTM. Returns the hidden-state sequence.
func (l *LSTM) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("LSTM.Forward: expected (batch, seq, input_size)")
	}
	B, T, _ := x.Shape[0], x.Shape[1], x.Shape[2]
	H := l.HiddenSize
	h := tensor.Zeros(B, H)
	c := tensor.Zeros(B, H)
	outs := make([]*tensor.Tensor, T)
	for t := 0; t < T; t++ {
		xt := sliceTime(x, t)
		gates := l.Wih.Forward(xt).Add(l.Whh.Forward(h)) // (B, 4H)
		i := sliceCol(gates, 0, H).Sigmoid()
		f := sliceCol(gates, H, 2*H).Sigmoid()
		g := sliceCol(gates, 2*H, 3*H).Tanh()
		o := sliceCol(gates, 3*H, 4*H).Sigmoid()
		c = f.Mul(c).Add(i.Mul(g))
		h = o.Mul(c.Tanh())
		outs[t] = h
	}
	return stackTime(outs, B, T, H)
}

// Parameters returns Wih + Whh params.
func (l *LSTM) Parameters() []*tensor.Tensor {
	return append(l.Wih.Parameters(), l.Whh.Parameters()...)
}

// GRU is a single-layer GRU.
type GRU struct {
	InputSize  int
	HiddenSize int
	Wih        *Linear // x -> 3H (r, z, n)
	Whh        *Linear // h -> 3H
}

// NewGRU builds a single-layer GRU.
func NewGRU(inputSize, hiddenSize int) *GRU {
	return &GRU{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wih:        NewLinear(inputSize, 3*hiddenSize, true),
		Whh:        NewLinear(hiddenSize, 3*hiddenSize, true),
	}
}

// Forward unrolls the GRU. Returns the hidden-state sequence.
func (g *GRU) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("GRU.Forward: expected (batch, seq, input_size)")
	}
	B, T, _ := x.Shape[0], x.Shape[1], x.Shape[2]
	H := g.HiddenSize
	h := tensor.Zeros(B, H)
	outs := make([]*tensor.Tensor, T)
	one := tensor.Scalar(1)
	for t := 0; t < T; t++ {
		xt := sliceTime(x, t)
		xg := g.Wih.Forward(xt)
		hg := g.Whh.Forward(h)
		rGate := sliceCol(xg, 0, H).Add(sliceCol(hg, 0, H)).Sigmoid()
		zGate := sliceCol(xg, H, 2*H).Add(sliceCol(hg, H, 2*H)).Sigmoid()
		nGate := sliceCol(xg, 2*H, 3*H).Add(rGate.Mul(sliceCol(hg, 2*H, 3*H))).Tanh()
		h = one.Sub(zGate).Mul(nGate).Add(zGate.Mul(h))
		outs[t] = h
	}
	return stackTime(outs, B, T, H)
}

// Parameters returns Wih + Whh params.
func (g *GRU) Parameters() []*tensor.Tensor {
	return append(g.Wih.Parameters(), g.Whh.Parameters()...)
}

// ============================================================================
// Single-timestep cells. These let callers drive the recurrence themselves,
// which is useful for multi-layer/bidirectional stacks and seq2seq decoders.
// ============================================================================

// RNNCell is a single Elman RNN step: h_t = tanh(W_ih x + W_hh h_{t-1}).
type RNNCell struct {
	InputSize  int
	HiddenSize int
	Wih        *Linear
	Whh        *Linear
}

// NewRNNCell creates an RNNCell.
func NewRNNCell(in, hidden int) *RNNCell {
	return &RNNCell{
		InputSize:  in,
		HiddenSize: hidden,
		Wih:        NewLinear(in, hidden, true),
		Whh:        NewLinear(hidden, hidden, false),
	}
}

// Forward consumes one timestep. x: (B, in); h: (B, hidden) (use zeros for t=0).
// Returns new h of shape (B, hidden).
func (c *RNNCell) Forward(x, h *tensor.Tensor) *tensor.Tensor {
	if h == nil {
		h = tensor.Zeros(x.Shape[0], c.HiddenSize)
	}
	return c.Wih.Forward(x).Add(c.Whh.Forward(h)).Tanh()
}

// Parameters returns the cell's parameters.
func (c *RNNCell) Parameters() []*tensor.Tensor {
	return append(c.Wih.Parameters(), c.Whh.Parameters()...)
}

// LSTMState bundles the hidden and cell state of an LSTM cell.
type LSTMState struct {
	H *tensor.Tensor
	C *tensor.Tensor
}

// LSTMCell is a single LSTM step.
type LSTMCell struct {
	InputSize  int
	HiddenSize int
	Wih        *Linear // x -> 4H (i, f, g, o)
	Whh        *Linear // h -> 4H
}

// NewLSTMCell creates an LSTMCell.
func NewLSTMCell(in, hidden int) *LSTMCell {
	return &LSTMCell{
		InputSize:  in,
		HiddenSize: hidden,
		Wih:        NewLinear(in, 4*hidden, true),
		Whh:        NewLinear(hidden, 4*hidden, false),
	}
}

// Forward consumes one timestep. x: (B, in); state may be nil for t=0.
// Returns new state (h_t, c_t).
func (c *LSTMCell) Forward(x *tensor.Tensor, state *LSTMState) *LSTMState {
	B := x.Shape[0]
	H := c.HiddenSize
	if state == nil {
		state = &LSTMState{H: tensor.Zeros(B, H), C: tensor.Zeros(B, H)}
	}
	gates := c.Wih.Forward(x).Add(c.Whh.Forward(state.H)) // (B, 4H)
	i := sliceCol(gates, 0, H).Sigmoid()
	f := sliceCol(gates, H, 2*H).Sigmoid()
	g := sliceCol(gates, 2*H, 3*H).Tanh()
	o := sliceCol(gates, 3*H, 4*H).Sigmoid()
	cNew := f.Mul(state.C).Add(i.Mul(g))
	hNew := o.Mul(cNew.Tanh())
	return &LSTMState{H: hNew, C: cNew}
}

// Parameters returns the cell's parameters.
func (c *LSTMCell) Parameters() []*tensor.Tensor {
	return append(c.Wih.Parameters(), c.Whh.Parameters()...)
}

// GRUCell is a single GRU step.
type GRUCell struct {
	InputSize  int
	HiddenSize int
	Wih        *Linear // x -> 3H (r, z, n)
	Whh        *Linear // h -> 3H
}

// NewGRUCell creates a GRUCell.
func NewGRUCell(in, hidden int) *GRUCell {
	return &GRUCell{
		InputSize:  in,
		HiddenSize: hidden,
		Wih:        NewLinear(in, 3*hidden, true),
		Whh:        NewLinear(hidden, 3*hidden, true),
	}
}

// Forward consumes one timestep. x: (B, in); h: (B, hidden) or nil.
// Returns new h.
func (c *GRUCell) Forward(x, h *tensor.Tensor) *tensor.Tensor {
	H := c.HiddenSize
	if h == nil {
		h = tensor.Zeros(x.Shape[0], H)
	}
	one := tensor.Scalar(1)
	xg := c.Wih.Forward(x)
	hg := c.Whh.Forward(h)
	rGate := sliceCol(xg, 0, H).Add(sliceCol(hg, 0, H)).Sigmoid()
	zGate := sliceCol(xg, H, 2*H).Add(sliceCol(hg, H, 2*H)).Sigmoid()
	nGate := sliceCol(xg, 2*H, 3*H).Add(rGate.Mul(sliceCol(hg, 2*H, 3*H))).Tanh()
	return one.Sub(zGate).Mul(nGate).Add(zGate.Mul(h))
}

// Parameters returns the cell's parameters.
func (c *GRUCell) Parameters() []*tensor.Tensor {
	return append(c.Wih.Parameters(), c.Whh.Parameters()...)
}

// sliceTime returns x[:, t, :] for a (B, T, F) tensor as (B, F).
// Implemented as a gather-style matmul so autograd is preserved.
func sliceTime(x *tensor.Tensor, t int) *tensor.Tensor {
	B, T, F := x.Shape[0], x.Shape[1], x.Shape[2]
	// Build selector S of shape (1, T) with 1 at position t.
	sel := tensor.Zeros(1, T)
	sel.Data[t] = 1
	// reshape x to (B, T, F) -> view as (B*T, F) and gather via row selection per batch.
	// Use the fact x[:, t, :] = einsum('btf,t->bf', x, sel_row).
	// We can implement as: x reshaped to (B, T*F), times a (T*F, F) selector.
	// Easier: build a (B*T, B) one-hot? No. Use this: reshape x to (B, T, F),
	// permute to (B, F, T), then matmul with sel^T (T,1) -> (B, F, 1) -> (B, F).
	xPerm := x.Permute(0, 2, 1).Reshape(B*F, T) // (B*F, T)
	out := xPerm.MatMul(sel.Reshape(T, 1))      // (B*F, 1)
	return out.Reshape(B, F)
}

// stackTime stacks T tensors of shape (B, H) into (B, T, H) using broadcasted
// multiplies and adds so autograd works without relying on Concat.
func stackTime(hs []*tensor.Tensor, B, T, H int) *tensor.Tensor {
	out := tensor.Zeros(B, T, H)
	for t, h := range hs {
		// selector e of shape (1, T, 1) with 1 at position t.
		sel := tensor.Zeros(1, T, 1)
		sel.Data[t] = 1
		placed := h.Reshape(B, 1, H).Mul(sel) // (B, T, H)
		out = out.Add(placed)
	}
	return out
}

// sliceCol returns x[:, lo:hi] for a 2D tensor (B, N).
func sliceCol(x *tensor.Tensor, lo, hi int) *tensor.Tensor {
	B, N := x.Shape[0], x.Shape[1]
	w := hi - lo
	// Build a (N, w) selector: identity rows from lo..hi-1.
	sel := tensor.Zeros(N, w)
	for j := 0; j < w; j++ {
		sel.Data[(lo+j)*w+j] = 1
	}
	_ = B
	return x.MatMul(sel)
}
