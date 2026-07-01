package nn

import (
	"fmt"
	"strconv"

	"gonn/tensor"
)

// Recurrent layers. The single-timestep cells (RNNCell/LSTMCell/GRUCell) are
// the weight-owning primitives; RNN/LSTM/GRU stack them into (optionally
// bidirectional) multi-layer networks. The default configuration — one
// layer, unidirectional — matches the historical single-layer types, with
// identical weight layouts and RNG draw order.
//
// All layer variants take input of shape (B, T, F) and return (B, T, H_out)
// where H_out = HiddenSize if unidirectional, else 2*HiddenSize.

// RNNOpt configures RNN/LSTM/GRU stacks.
type RNNOpt func(*rnnOpts)

type rnnOpts struct {
	layers int
	bidir  bool
}

// WithLayers sets the number of stacked layers (default 1).
func WithLayers(n int) RNNOpt { return func(o *rnnOpts) { o.layers = n } }

// WithBidirectional adds a reverse-direction cell per layer; the two streams
// are concatenated along the feature axis.
func WithBidirectional() RNNOpt { return func(o *rnnOpts) { o.bidir = true } }

func resolveRNNOpts(opts []RNNOpt) rnnOpts {
	o := rnnOpts{layers: 1}
	for _, fn := range opts {
		fn(&o)
	}
	if o.layers < 1 {
		panic(fmt.Sprintf("nn: rnn layers must be >= 1, got %d", o.layers))
	}
	return o
}

// layerInputSize returns the input size of stack layer i.
func layerInputSize(i, in, hidden int, bidir bool) int {
	if i == 0 {
		return in
	}
	if bidir {
		return 2 * hidden
	}
	return hidden
}

// ============================================================================
// Single-timestep cells
// ============================================================================

// RNNCell is a single Elman RNN step: h_t = tanh(W_ih x + W_hh h_{t-1}).
type RNNCell struct {
	Base
	InputSize  int
	HiddenSize int
	Wih        *Linear
	Whh        *Linear
}

// NewRNNCell creates an RNNCell.
func NewRNNCell(in, hidden int) *RNNCell {
	c := &RNNCell{
		InputSize:  in,
		HiddenSize: hidden,
		Wih:        NewLinear(in, hidden, true),
		Whh:        NewLinear(hidden, hidden, false),
	}
	c.regChild("wih", c.Wih)
	c.regChild("whh", c.Whh)
	return c
}

// Forward consumes one timestep. x: (B, in); h: (B, hidden) (use nil for t=0).
// Returns new h of shape (B, hidden).
func (c *RNNCell) Forward(x, h *tensor.Tensor) *tensor.Tensor {
	if h == nil {
		h = tensor.Zeros(x.Shape[0], c.HiddenSize)
	}
	return c.Wih.Forward(x).Add(c.Whh.Forward(h)).Tanh()
}

// LSTMState bundles the hidden and cell state of an LSTM cell.
type LSTMState struct {
	H *tensor.Tensor
	C *tensor.Tensor
}

// LSTMCell is a single LSTM step.
type LSTMCell struct {
	Base
	InputSize  int
	HiddenSize int
	Wih        *Linear // x -> 4H (i, f, g, o)
	Whh        *Linear // h -> 4H
}

// NewLSTMCell creates an LSTMCell.
func NewLSTMCell(in, hidden int) *LSTMCell {
	c := &LSTMCell{
		InputSize:  in,
		HiddenSize: hidden,
		Wih:        NewLinear(in, 4*hidden, true),
		Whh:        NewLinear(hidden, 4*hidden, false),
	}
	c.regChild("wih", c.Wih)
	c.regChild("whh", c.Whh)
	return c
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

// GRUCell is a single GRU step.
type GRUCell struct {
	Base
	InputSize  int
	HiddenSize int
	Wih        *Linear // x -> 3H (r, z, n)
	Whh        *Linear // h -> 3H
}

// NewGRUCell creates a GRUCell.
func NewGRUCell(in, hidden int) *GRUCell {
	c := &GRUCell{
		InputSize:  in,
		HiddenSize: hidden,
		Wih:        NewLinear(in, 3*hidden, true),
		Whh:        NewLinear(hidden, 3*hidden, true),
	}
	c.regChild("wih", c.Wih)
	c.regChild("whh", c.Whh)
	return c
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

// ============================================================================
// Direction runners
// ============================================================================

// runRNNDir runs one RNNCell over x (B, T, F) in either time order. (B,T,H).
func runRNNDir(cell *RNNCell, x *tensor.Tensor, reverse bool) *tensor.Tensor {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	var h *tensor.Tensor
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		h = cell.Forward(sliceTime(x, t), h)
		outs[t] = h
	}
	return stackTime(outs, B, T, cell.HiddenSize)
}

// runLSTMDir runs one LSTMCell over x in either direction. (B,T,H).
func runLSTMDir(cell *LSTMCell, x *tensor.Tensor, reverse bool) *tensor.Tensor {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	var state *LSTMState
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		state = cell.Forward(sliceTime(x, t), state)
		outs[t] = state.H
	}
	return stackTime(outs, B, T, cell.HiddenSize)
}

// runGRUDir runs one GRUCell over x in either direction. (B,T,H).
func runGRUDir(cell *GRUCell, x *tensor.Tensor, reverse bool) *tensor.Tensor {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	var h *tensor.Tensor
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		h = cell.Forward(sliceTime(x, t), h)
		outs[t] = h
	}
	return stackTime(outs, B, T, cell.HiddenSize)
}

// ============================================================================
// Stacked (optionally bidirectional) layers
// ============================================================================

// RNN is a stack of Elman RNN layers (tanh nonlinearity), optionally
// bidirectional. Default: one layer, unidirectional.
type RNN struct {
	Base
	Cells         []*RNNCell // forward direction, one per layer
	BackCells     []*RNNCell // reverse direction (nil if !Bidirectional)
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewRNN builds an RNN stack: NewRNN(in, hidden), or with options
// NewRNN(in, hidden, WithLayers(2), WithBidirectional()).
func NewRNN(in, hidden int, opts ...RNNOpt) *RNN {
	o := resolveRNNOpts(opts)
	r := &RNN{NumLayers: o.layers, HiddenSize: hidden, Bidirectional: o.bidir}
	r.Cells = make([]*RNNCell, o.layers)
	if o.bidir {
		r.BackCells = make([]*RNNCell, o.layers)
	}
	for i := 0; i < o.layers; i++ {
		layerIn := layerInputSize(i, in, hidden, o.bidir)
		r.Cells[i] = NewRNNCell(layerIn, hidden)
		if o.bidir {
			r.BackCells[i] = NewRNNCell(layerIn, hidden)
		}
	}
	for i, c := range r.Cells {
		r.regChild("cells."+strconv.Itoa(i), c)
	}
	for i, c := range r.BackCells {
		r.regChild("backcells."+strconv.Itoa(i), c)
	}
	return r
}

// Forward runs the stack. x: (B, T, F). Returns (B, T, H) or (B, T, 2H).
func (r *RNN) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("RNN.Forward: expected (B, T, F)")
	}
	cur := x
	for li := 0; li < r.NumLayers; li++ {
		fwd := runRNNDir(r.Cells[li], cur, false)
		if !r.Bidirectional {
			cur = fwd
			continue
		}
		bwd := runRNNDir(r.BackCells[li], cur, true)
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur
}

// LSTM is a stack of LSTM layers, optionally bidirectional.
// Default: one layer, unidirectional.
type LSTM struct {
	Base
	Cells         []*LSTMCell
	BackCells     []*LSTMCell
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewLSTM builds an LSTM stack.
func NewLSTM(in, hidden int, opts ...RNNOpt) *LSTM {
	o := resolveRNNOpts(opts)
	l := &LSTM{NumLayers: o.layers, HiddenSize: hidden, Bidirectional: o.bidir}
	l.Cells = make([]*LSTMCell, o.layers)
	if o.bidir {
		l.BackCells = make([]*LSTMCell, o.layers)
	}
	for i := 0; i < o.layers; i++ {
		layerIn := layerInputSize(i, in, hidden, o.bidir)
		l.Cells[i] = NewLSTMCell(layerIn, hidden)
		if o.bidir {
			l.BackCells[i] = NewLSTMCell(layerIn, hidden)
		}
	}
	for i, c := range l.Cells {
		l.regChild("cells."+strconv.Itoa(i), c)
	}
	for i, c := range l.BackCells {
		l.regChild("backcells."+strconv.Itoa(i), c)
	}
	return l
}

// Forward runs the stack. x: (B, T, F).
func (l *LSTM) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("LSTM.Forward: expected (B, T, F)")
	}
	cur := x
	for li := 0; li < l.NumLayers; li++ {
		fwd := runLSTMDir(l.Cells[li], cur, false)
		if !l.Bidirectional {
			cur = fwd
			continue
		}
		bwd := runLSTMDir(l.BackCells[li], cur, true)
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur
}

// GRU is a stack of GRU layers, optionally bidirectional.
// Default: one layer, unidirectional.
type GRU struct {
	Base
	Cells         []*GRUCell
	BackCells     []*GRUCell
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewGRU builds a GRU stack.
func NewGRU(in, hidden int, opts ...RNNOpt) *GRU {
	o := resolveRNNOpts(opts)
	g := &GRU{NumLayers: o.layers, HiddenSize: hidden, Bidirectional: o.bidir}
	g.Cells = make([]*GRUCell, o.layers)
	if o.bidir {
		g.BackCells = make([]*GRUCell, o.layers)
	}
	for i := 0; i < o.layers; i++ {
		layerIn := layerInputSize(i, in, hidden, o.bidir)
		g.Cells[i] = NewGRUCell(layerIn, hidden)
		if o.bidir {
			g.BackCells[i] = NewGRUCell(layerIn, hidden)
		}
	}
	for i, c := range g.Cells {
		g.regChild("cells."+strconv.Itoa(i), c)
	}
	for i, c := range g.BackCells {
		g.regChild("backcells."+strconv.Itoa(i), c)
	}
	return g
}

// Forward runs the stack. x: (B, T, F).
func (g *GRU) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("GRU.Forward: expected (B, T, F)")
	}
	cur := x
	for li := 0; li < g.NumLayers; li++ {
		fwd := runGRUDir(g.Cells[li], cur, false)
		if !g.Bidirectional {
			cur = fwd
			continue
		}
		bwd := runGRUDir(g.BackCells[li], cur, true)
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur
}
