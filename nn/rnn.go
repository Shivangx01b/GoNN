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
	relu   bool
}

// WithLayers sets the number of stacked layers (default 1).
func WithLayers(n int) RNNOpt { return func(o *rnnOpts) { o.layers = n } }

// WithBidirectional adds a reverse-direction cell per layer; the two streams
// are concatenated along the feature axis.
func WithBidirectional() RNNOpt { return func(o *rnnOpts) { o.bidir = true } }

// WithReLU selects the ReLU nonlinearity instead of tanh for RNN/RNNCell
// (PyTorch nonlinearity='relu'). It has no effect on LSTM/GRU, whose gate
// nonlinearities are fixed.
func WithReLU() RNNOpt { return func(o *rnnOpts) { o.relu = true } }

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

// RNNCell is a single Elman RNN step: h_t = act(W_ih x + W_hh h_{t-1}),
// where act is tanh by default or ReLU with WithReLU().
type RNNCell struct {
	Base
	InputSize  int
	HiddenSize int
	// ReLUAct selects h_t = relu(...) instead of tanh (PyTorch
	// nonlinearity='relu'). Zero value (tanh) preserves historical behavior.
	ReLUAct bool
	Wih     *Linear
	Whh     *Linear
}

// NewRNNCell creates an RNNCell. NewRNNCell(in, hidden) defaults to tanh;
// pass WithReLU() for the ReLU nonlinearity. WithLayers/WithBidirectional do
// not apply to a single cell and are ignored.
func NewRNNCell(in, hidden int, opts ...RNNOpt) *RNNCell {
	o := resolveRNNOpts(opts)
	c := &RNNCell{
		InputSize:  in,
		HiddenSize: hidden,
		ReLUAct:    o.relu,
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
	pre := c.Wih.Forward(x).Add(c.Whh.Forward(h))
	if c.ReLUAct {
		return pre.ReLU()
	}
	return pre.Tanh()
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

// runRNNDir runs one RNNCell over x (B, T, F) in either time order, starting
// from h0 (nil = zeros). Returns the full sequence (B, T, H) and the final
// hidden state (B, H).
func runRNNDir(cell *RNNCell, x *tensor.Tensor, reverse bool, h0 *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	h := h0
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		h = cell.Forward(sliceTime(x, t), h)
		outs[t] = h
	}
	return stackTime(outs, B, T, cell.HiddenSize), h
}

// runLSTMDir runs one LSTMCell over x in either direction, starting from
// state0 (nil = zeros). Returns the full sequence (B, T, H) and the final
// state.
func runLSTMDir(cell *LSTMCell, x *tensor.Tensor, reverse bool, state0 *LSTMState) (*tensor.Tensor, *LSTMState) {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	state := state0
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		state = cell.Forward(sliceTime(x, t), state)
		outs[t] = state.H
	}
	return stackTime(outs, B, T, cell.HiddenSize), state
}

// runGRUDir runs one GRUCell over x in either direction, starting from h0
// (nil = zeros). Returns the full sequence (B, T, H) and the final hidden
// state (B, H).
func runGRUDir(cell *GRUCell, x *tensor.Tensor, reverse bool, h0 *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	h := h0
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		h = cell.Forward(sliceTime(x, t), h)
		outs[t] = h
	}
	return stackTime(outs, B, T, cell.HiddenSize), h
}

// numDirections returns 2 for bidirectional stacks, else 1.
func numDirections(bidir bool) int {
	if bidir {
		return 2
	}
	return 1
}

// checkStateShape validates an explicit initial state of shape (LD, B, H).
func checkStateShape(kind string, s *tensor.Tensor, LD, B, H int) {
	if len(s.Shape) != 3 || s.Shape[0] != LD || s.Shape[1] != B || s.Shape[2] != H {
		panic(fmt.Sprintf("%s: state shape %v, want (numLayers*numDirections, B, H) = (%d, %d, %d)",
			kind, s.Shape, LD, B, H))
	}
}

// stateSlice returns state[idx] as (B, H); nil state yields nil (= zeros at
// the first cell step). Differentiable via IndexSelect.
func stateSlice(state *tensor.Tensor, idx, B, H int) *tensor.Tensor {
	if state == nil {
		return nil
	}
	return state.IndexSelect(0, tensor.New([]float64{float64(idx)}, 1)).Reshape(B, H)
}

// stackStates stacks LD final states of shape (B, H) into (LD, B, H).
func stackStates(states []*tensor.Tensor) *tensor.Tensor {
	return tensor.Stack(0, states...)
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
	var cellOpts []RNNOpt
	if o.relu {
		cellOpts = append(cellOpts, WithReLU())
	}
	for i := 0; i < o.layers; i++ {
		layerIn := layerInputSize(i, in, hidden, o.bidir)
		r.Cells[i] = NewRNNCell(layerIn, hidden, cellOpts...)
		if o.bidir {
			r.BackCells[i] = NewRNNCell(layerIn, hidden, cellOpts...)
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
		fwd, _ := runRNNDir(r.Cells[li], cur, false, nil)
		if !r.Bidirectional {
			cur = fwd
			continue
		}
		bwd, _ := runRNNDir(r.BackCells[li], cur, true, nil)
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur
}

// ForwardWithState runs the stack from an explicit initial hidden state and
// also returns the final hidden state, PyTorch style. x: (B, T, F); h0:
// (numLayers*numDirections, B, H) or nil for zeros. Layout matches PyTorch:
// index layer*numDirections + direction (0 = forward, 1 = backward). Returns
// the output sequence (as Forward) and hN of the same shape as h0. With nil
// h0 the sequence output equals Forward(x) exactly.
func (r *RNN) ForwardWithState(x, h0 *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	if len(x.Shape) != 3 {
		panic("RNN.ForwardWithState: expected (B, T, F)")
	}
	B := x.Shape[0]
	D := numDirections(r.Bidirectional)
	LD := r.NumLayers * D
	if h0 != nil {
		checkStateShape("RNN.ForwardWithState", h0, LD, B, r.HiddenSize)
	}
	finals := make([]*tensor.Tensor, LD)
	cur := x
	for li := 0; li < r.NumLayers; li++ {
		fwd, hF := runRNNDir(r.Cells[li], cur, false, stateSlice(h0, li*D, B, r.HiddenSize))
		finals[li*D] = hF
		if !r.Bidirectional {
			cur = fwd
			continue
		}
		bwd, hB := runRNNDir(r.BackCells[li], cur, true, stateSlice(h0, li*D+1, B, r.HiddenSize))
		finals[li*D+1] = hB
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur, stackStates(finals)
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
		fwd, _ := runLSTMDir(l.Cells[li], cur, false, nil)
		if !l.Bidirectional {
			cur = fwd
			continue
		}
		bwd, _ := runLSTMDir(l.BackCells[li], cur, true, nil)
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur
}

// ForwardWithState runs the stack from explicit initial (h0, c0) and also
// returns the final (hN, cN), PyTorch style. x: (B, T, F); h0 and c0:
// (numLayers*numDirections, B, H) or nil for zeros (they must be both nil or
// both set). Layout matches PyTorch: index layer*numDirections + direction.
// With nil state the sequence output equals Forward(x) exactly.
func (l *LSTM) ForwardWithState(x, h0, c0 *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	if len(x.Shape) != 3 {
		panic("LSTM.ForwardWithState: expected (B, T, F)")
	}
	if (h0 == nil) != (c0 == nil) {
		panic("LSTM.ForwardWithState: h0 and c0 must be both nil or both set")
	}
	B := x.Shape[0]
	D := numDirections(l.Bidirectional)
	LD := l.NumLayers * D
	if h0 != nil {
		checkStateShape("LSTM.ForwardWithState", h0, LD, B, l.HiddenSize)
		checkStateShape("LSTM.ForwardWithState", c0, LD, B, l.HiddenSize)
	}
	initState := func(idx int) *LSTMState {
		if h0 == nil {
			return nil
		}
		return &LSTMState{
			H: stateSlice(h0, idx, B, l.HiddenSize),
			C: stateSlice(c0, idx, B, l.HiddenSize),
		}
	}
	finalH := make([]*tensor.Tensor, LD)
	finalC := make([]*tensor.Tensor, LD)
	cur := x
	for li := 0; li < l.NumLayers; li++ {
		fwd, stF := runLSTMDir(l.Cells[li], cur, false, initState(li*D))
		finalH[li*D], finalC[li*D] = stF.H, stF.C
		if !l.Bidirectional {
			cur = fwd
			continue
		}
		bwd, stB := runLSTMDir(l.BackCells[li], cur, true, initState(li*D+1))
		finalH[li*D+1], finalC[li*D+1] = stB.H, stB.C
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur, stackStates(finalH), stackStates(finalC)
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
		fwd, _ := runGRUDir(g.Cells[li], cur, false, nil)
		if !g.Bidirectional {
			cur = fwd
			continue
		}
		bwd, _ := runGRUDir(g.BackCells[li], cur, true, nil)
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur
}

// ForwardWithState runs the stack from an explicit initial hidden state and
// also returns the final hidden state, PyTorch style. x: (B, T, F); h0:
// (numLayers*numDirections, B, H) or nil for zeros. Layout matches PyTorch:
// index layer*numDirections + direction (0 = forward, 1 = backward). Returns
// the output sequence (as Forward) and hN of the same shape as h0. With nil
// h0 the sequence output equals Forward(x) exactly.
func (g *GRU) ForwardWithState(x, h0 *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	if len(x.Shape) != 3 {
		panic("GRU.ForwardWithState: expected (B, T, F)")
	}
	B := x.Shape[0]
	D := numDirections(g.Bidirectional)
	LD := g.NumLayers * D
	if h0 != nil {
		checkStateShape("GRU.ForwardWithState", h0, LD, B, g.HiddenSize)
	}
	finals := make([]*tensor.Tensor, LD)
	cur := x
	for li := 0; li < g.NumLayers; li++ {
		fwd, hF := runGRUDir(g.Cells[li], cur, false, stateSlice(h0, li*D, B, g.HiddenSize))
		finals[li*D] = hF
		if !g.Bidirectional {
			cur = fwd
			continue
		}
		bwd, hB := runGRUDir(g.BackCells[li], cur, true, stateSlice(h0, li*D+1, B, g.HiddenSize))
		finals[li*D+1] = hB
		cur = tensor.Concat(2, fwd, bwd)
	}
	return cur, stackStates(finals)
}
