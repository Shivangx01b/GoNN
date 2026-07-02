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
	layers  int
	bidir   bool
	relu    bool
	dropout float64
	proj    int
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

// WithRNNDropout applies dropout with probability p to the output of every
// layer except the last (PyTorch RNN/LSTM/GRU dropout semantics). Active only
// in Training(); identity in Eval(). With p == 0 (the default) or a single
// layer no Dropout modules are constructed at all, so the module tree is
// identical to the historical one.
func WithRNNDropout(p float64) RNNOpt { return func(o *rnnOpts) { o.dropout = p } }

// WithProjSize enables LSTM output projections (PyTorch proj_size): each cell
// gains W_hr = Linear(hidden, proj, no bias) and emits h_t = W_hr(o * tanh(c)),
// so the hidden output (and the recurrent Whh input) has size proj while the
// cell state keeps size hidden. LSTM/LSTMCell only; requires 0 < proj < hidden.
func WithProjSize(p int) RNNOpt { return func(o *rnnOpts) { o.proj = p } }

func resolveRNNOpts(opts []RNNOpt) rnnOpts {
	o := rnnOpts{layers: 1}
	for _, fn := range opts {
		fn(&o)
	}
	if o.layers < 1 {
		panic(fmt.Sprintf("nn: rnn layers must be >= 1, got %d", o.layers))
	}
	if o.dropout < 0 || o.dropout > 1 {
		panic(fmt.Sprintf("nn: rnn dropout must be in [0, 1], got %g", o.dropout))
	}
	if o.proj < 0 {
		panic(fmt.Sprintf("nn: rnn proj size must be >= 0, got %d", o.proj))
	}
	return o
}

// checkProjSize validates a WithProjSize value against the hidden size (and
// rejects it entirely for non-LSTM kinds).
func checkProjSize(kind string, proj, hidden int) {
	if proj == 0 {
		return
	}
	if kind != "LSTM" && kind != "LSTMCell" {
		panic(fmt.Sprintf("nn: WithProjSize applies only to LSTM/LSTMCell, not %s", kind))
	}
	if proj >= hidden {
		panic(fmt.Sprintf("nn: %s proj size must satisfy 0 < proj < hidden, got proj=%d hidden=%d", kind, proj, hidden))
	}
}

// makeInterLayerDropout builds the per-layer Dropout children used by
// WithRNNDropout: one per layer boundary (layers-1 total), or nil when p == 0
// or there is a single layer — in that case the module tree is untouched.
func makeInterLayerDropout(b *Base, p float64, layers int) []*Dropout {
	if p <= 0 || layers < 2 {
		return nil
	}
	ds := make([]*Dropout, layers-1)
	for i := range ds {
		ds[i] = NewDropout(p)
		b.regChild("dropout."+strconv.Itoa(i), ds[i])
	}
	return ds
}

// applyInterLayerDropout applies ds[li] to a layer's output when li is not the
// last layer (PyTorch semantics: no dropout after the final layer).
func applyInterLayerDropout(ds []*Dropout, li, layers int, x *tensor.Tensor) *tensor.Tensor {
	if ds == nil || li >= layers-1 {
		return x
	}
	return ds[li].Forward(x)
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
// pass WithReLU() for the ReLU nonlinearity. WithLayers/WithBidirectional/
// WithRNNDropout do not apply to a single cell and are ignored; WithProjSize
// is LSTM-only and panics.
func NewRNNCell(in, hidden int, opts ...RNNOpt) *RNNCell {
	o := resolveRNNOpts(opts)
	checkProjSize("RNNCell", o.proj, hidden)
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
	// ProjSize > 0 enables the PyTorch proj_size projection: the emitted
	// hidden state (and Whh's input) has size ProjSize while the cell state
	// stays HiddenSize. Zero (the default) means no projection.
	ProjSize int
	Wih      *Linear // x -> 4H (i, f, g, o)
	Whh      *Linear // h -> 4H
	Whr      *Linear // (ProjSize > 0 only) hidden -> proj, no bias
}

// NewLSTMCell creates an LSTMCell. Pass WithProjSize(p) (0 < p < hidden) for
// the PyTorch proj_size variant: h_t = W_hr(o * tanh(c)) with W_hr =
// Linear(hidden, proj, no bias). Other stack options are ignored. The default
// (no projection) draws the identical RNG sequence as the historical
// constructor.
func NewLSTMCell(in, hidden int, opts ...RNNOpt) *LSTMCell {
	o := resolveRNNOpts(opts)
	checkProjSize("LSTMCell", o.proj, hidden)
	hOut := hidden
	if o.proj > 0 {
		hOut = o.proj
	}
	c := &LSTMCell{
		InputSize:  in,
		HiddenSize: hidden,
		ProjSize:   o.proj,
		Wih:        NewLinear(in, 4*hidden, true),
		Whh:        NewLinear(hOut, 4*hidden, false),
	}
	if o.proj > 0 {
		c.Whr = NewLinear(hidden, o.proj, false)
	}
	c.regChild("wih", c.Wih)
	c.regChild("whh", c.Whh)
	if c.Whr != nil {
		c.regChild("whr", c.Whr)
	}
	return c
}

// hiddenOut returns the emitted hidden-state size: ProjSize when projecting,
// else HiddenSize.
func (c *LSTMCell) hiddenOut() int {
	if c.ProjSize > 0 {
		return c.ProjSize
	}
	return c.HiddenSize
}

// Forward consumes one timestep. x: (B, in); state may be nil for t=0.
// Returns new state (h_t, c_t): h_t is (B, hiddenOut()), c_t is (B, hidden).
func (c *LSTMCell) Forward(x *tensor.Tensor, state *LSTMState) *LSTMState {
	B := x.Shape[0]
	H := c.HiddenSize
	if state == nil {
		state = &LSTMState{H: tensor.Zeros(B, c.hiddenOut()), C: tensor.Zeros(B, H)}
	}
	gates := c.Wih.Forward(x).Add(c.Whh.Forward(state.H)) // (B, 4H)
	i := sliceCol(gates, 0, H).Sigmoid()
	f := sliceCol(gates, H, 2*H).Sigmoid()
	g := sliceCol(gates, 2*H, 3*H).Tanh()
	o := sliceCol(gates, 3*H, 4*H).Sigmoid()
	cNew := f.Mul(state.C).Add(i.Mul(g))
	hNew := o.Mul(cNew.Tanh())
	if c.Whr != nil {
		hNew = c.Whr.Forward(hNew) // (B, proj)
	}
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
// state0 (nil = zeros). Returns the full sequence (B, T, hiddenOut) and the
// final state.
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
	return stackTime(outs, B, T, cell.hiddenOut()), state
}

// runLSTMDirC is runLSTMDir plus a stacked (B, T, hidden) sequence of the
// per-step cell states, needed by ForwardPacked to gather each sequence's
// c at its true last step. Used only on the packed path so the default
// Forward/ForwardWithState graphs stay untouched.
func runLSTMDirC(cell *LSTMCell, x *tensor.Tensor, reverse bool, state0 *LSTMState) (*tensor.Tensor, *tensor.Tensor, *LSTMState) {
	B, T := x.Shape[0], x.Shape[1]
	outs := make([]*tensor.Tensor, T)
	cs := make([]*tensor.Tensor, T)
	state := state0
	for step := 0; step < T; step++ {
		t := step
		if reverse {
			t = T - 1 - step
		}
		state = cell.Forward(sliceTime(x, t), state)
		outs[t] = state.H
		cs[t] = state.C
	}
	return stackTime(outs, B, T, cell.hiddenOut()), stackTime(cs, B, T, cell.HiddenSize), state
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
	Dropouts      []*Dropout // inter-layer dropout (nil unless WithRNNDropout(p>0) and layers > 1)
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewRNN builds an RNN stack: NewRNN(in, hidden), or with options
// NewRNN(in, hidden, WithLayers(2), WithBidirectional()).
func NewRNN(in, hidden int, opts ...RNNOpt) *RNN {
	o := resolveRNNOpts(opts)
	checkProjSize("RNN", o.proj, hidden)
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
	r.Dropouts = makeInterLayerDropout(&r.Base, o.dropout, o.layers)
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
			cur = applyInterLayerDropout(r.Dropouts, li, r.NumLayers, fwd)
			continue
		}
		bwd, _ := runRNNDir(r.BackCells[li], cur, true, nil)
		cur = applyInterLayerDropout(r.Dropouts, li, r.NumLayers, tensor.Concat(2, fwd, bwd))
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
			cur = applyInterLayerDropout(r.Dropouts, li, r.NumLayers, fwd)
			continue
		}
		bwd, hB := runRNNDir(r.BackCells[li], cur, true, stateSlice(h0, li*D+1, B, r.HiddenSize))
		finals[li*D+1] = hB
		cur = applyInterLayerDropout(r.Dropouts, li, r.NumLayers, tensor.Concat(2, fwd, bwd))
	}
	return cur, stackStates(finals)
}

// LSTM is a stack of LSTM layers, optionally bidirectional.
// Default: one layer, unidirectional.
type LSTM struct {
	Base
	Cells      []*LSTMCell
	BackCells  []*LSTMCell
	Dropouts   []*Dropout // inter-layer dropout (nil unless WithRNNDropout(p>0) and layers > 1)
	NumLayers  int
	HiddenSize int
	// ProjSize > 0 means every cell projects its hidden output to ProjSize
	// (PyTorch proj_size); the cell states keep HiddenSize.
	ProjSize      int
	Bidirectional bool
}

// NewLSTM builds an LSTM stack. WithProjSize(p) (0 < p < hidden) enables
// PyTorch proj_size: outputs and hN have feature size p (2p bidirectional)
// while cN keeps the hidden size.
func NewLSTM(in, hidden int, opts ...RNNOpt) *LSTM {
	o := resolveRNNOpts(opts)
	checkProjSize("LSTM", o.proj, hidden)
	l := &LSTM{NumLayers: o.layers, HiddenSize: hidden, ProjSize: o.proj, Bidirectional: o.bidir}
	l.Cells = make([]*LSTMCell, o.layers)
	if o.bidir {
		l.BackCells = make([]*LSTMCell, o.layers)
	}
	var cellOpts []RNNOpt
	if o.proj > 0 {
		cellOpts = append(cellOpts, WithProjSize(o.proj))
	}
	hOut := l.hiddenOut()
	for i := 0; i < o.layers; i++ {
		layerIn := layerInputSize(i, in, hOut, o.bidir)
		l.Cells[i] = NewLSTMCell(layerIn, hidden, cellOpts...)
		if o.bidir {
			l.BackCells[i] = NewLSTMCell(layerIn, hidden, cellOpts...)
		}
	}
	for i, c := range l.Cells {
		l.regChild("cells."+strconv.Itoa(i), c)
	}
	for i, c := range l.BackCells {
		l.regChild("backcells."+strconv.Itoa(i), c)
	}
	l.Dropouts = makeInterLayerDropout(&l.Base, o.dropout, o.layers)
	return l
}

// hiddenOut returns the per-direction output feature size: ProjSize when
// projecting, else HiddenSize.
func (l *LSTM) hiddenOut() int {
	if l.ProjSize > 0 {
		return l.ProjSize
	}
	return l.HiddenSize
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
			cur = applyInterLayerDropout(l.Dropouts, li, l.NumLayers, fwd)
			continue
		}
		bwd, _ := runLSTMDir(l.BackCells[li], cur, true, nil)
		cur = applyInterLayerDropout(l.Dropouts, li, l.NumLayers, tensor.Concat(2, fwd, bwd))
	}
	return cur
}

// ForwardWithState runs the stack from explicit initial (h0, c0) and also
// returns the final (hN, cN), PyTorch style. x: (B, T, F); h0:
// (numLayers*numDirections, B, hiddenOut) and c0: (numLayers*numDirections,
// B, H), or nil for zeros (they must be both nil or both set) — with
// WithProjSize the h states have feature size proj while the c states keep
// the hidden size. Layout matches PyTorch: index layer*numDirections +
// direction. With nil state the sequence output equals Forward(x) exactly.
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
	hOut := l.hiddenOut()
	if h0 != nil {
		checkStateShape("LSTM.ForwardWithState", h0, LD, B, hOut)
		checkStateShape("LSTM.ForwardWithState", c0, LD, B, l.HiddenSize)
	}
	initState := func(idx int) *LSTMState {
		if h0 == nil {
			return nil
		}
		return &LSTMState{
			H: stateSlice(h0, idx, B, hOut),
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
			cur = applyInterLayerDropout(l.Dropouts, li, l.NumLayers, fwd)
			continue
		}
		bwd, stB := runLSTMDir(l.BackCells[li], cur, true, initState(li*D+1))
		finalH[li*D+1], finalC[li*D+1] = stB.H, stB.C
		cur = applyInterLayerDropout(l.Dropouts, li, l.NumLayers, tensor.Concat(2, fwd, bwd))
	}
	return cur, stackStates(finalH), stackStates(finalC)
}

// GRU is a stack of GRU layers, optionally bidirectional.
// Default: one layer, unidirectional.
type GRU struct {
	Base
	Cells         []*GRUCell
	BackCells     []*GRUCell
	Dropouts      []*Dropout // inter-layer dropout (nil unless WithRNNDropout(p>0) and layers > 1)
	NumLayers     int
	HiddenSize    int
	Bidirectional bool
}

// NewGRU builds a GRU stack.
func NewGRU(in, hidden int, opts ...RNNOpt) *GRU {
	o := resolveRNNOpts(opts)
	checkProjSize("GRU", o.proj, hidden)
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
	g.Dropouts = makeInterLayerDropout(&g.Base, o.dropout, o.layers)
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
			cur = applyInterLayerDropout(g.Dropouts, li, g.NumLayers, fwd)
			continue
		}
		bwd, _ := runGRUDir(g.BackCells[li], cur, true, nil)
		cur = applyInterLayerDropout(g.Dropouts, li, g.NumLayers, tensor.Concat(2, fwd, bwd))
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
			cur = applyInterLayerDropout(g.Dropouts, li, g.NumLayers, fwd)
			continue
		}
		bwd, hB := runGRUDir(g.BackCells[li], cur, true, stateSlice(h0, li*D+1, B, g.HiddenSize))
		finals[li*D+1] = hB
		cur = applyInterLayerDropout(g.Dropouts, li, g.NumLayers, tensor.Concat(2, fwd, bwd))
	}
	return cur, stackStates(finals)
}

// ============================================================================
// Packed-sequence forwards
// ============================================================================
//
// DEVIATION (padded-storage adaptation): PackedSequence stores padded data +
// lengths, so ForwardPacked runs the ordinary padded forward and then
// (a) zeroes the outputs at t >= length_i (differentiable 0/1 mask) and
// (b) gathers each sequence's final state at its true last step
// t = length_i - 1 per layer and direction. For the FORWARD direction this
// is exactly equivalent to PyTorch's packed evaluation: causality guarantees
// every valid timestep never sees a padded step.
//
// The REVERSE direction is made exact via per-sequence time reversal
// (reverseValidPrefix). Let R be the involution R(x)[b, t] = x[b, len_b-1-t]
// for t < len_b and R(x)[b, t] = x[b, t] for t >= len_b. PyTorch's packed
// reverse pass runs the reverse cell over each sequence's own last-to-first
// valid elements; that is identical to running the same cell FORWARD over
// R(x): at forward step t < len_b the cell has consumed R(x)[b, 0..t] =
// x[b, len_b-1], ..., x[b, len_b-1-t] — exactly the suffix the packed
// reverse pass consumes when emitting the output for original time
// len_b-1-t. Re-applying R to the forward-run outputs therefore places every
// valid reverse output back at its original timestep (revSeq[b, t] =
// tmp[b, len_b-1-t] for t < len_b), and the reverse-direction final state is
// the forward run's state at step len_b-1 — the same per-sequence gather the
// forward direction uses.
//
// Why valid outputs are unaffected by padding: for t < len_b the forward
// cell reads cur[b, 0..t] (all valid) and the reversed run reads
// R(cur)[b, 0..t] (all images of valid entries), so every valid step of a
// layer depends only on valid steps of the previous layer. Positions
// t >= len_b of the reversed run ARE garbage (they mix in padding), but they
// land back on t >= len_b after re-reversal and the concatenated layer
// output is masked to zero there before the next layer, so the garbage never
// propagates. Each sequence's trimmed output and final states are therefore
// bit-identical to running the stack on that sequence alone.

// packedResult re-wraps a masked batch-first output (B, T, H) in the caller's
// layout with a fresh copy of lengths.
func packedResult(masked *tensor.Tensor, ps PackedSequence) PackedSequence {
	out := masked
	if !ps.BatchFirst {
		out = masked.Permute(1, 0, 2)
	}
	lengths := make([]int, len(ps.Lengths))
	copy(lengths, ps.Lengths)
	return PackedSequence{Padded: out, Lengths: lengths, BatchFirst: ps.BatchFirst}
}

// ForwardPacked runs the stack over a PackedSequence. It returns the packed
// output — padded positions t >= length_i zeroed, layout matching
// ps.BatchFirst — and hN of shape (numLayers*numDirections, B, H) taken at
// each sequence's true last step, indexed layer*numDirections + direction
// (0 = forward, 1 = reverse), PyTorch style. Bidirectional stacks are exact:
// the reverse direction runs the back cell forward over the per-sequence
// time reversal of the layer input (see the deviation note above).
// Inter-layer dropout (WithRNNDropout) applies as in Forward; hN is gathered
// from the pre-dropout layer outputs, PyTorch style.
func (r *RNN) ForwardPacked(ps PackedSequence) (PackedSequence, *tensor.Tensor) {
	ps.checkAgainst("RNN.ForwardPacked")
	x := ps.batchFirstPadded() // (B, T, F)
	B, T := x.Shape[0], x.Shape[1]
	D := numDirections(r.Bidirectional)
	mask := packedOutputMask(B, T, ps.Lengths)
	finals := make([]*tensor.Tensor, r.NumLayers*D)
	cur := x
	last := x
	for li := 0; li < r.NumLayers; li++ {
		out, _ := runRNNDir(r.Cells[li], cur, false, nil)
		finals[li*D] = gatherLastSteps(out, ps.Lengths)
		if r.Bidirectional {
			revIn := reverseValidPrefix(cur, ps.Lengths)
			tmp, _ := runRNNDir(r.BackCells[li], revIn, false, nil)
			finals[li*D+1] = gatherLastSteps(tmp, ps.Lengths)
			out = tensor.Concat(2, out, reverseValidPrefix(tmp, ps.Lengths)).Mul(mask)
		}
		last = out
		cur = applyInterLayerDropout(r.Dropouts, li, r.NumLayers, out)
	}
	masked := last
	if !r.Bidirectional {
		masked = last.Mul(mask)
	}
	return packedResult(masked, ps), stackStates(finals)
}

// ForwardPacked runs the LSTM stack over a PackedSequence. It returns the
// packed output (padded positions zeroed), hN of shape (numLayers*
// numDirections, B, hiddenOut) and cN of shape (numLayers*numDirections, B,
// H), both taken at each sequence's true last step and indexed
// layer*numDirections + direction (0 = forward, 1 = reverse), PyTorch style.
// Bidirectional stacks are exact: the reverse direction runs the back cell
// forward over the per-sequence time reversal of the layer input (see the
// deviation note above). With WithProjSize the h states have feature size
// proj while the c states keep the hidden size.
func (l *LSTM) ForwardPacked(ps PackedSequence) (PackedSequence, *tensor.Tensor, *tensor.Tensor) {
	ps.checkAgainst("LSTM.ForwardPacked")
	x := ps.batchFirstPadded() // (B, T, F)
	B, T := x.Shape[0], x.Shape[1]
	D := numDirections(l.Bidirectional)
	mask := packedOutputMask(B, T, ps.Lengths)
	finalH := make([]*tensor.Tensor, l.NumLayers*D)
	finalC := make([]*tensor.Tensor, l.NumLayers*D)
	cur := x
	last := x
	for li := 0; li < l.NumLayers; li++ {
		out, cs, _ := runLSTMDirC(l.Cells[li], cur, false, nil)
		finalH[li*D] = gatherLastSteps(out, ps.Lengths)
		finalC[li*D] = gatherLastSteps(cs, ps.Lengths)
		if l.Bidirectional {
			revIn := reverseValidPrefix(cur, ps.Lengths)
			tmp, tmpC, _ := runLSTMDirC(l.BackCells[li], revIn, false, nil)
			finalH[li*D+1] = gatherLastSteps(tmp, ps.Lengths)
			finalC[li*D+1] = gatherLastSteps(tmpC, ps.Lengths)
			out = tensor.Concat(2, out, reverseValidPrefix(tmp, ps.Lengths)).Mul(mask)
		}
		last = out
		cur = applyInterLayerDropout(l.Dropouts, li, l.NumLayers, out)
	}
	masked := last
	if !l.Bidirectional {
		masked = last.Mul(mask)
	}
	return packedResult(masked, ps), stackStates(finalH), stackStates(finalC)
}

// ForwardPacked runs the GRU stack over a PackedSequence. It returns the
// packed output (padded positions zeroed) and hN of shape
// (numLayers*numDirections, B, H) taken at each sequence's true last step,
// indexed layer*numDirections + direction (0 = forward, 1 = reverse),
// PyTorch style. Bidirectional stacks are exact: the reverse direction runs
// the back cell forward over the per-sequence time reversal of the layer
// input (see the deviation note above).
func (g *GRU) ForwardPacked(ps PackedSequence) (PackedSequence, *tensor.Tensor) {
	ps.checkAgainst("GRU.ForwardPacked")
	x := ps.batchFirstPadded() // (B, T, F)
	B, T := x.Shape[0], x.Shape[1]
	D := numDirections(g.Bidirectional)
	mask := packedOutputMask(B, T, ps.Lengths)
	finals := make([]*tensor.Tensor, g.NumLayers*D)
	cur := x
	last := x
	for li := 0; li < g.NumLayers; li++ {
		out, _ := runGRUDir(g.Cells[li], cur, false, nil)
		finals[li*D] = gatherLastSteps(out, ps.Lengths)
		if g.Bidirectional {
			revIn := reverseValidPrefix(cur, ps.Lengths)
			tmp, _ := runGRUDir(g.BackCells[li], revIn, false, nil)
			finals[li*D+1] = gatherLastSteps(tmp, ps.Lengths)
			out = tensor.Concat(2, out, reverseValidPrefix(tmp, ps.Lengths)).Mul(mask)
		}
		last = out
		cur = applyInterLayerDropout(g.Dropouts, li, g.NumLayers, out)
	}
	masked := last
	if !g.Bidirectional {
		masked = last.Mul(mask)
	}
	return packedResult(masked, ps), stackStates(finals)
}
