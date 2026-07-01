package nn

import (
	"strconv"

	"gonn/tensor"
)

// Param is a named tensor: a trainable parameter or a persistent buffer
// (e.g. BatchNorm running statistics). Names are hierarchical dotted paths
// ("layers.0.weight", "attn.qproj.bias") assembled from registration names.
type Param struct {
	Name string
	T    *tensor.Tensor
}

// Child is the state-carrying surface every module gets by embedding Base.
// Multi-input modules (MultiHeadAttention, Bilinear, Seq2Seq, ...) satisfy
// Child but not Module — they participate in the parameter/mode tree without
// faking a single-input Forward.
type Child interface {
	Parameters() []*tensor.Tensor
	NamedParameters() []Param
	Buffers() []Param
	SetTraining(bool)
}

// Module is a single-input layer usable inside Sequential.
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Child
}

type childEntry struct {
	name string
	c    Child
}

// Base provides parameter/buffer/child registration and train/eval state.
// Embed it in every module (with pointer receivers) and register state in
// the constructor:
//
//	l.Weight = l.reg("weight", w)
//	l.Bias   = l.reg("bias", b)
//	m.regChild("qproj", m.QProj)
//
// Registration order defines Parameters() order (direct params first, then
// children depth-first), so constructors register in the same order the
// historical hand-written Parameters() methods appended.
//
// The zero value is usable and starts in training mode (the eval flag is
// inverted for exactly that reason), so struct-literal modules behave like
// PyTorch modules do by default.
type Base struct {
	eval     bool
	params   []Param
	buffers  []Param
	children []childEntry
}

// reg registers a trainable parameter under name and returns it.
func (b *Base) reg(name string, t *tensor.Tensor) *tensor.Tensor {
	b.params = append(b.params, Param{Name: name, T: t})
	return t
}

// regBuf registers a non-trainable buffer (running stats, ...) and returns it.
func (b *Base) regBuf(name string, t *tensor.Tensor) *tensor.Tensor {
	b.buffers = append(b.buffers, Param{Name: name, T: t})
	return t
}

// regChild registers a sub-module under name.
func (b *Base) regChild(name string, c Child) {
	b.children = append(b.children, childEntry{name: name, c: c})
}

// RegisterParam registers a trainable parameter under name and returns it.
// Exported for custom modules defined outside this package:
//
//	type MyLayer struct{ nn.Base; W *tensor.Tensor }
//	l.W = l.RegisterParam("weight", tensor.Randn(4, 4).SetRequiresGrad(true))
func (b *Base) RegisterParam(name string, t *tensor.Tensor) *tensor.Tensor {
	return b.reg(name, t)
}

// RegisterBuffer registers a non-trainable buffer (running statistics, ...)
// under name and returns it. Buffers appear in Buffers(), not Parameters().
func (b *Base) RegisterBuffer(name string, t *tensor.Tensor) *tensor.Tensor {
	return b.regBuf(name, t)
}

// RegisterChild registers a sub-module under name, wiring its parameters,
// buffers, and train/eval mode into this module's tree.
func (b *Base) RegisterChild(name string, c Child) {
	b.regChild(name, c)
}

// Parameters returns this module's direct parameters followed by every
// child's, depth-first, in registration order.
func (b *Base) Parameters() []*tensor.Tensor {
	var out []*tensor.Tensor
	for _, p := range b.params {
		out = append(out, p.T)
	}
	for _, ce := range b.children {
		out = append(out, ce.c.Parameters()...)
	}
	return out
}

// NamedParameters returns all parameters with hierarchical dotted names,
// in the same order as Parameters(). Use with FilterParams to build
// optimizer parameter groups (e.g. exempting biases from weight decay).
func (b *Base) NamedParameters() []Param {
	var out []Param
	out = append(out, b.params...)
	for _, ce := range b.children {
		for _, p := range ce.c.NamedParameters() {
			out = append(out, Param{Name: ce.name + "." + p.Name, T: p.T})
		}
	}
	return out
}

// Buffers returns all persistent non-trainable buffers with dotted names.
func (b *Base) Buffers() []Param {
	var out []Param
	out = append(out, b.buffers...)
	for _, ce := range b.children {
		for _, p := range ce.c.Buffers() {
			out = append(out, Param{Name: ce.name + "." + p.Name, T: p.T})
		}
	}
	return out
}

// Training reports whether the module is in training mode.
func (b *Base) Training() bool { return !b.eval }

// SetTraining sets training (true) or eval (false) mode on this module and
// recursively on every registered child — Dropout and BatchNorm anywhere in
// the tree switch together.
func (b *Base) SetTraining(m bool) {
	b.eval = !m
	for _, ce := range b.children {
		ce.c.SetTraining(m)
	}
}

// Train puts the module (and all children) in training mode.
func (b *Base) Train() { b.SetTraining(true) }

// Eval puts the module (and all children) in eval mode.
func (b *Base) Eval() { b.SetTraining(false) }

// FilterParams returns the parameters of c whose dotted name satisfies pred.
// Typical use — exclude biases and norm gains from weight decay:
//
//	decay := nn.FilterParams(model, func(name string) bool {
//	    return strings.HasSuffix(name, ".weight")
//	})
func FilterParams(c Child, pred func(name string) bool) []*tensor.Tensor {
	var out []*tensor.Tensor
	for _, p := range c.NamedParameters() {
		if pred(p.Name) {
			out = append(out, p.T)
		}
	}
	return out
}

// Sequential chains a list of modules.
type Sequential struct {
	Base
	Layers []Module
}

// NewSequential builds a Sequential from the given modules.
func NewSequential(layers ...Module) *Sequential {
	s := &Sequential{}
	for _, m := range layers {
		s.Add(m)
	}
	return s
}

// Forward applies each layer in order.
func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

// Add appends a module (registered as a child, so parameters and train/eval
// mode propagate).
func (s *Sequential) Add(m Module) *Sequential {
	s.regChild(strconv.Itoa(len(s.Layers)), m)
	s.Layers = append(s.Layers, m)
	return s
}
