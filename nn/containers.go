package nn

// Containers — Go adaptations of torch.nn.ModuleList / ModuleDict /
// ParameterList / ParameterDict. All four satisfy Child (by embedding Base),
// so registering one on a parent wires its contents into the parent's
// parameter tree and train/eval propagation. Like their PyTorch namesakes,
// none of them is a Module: they have no Forward and exist purely to hold
// state. Deviations from PyTorch are documented per type.

import (
	"sort"
	"strconv"

	"gonn/tensor"
)

// ModuleList holds submodules in a list, registering each as a child under
// its index ("0", "1", ...) so parameters, buffers, and train/eval mode
// propagate. It is NOT a Module — it has no Forward (same as PyTorch's
// nn.ModuleList); iterate it yourself:
//
//	for i := 0; i < ml.Len(); i++ { x = nn.Call(ml.Get(i), x) }
//
// Deviation from PyTorch: entries are single-input Modules (GoNN's Module
// interface), and there is no insert/delete — the child registry is
// append-only.
type ModuleList struct {
	Base
	mods []Module
}

// NewModuleList builds a ModuleList from the given modules.
func NewModuleList(mods ...Module) *ModuleList {
	l := &ModuleList{}
	for _, m := range mods {
		l.Append(m)
	}
	return l
}

// Append adds m to the list, registering it as child strconv(len). Returns
// the list for chaining.
func (l *ModuleList) Append(m Module) *ModuleList {
	l.regChild(strconv.Itoa(len(l.mods)), m)
	l.mods = append(l.mods, m)
	return l
}

// Get returns the i-th module.
func (l *ModuleList) Get(i int) Module { return l.mods[i] }

// Len returns the number of modules.
func (l *ModuleList) Len() int { return len(l.mods) }

// ModuleDict holds named submodules. Keys are iterated in SORTED order so
// Parameters()/NamedParameters() are deterministic regardless of insertion
// order (deviation from PyTorch, which preserves insertion order — Go maps
// have no stable iteration order to preserve). Set replaces an existing key.
type ModuleDict struct {
	Base
	mods map[string]Module
}

// NewModuleDict builds an empty ModuleDict.
func NewModuleDict() *ModuleDict { return &ModuleDict{mods: map[string]Module{}} }

// Set stores m under name, replacing any previous entry. Returns the dict
// for chaining.
func (d *ModuleDict) Set(name string, m Module) *ModuleDict {
	if d.mods == nil {
		d.mods = map[string]Module{}
	}
	d.mods[name] = m
	m.SetTraining(d.Training())
	return d
}

// Get returns the module stored under name, or nil if absent.
func (d *ModuleDict) Get(name string) Module { return d.mods[name] }

// Len returns the number of entries.
func (d *ModuleDict) Len() int { return len(d.mods) }

// Keys returns the keys in sorted order — the order used by Parameters().
func (d *ModuleDict) Keys() []string {
	keys := make([]string, 0, len(d.mods))
	for k := range d.mods {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Parameters returns the entries' parameters in sorted-key order.
func (d *ModuleDict) Parameters() []*tensor.Tensor {
	out := d.Base.Parameters()
	for _, k := range d.Keys() {
		out = append(out, d.mods[k].Parameters()...)
	}
	return out
}

// NamedParameters returns the entries' parameters with "key."-prefixed names,
// in sorted-key order.
func (d *ModuleDict) NamedParameters() []Param {
	out := d.Base.NamedParameters()
	for _, k := range d.Keys() {
		for _, p := range d.mods[k].NamedParameters() {
			out = append(out, Param{Name: k + "." + p.Name, T: p.T})
		}
	}
	return out
}

// Buffers returns the entries' buffers with "key."-prefixed names, in
// sorted-key order.
func (d *ModuleDict) Buffers() []Param {
	out := d.Base.Buffers()
	for _, k := range d.Keys() {
		for _, p := range d.mods[k].Buffers() {
			out = append(out, Param{Name: k + "." + p.Name, T: p.T})
		}
	}
	return out
}

// SetTraining sets the mode on the dict and every entry.
func (d *ModuleDict) SetTraining(m bool) {
	d.Base.SetTraining(m)
	for _, mod := range d.mods {
		mod.SetTraining(m)
	}
}

// Train puts the dict and every entry in training mode. (Overridden because
// the promoted Base.Train would call Base.SetTraining, not the dict's
// override — method promotion is not virtual dispatch.)
func (d *ModuleDict) Train() { d.SetTraining(true) }

// Eval puts the dict and every entry in eval mode.
func (d *ModuleDict) Eval() { d.SetTraining(false) }

// ParameterList holds bare parameter tensors in a list, registering each
// under its index ("0", "1", ...). Like PyTorch's nn.ParameterList, appended
// tensors become trainable: Append sets RequiresGrad (GoNN has no separate
// nn.Parameter wrapper type — that is the deviation).
type ParameterList struct {
	Base
	ps []*tensor.Tensor
}

// NewParameterList builds a ParameterList from the given tensors.
func NewParameterList(ps ...*tensor.Tensor) *ParameterList {
	l := &ParameterList{}
	for _, p := range ps {
		l.Append(p)
	}
	return l
}

// Append adds t as a trainable parameter (setting t.RequiresGrad, as
// wrapping in nn.Parameter would in PyTorch). Returns the list for chaining.
func (l *ParameterList) Append(t *tensor.Tensor) *ParameterList {
	l.reg(strconv.Itoa(len(l.ps)), t.SetRequiresGrad(true))
	l.ps = append(l.ps, t)
	return l
}

// Get returns the i-th parameter.
func (l *ParameterList) Get(i int) *tensor.Tensor { return l.ps[i] }

// Len returns the number of parameters.
func (l *ParameterList) Len() int { return len(l.ps) }

// ParameterDict holds named bare parameter tensors. Keys are iterated in
// SORTED order for deterministic Parameters()/NamedParameters() (deviation
// from PyTorch's insertion order, as with ModuleDict). Set marks the tensor
// trainable and replaces any previous entry.
type ParameterDict struct {
	Base
	ps map[string]*tensor.Tensor
}

// NewParameterDict builds an empty ParameterDict.
func NewParameterDict() *ParameterDict { return &ParameterDict{ps: map[string]*tensor.Tensor{}} }

// Set stores t under name as a trainable parameter (setting t.RequiresGrad),
// replacing any previous entry. Returns the dict for chaining.
func (d *ParameterDict) Set(name string, t *tensor.Tensor) *ParameterDict {
	if d.ps == nil {
		d.ps = map[string]*tensor.Tensor{}
	}
	d.ps[name] = t.SetRequiresGrad(true)
	return d
}

// Get returns the parameter stored under name, or nil if absent.
func (d *ParameterDict) Get(name string) *tensor.Tensor { return d.ps[name] }

// Len returns the number of entries.
func (d *ParameterDict) Len() int { return len(d.ps) }

// Keys returns the keys in sorted order — the order used by Parameters().
func (d *ParameterDict) Keys() []string {
	keys := make([]string, 0, len(d.ps))
	for k := range d.ps {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Parameters returns the stored tensors in sorted-key order.
func (d *ParameterDict) Parameters() []*tensor.Tensor {
	out := d.Base.Parameters()
	for _, k := range d.Keys() {
		out = append(out, d.ps[k])
	}
	return out
}

// NamedParameters returns the stored tensors named by key, in sorted order.
func (d *ParameterDict) NamedParameters() []Param {
	out := d.Base.NamedParameters()
	for _, k := range d.Keys() {
		out = append(out, Param{Name: k, T: d.ps[k]})
	}
	return out
}
