package nn

// Module hooks — an honest Go adaptation of torch.nn.Module's hook machinery.
//
// PyTorch fires hooks inside Module.__call__, which wraps every forward. Go
// has no __call__, so GoNN hooks fire in exactly two places:
//
//  1. inside container forwards — Sequential.Forward runs each child through
//     the hook pipeline, and
//  2. through the explicit Call(m, x) wrapper for direct invocations.
//
// A bare layer.Forward(x) BYPASSES all hooks; use nn.Call(layer, x) when you
// want hooks to fire on a directly-invoked module.
//
// Deviations from PyTorch, beyond the call-site difference above:
//
//   - Hooks are single-input/single-output: they see one *tensor.Tensor, not
//     an args tuple (GoNN's Module interface is single-input by design).
//   - The full backward hook receives only the module's OUTPUT gradient
//     (PyTorch's grad_output) and may replace it before it propagates into
//     the module's internal graph; PyTorch's hook receives both grad_input
//     and grad_output and replaces grad_input. The two coincide for modules
//     where the observed gradient is the one flowing through, and the GoNN
//     form is the one implementable as a graph node without __call__.
//   - Backward hooks do not fire when the module's output does not require
//     grad (same as PyTorch's rule that the hook is skipped when no gradient
//     flows).
//   - Removal uses integer ids (RemoveHook / RemoveModuleHook) instead of
//     RemovableHandle objects.
//   - Hook registries are not synchronized; register/remove hooks from the
//     same goroutine that runs forward/backward (the rest of the package has
//     the same discipline).
//
// Ordering matches PyTorch: global hooks (RegisterModuleForwardPreHook, ...)
// run before per-module hooks, and within each registry hooks run in
// registration order.

import "gonn/tensor"

// ForwardPreHook runs before Forward. It may replace the input by returning a
// non-nil tensor; returning nil leaves the input unchanged. m is the module
// being called.
type ForwardPreHook func(m Child, x *tensor.Tensor) *tensor.Tensor

// ForwardHook runs after Forward with the (possibly pre-hook-replaced) input
// x and the output y. It may replace the output by returning a non-nil
// tensor; returning nil leaves the output unchanged.
type ForwardHook func(m Child, x, y *tensor.Tensor) *tensor.Tensor

// BackwardHook runs during the backward pass when the gradient of the
// module's output is computed. It may replace that gradient by returning a
// non-nil tensor (which then propagates into the module's graph); returning
// nil passes gradOut through unchanged.
type BackwardHook func(m Child, gradOut *tensor.Tensor) *tensor.Tensor

type preHookEntry struct {
	id int
	fn ForwardPreHook
}

type fwdHookEntry struct {
	id int
	fn ForwardHook
}

type bwdHookEntry struct {
	id int
	fn BackwardHook
}

// hookSet holds the hooks of one registry (either one module's, or the
// package-global one).
type hookSet struct {
	pre []preHookEntry
	fwd []fwdHookEntry
	bwd []bwdHookEntry
}

// empty reports whether the set has no hooks (nil-safe).
func (h *hookSet) empty() bool {
	return h == nil || (len(h.pre) == 0 && len(h.fwd) == 0 && len(h.bwd) == 0)
}

// remove deletes the hook with the given id from whichever list holds it.
func (h *hookSet) remove(id int) bool {
	if h == nil {
		return false
	}
	for i, e := range h.pre {
		if e.id == id {
			h.pre = append(h.pre[:i], h.pre[i+1:]...)
			return true
		}
	}
	for i, e := range h.fwd {
		if e.id == id {
			h.fwd = append(h.fwd[:i], h.fwd[i+1:]...)
			return true
		}
	}
	for i, e := range h.bwd {
		if e.id == id {
			h.bwd = append(h.bwd[:i], h.bwd[i+1:]...)
			return true
		}
	}
	return false
}

// nextHookID is shared by per-module and global registries so every id is
// unique process-wide (ids play the role of PyTorch's RemovableHandle).
var nextHookID int

func newHookID() int {
	nextHookID++
	return nextHookID
}

func (b *Base) ensureHooks() *hookSet {
	if b.hooks == nil {
		b.hooks = &hookSet{}
	}
	return b.hooks
}

// RegisterForwardPreHook registers fn to run before this module's Forward
// whenever the module is invoked through the hook pipeline (nn.Call or a
// container such as Sequential). The returned id removes it via RemoveHook.
func (b *Base) RegisterForwardPreHook(fn ForwardPreHook) int {
	id := newHookID()
	h := b.ensureHooks()
	h.pre = append(h.pre, preHookEntry{id: id, fn: fn})
	return id
}

// RegisterForwardHook registers fn to run after this module's Forward
// whenever the module is invoked through the hook pipeline. The returned id
// removes it via RemoveHook.
func (b *Base) RegisterForwardHook(fn ForwardHook) int {
	id := newHookID()
	h := b.ensureHooks()
	h.fwd = append(h.fwd, fwdHookEntry{id: id, fn: fn})
	return id
}

// RegisterFullBackwardHook registers fn to run when the gradient of this
// module's output is computed during Backward (the module must have been
// invoked through the hook pipeline, and its output must require grad).
// The returned id removes it via RemoveHook.
func (b *Base) RegisterFullBackwardHook(fn BackwardHook) int {
	id := newHookID()
	h := b.ensureHooks()
	h.bwd = append(h.bwd, bwdHookEntry{id: id, fn: fn})
	return id
}

// RemoveHook removes the per-module hook with the given id, reporting whether
// it was found. (Global hooks are removed with RemoveModuleHook.)
func (b *Base) RemoveHook(id int) bool { return b.hooks.remove(id) }

// hookState exposes a module's hook set to the pipeline. Every module gets it
// by embedding Base.
func (b *Base) hookState() *hookSet { return b.hooks }

type hookHost interface {
	hookState() *hookSet
}

// globalHooks fire for EVERY module run through the hook pipeline, before the
// module's own hooks — mirroring torch.nn.modules.module.register_module_*.
var globalHooks hookSet

// RegisterModuleForwardPreHook registers a global forward pre-hook that runs
// (before per-module pre-hooks) for every module invoked through the hook
// pipeline. Remove with RemoveModuleHook(id).
func RegisterModuleForwardPreHook(fn ForwardPreHook) int {
	id := newHookID()
	globalHooks.pre = append(globalHooks.pre, preHookEntry{id: id, fn: fn})
	return id
}

// RegisterModuleForwardHook registers a global forward hook that runs (before
// per-module forward hooks) for every module invoked through the hook
// pipeline. Remove with RemoveModuleHook(id).
func RegisterModuleForwardHook(fn ForwardHook) int {
	id := newHookID()
	globalHooks.fwd = append(globalHooks.fwd, fwdHookEntry{id: id, fn: fn})
	return id
}

// RegisterModuleFullBackwardHook registers a global backward hook that runs
// (before per-module backward hooks) for every module invoked through the
// hook pipeline whose output requires grad. Remove with RemoveModuleHook(id).
func RegisterModuleFullBackwardHook(fn BackwardHook) int {
	id := newHookID()
	globalHooks.bwd = append(globalHooks.bwd, bwdHookEntry{id: id, fn: fn})
	return id
}

// RemoveModuleHook removes the global hook with the given id, reporting
// whether it was found.
func RemoveModuleHook(id int) bool { return globalHooks.remove(id) }

// Call invokes m through the hook pipeline: global then per-module forward
// pre-hooks (which may replace the input), m.Forward, global then per-module
// forward hooks (which may replace the output), and finally — if any backward
// hooks are registered — an identity autograd node whose backward runs the
// backward hooks on the output gradient.
//
// This is the GoNN stand-in for PyTorch's module(x) __call__ sugar: bare
// m.Forward(x) bypasses hooks entirely. With no hooks registered anywhere,
// Call is a nil-check away from plain m.Forward(x).
func Call(m Module, x *tensor.Tensor) *tensor.Tensor {
	var mh *hookSet
	if hh, ok := m.(hookHost); ok {
		mh = hh.hookState()
	}
	if globalHooks.empty() && mh.empty() {
		return m.Forward(x)
	}

	for _, e := range globalHooks.pre {
		if r := e.fn(m, x); r != nil {
			x = r
		}
	}
	if mh != nil {
		for _, e := range mh.pre {
			if r := e.fn(m, x); r != nil {
				x = r
			}
		}
	}

	y := m.Forward(x)

	for _, e := range globalHooks.fwd {
		if r := e.fn(m, x, y); r != nil {
			y = r
		}
	}
	if mh != nil {
		for _, e := range mh.fwd {
			if r := e.fn(m, x, y); r != nil {
				y = r
			}
		}
	}

	nBwd := len(globalHooks.bwd)
	if mh != nil {
		nBwd += len(mh.bwd)
	}
	if nBwd > 0 {
		hooks := make([]BackwardHook, 0, nBwd)
		for _, e := range globalHooks.bwd {
			hooks = append(hooks, e.fn)
		}
		if mh != nil {
			for _, e := range mh.bwd {
				hooks = append(hooks, e.fn)
			}
		}
		y = attachBackwardHooks(m, y, hooks)
	}
	return y
}

// attachBackwardHooks wraps y in an identity node (shared data, fresh
// shape/strides) whose backward runs each hook on the incoming gradient —
// each may replace it — and passes the result through to y. If y does not
// require grad the node is not attached and the hooks never fire, matching
// PyTorch's full-backward-hook skip rule. The hook list is snapshotted at
// call time, so removing a hook between forward and backward does not
// unregister it from an already-built graph.
func attachBackwardHooks(m Child, y *tensor.Tensor, hooks []BackwardHook) *tensor.Tensor {
	out := &tensor.Tensor{
		Data:    y.Data,
		Shape:   append([]int(nil), y.Shape...),
		Strides: append([]int(nil), y.Strides...),
		Dtype:   y.Dtype,
	}
	tensor.MakeNode(out, "nn.BackwardHook", []*tensor.Tensor{y}, func(grad *tensor.Tensor) []*tensor.Tensor {
		g := grad
		for _, h := range hooks {
			if r := h(m, g); r != nil {
				g = r
			}
		}
		return []*tensor.Tensor{g}
	})
	return out
}
