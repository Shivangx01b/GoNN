package nn

// Stateless functional calls — an honest Go adaptation of
// torch.func.functional_call (formerly torch.nn.utils.stateless).
//
// PyTorch's functional_call(module, params_and_buffers, args) runs the module
// with the given tensors substituted for its parameters/buffers by dotted
// name; gradients flow to the REPLACEMENT tensors, and the module is left
// untouched.
//
// GoNN's autograd tracks tensor IDENTITY: the graph's leaves are the module's
// own parameter tensors, and there is no reparametrization pass that could
// substitute graph leaves. The adaptation therefore swaps the Data slice (and
// Grad pointer) of the selected module parameters IN PLACE for the duration
// of the call and restores the originals before returning (defer, so
// restoration survives panics). The consequences — which differ from PyTorch
// in load-bearing ways — are documented on FunctionalCall; use
// FunctionalCallGrad whenever you need gradients.
//
// Related non-feature, for the parity doc: torch.nn.utils.skip_init is
// intentionally N/A in GoNN. skip_init exists to avoid paying an expensive
// initialization for parameters that are about to be overwritten (PyTorch
// constructs on a meta device and re-materializes). GoNN has no meta device
// and constructor init is a cheap flat-slice fill, so there is nothing worth
// skipping — overwrite param.Data after construction instead.

import (
	"fmt"

	"gonn/tensor"
)

// paramSlot records one swapped tensor's original Data and Grad for restore.
type paramSlot struct {
	t    *tensor.Tensor
	data []float64
	grad *tensor.Tensor
}

// WithReplacedParams runs fn while the parameters (or buffers) of m selected
// by dotted name have their Data swapped for the replacement tensors' Data.
// The swapped tensors' Grad pointers are cleared for the duration of the
// window and restored afterwards; restoration runs in a defer (LIFO), so the
// originals come back even if fn panics.
//
// Names are the dotted paths returned by NamedParameters() / Buffers()
// ("0.weight", "encoder.bias", "running_mean", ...). Validation happens
// before any swap: an unknown name, a nil replacement, or a replacement
// whose shape differs from the module tensor's shape panics, leaving the
// module untouched.
//
// Tied tensors (one tensor registered under several names): each name in
// replace triggers its own swap; LIFO restore brings back the true original,
// but supplying DIFFERENT replacements for two names of the same tensor has
// last-swap-wins semantics during the window (PyTorch with tie_weights=True
// raises instead).
//
// This is the primitive FunctionalCall and FunctionalCallGrad build on.
func WithReplacedParams(m Child, replace map[string]*tensor.Tensor, fn func()) {
	if len(replace) == 0 {
		fn()
		return
	}
	byName := make(map[string]*tensor.Tensor)
	for _, p := range m.NamedParameters() {
		byName[p.Name] = p.T
	}
	for _, b := range m.Buffers() {
		if _, exists := byName[b.Name]; !exists {
			byName[b.Name] = b.T
		}
	}
	// Validate everything up front so a bad entry never leaves a partial swap.
	for name, r := range replace {
		t, ok := byName[name]
		if !ok {
			panic(fmt.Sprintf("nn: WithReplacedParams: module has no parameter or buffer named %q", name))
		}
		if r == nil {
			panic(fmt.Sprintf("nn: WithReplacedParams: replacement for %q is nil", name))
		}
		if !intsEqual(t.Shape, r.Shape) {
			panic(fmt.Sprintf("nn: WithReplacedParams: shape mismatch for %q: module %v, replacement %v",
				name, t.Shape, r.Shape))
		}
	}

	saved := make([]paramSlot, 0, len(replace))
	defer func() {
		for i := len(saved) - 1; i >= 0; i-- {
			s := saved[i]
			s.t.Data = s.data
			s.t.Grad = s.grad
		}
	}()
	for name, r := range replace {
		t := byName[name]
		saved = append(saved, paramSlot{t: t, data: t.Data, grad: t.Grad})
		t.Data = r.Data
		t.Grad = nil
	}
	fn()
}

// FunctionalCall runs m.Forward(x) — through nn.Call, so forward/backward
// hooks fire — with the named parameters temporarily replaced, restoring the
// originals before returning. It is the adaptation of
// torch.func.functional_call(m, replace, x).
//
// DOCUMENTED DEVIATIONS from PyTorch — sharp edges, read before use:
//
//  1. Gradients flow to the MODULE's parameter tensors, not to the
//     replacement tensors. GoNN autograd tracks tensor identity, so the
//     graph's leaves are the module's own parameters (whose Data was swapped
//     for the window); the replacement tensors never enter the graph and
//     never receive a Grad.
//  2. Do NOT call Backward on the returned tensor after FunctionalCall has
//     returned. Restoration has already put the ORIGINAL data back into the
//     parameter tensors, and backward formulas re-read those tensors' Data
//     (e.g. MatMul's dA = grad @ B^T reads B.Data at backward time), so a
//     late Backward silently mixes replacement-value forward activations
//     with original-value parameter data and produces garbage gradients.
//     Run forward AND backward inside the swap window instead — that is
//     exactly what FunctionalCallGrad does.
//  3. Shapes must match exactly and unknown names panic (see
//     WithReplacedParams). Like PyTorch's default strict=False, replacing a
//     SUBSET of the parameters is fine; unlike PyTorch, a name the module
//     does not have is an error rather than being ignored.
//  4. Buffer replacement is supported through the same map (dotted buffer
//     names), matching functional_call's params_and_buffers dict.
func FunctionalCall(m Module, x *tensor.Tensor, replace map[string]*tensor.Tensor) *tensor.Tensor {
	var y *tensor.Tensor
	WithReplacedParams(m, replace, func() { y = Call(m, x) })
	return y
}

// FunctionalCallOpt configures the gradient-taking functional calls.
type FunctionalCallOpt func(*fcOpts)

type fcOpts struct {
	gradsToReplacements bool
}

// WithGradsToReplacements makes FunctionalCallGrad(Multi) additionally
// ACCUMULATE each swapped name's gradient into the corresponding replacement
// tensor's .Grad (allocated if nil) — giving PyTorch functional_call
// semantics, where gradients w.r.t. the supplied parameters live on the
// supplied tensors. The returned map is unchanged; the module's own
// parameters still end the call with their original Grad pointers.
func WithGradsToReplacements() FunctionalCallOpt {
	return func(o *fcOpts) { o.gradsToReplacements = true }
}

// functionalGrad is the shared core of FunctionalCallGrad/-Multi: it runs
// compute() (which must build the graph and return a SCALAR loss) inside the
// swap window with every parameter's pre-existing Grad parked, backwards it,
// and harvests copies of the gradients by dotted name.
func functionalGrad(m Child, replace map[string]*tensor.Tensor,
	compute func() *tensor.Tensor, opts []FunctionalCallOpt) (lossVal float64, grads map[string][]float64) {

	var o fcOpts
	for _, fn := range opts {
		fn(&o)
	}
	grads = make(map[string][]float64)
	WithReplacedParams(m, replace, func() {
		// Park every parameter's current Grad and accumulate into fresh ones,
		// so pre-existing gradients neither pollute nor are polluted by this
		// call. (For swapped parameters WithReplacedParams already cleared
		// Grad; parking nil and restoring nil is harmless — the outer defer
		// restores their true original afterwards.)
		params := m.NamedParameters()
		savedGrads := make([]*tensor.Tensor, len(params))
		for i, p := range params {
			savedGrads[i] = p.T.Grad
			p.T.Grad = nil
		}
		defer func() {
			for i, p := range params {
				p.T.Grad = savedGrads[i]
			}
		}()

		l := compute()
		lossVal = l.Item() // also validates the loss is scalar
		l.Backward()
		for _, p := range params {
			if p.T.Grad != nil {
				grads[p.Name] = append([]float64(nil), p.T.Grad.Data...)
			}
		}
	})
	if o.gradsToReplacements {
		for name, r := range replace {
			g, ok := grads[name]
			if !ok {
				continue
			}
			if r.Grad == nil {
				r.Grad = tensor.Zeros(r.Shape...)
			}
			for i := range g {
				r.Grad.Data[i] += g[i]
			}
		}
	}
	return lossVal, grads
}

// FunctionalCallGrad is the safe gradient-taking variant of FunctionalCall:
// it runs forward (through nn.Call) AND backward entirely inside the swap
// window, so the backward pass sees the replacement data it needs. It
// returns the scalar loss value and deep copies of the gradients of ALL of
// m's named parameters — computed w.r.t. the replaced values for swapped
// parameters and the module's own values for the rest — keyed by dotted
// name. Parameters that received no gradient are omitted from the map.
// With WithGradsToReplacements, the swapped names' gradients are also
// accumulated into the replacement tensors' .Grad (PyTorch semantics).
//
// The module is left exactly as found: parameter Data, and every parameter's
// Grad pointer (including pre-existing gradients), are restored before
// returning; the gradients accumulated during the call live only in the
// returned map (and, optionally, on the replacements).
//
// loss must map the module output to a scalar tensor (e.g. mean-squared
// error against a captured target).
func FunctionalCallGrad(m Module, x *tensor.Tensor, replace map[string]*tensor.Tensor,
	loss func(y *tensor.Tensor) *tensor.Tensor, opts ...FunctionalCallOpt) (lossVal float64, grads map[string][]float64) {

	return functionalGrad(m, replace, func() *tensor.Tensor { return loss(Call(m, x)) }, opts)
}

// FunctionalCallMulti is FunctionalCall for modules whose Forward takes more
// than one input (MultiHeadAttention, Bilinear, Seq2Seq, ...): the caller
// closes over the inputs and invokes the module inside forward. Hooks do NOT
// fire automatically (the module is invoked by the closure, not by nn.Call —
// wrap with nn.Call inside the closure for single-input children if needed).
// The same sharp edges as FunctionalCall apply: do not Backward the returned
// tensor after the call has returned — use FunctionalCallGradMulti.
func FunctionalCallMulti(m Child, replace map[string]*tensor.Tensor,
	forward func() *tensor.Tensor) *tensor.Tensor {

	var y *tensor.Tensor
	WithReplacedParams(m, replace, func() { y = forward() })
	return y
}

// FunctionalCallGradMulti is FunctionalCallGrad for multi-input modules: run
// must build the full graph — module invocation AND loss — and return the
// scalar loss tensor. Forward and backward both happen inside the swap
// window; gradients are returned by dotted name (and accumulated onto the
// replacements with WithGradsToReplacements).
func FunctionalCallGradMulti(m Child, replace map[string]*tensor.Tensor,
	run func() *tensor.Tensor, opts ...FunctionalCallOpt) (lossVal float64, grads map[string][]float64) {

	return functionalGrad(m, replace, run, opts)
}
