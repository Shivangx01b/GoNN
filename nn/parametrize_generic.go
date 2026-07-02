package nn

// Generic weight parametrizations — an honest Go adaptation of
// torch.nn.utils.parametrize.register_parametrization.
//
// PyTorch injects parametrizations into an existing module: module.weight
// becomes a computed property, the raw tensor moves to
// module.parametrizations.weight.original, and every attribute access
// recomputes weight = p_n(...p_2(p_1(original))). Go has no properties or
// __getattr__, so — exactly like the fixed-function wrappers in
// parametrize.go (WeightNormLinear, SpectralNormLinear, ...) — a
// parametrization here is an explicit WRAPPER MODULE that owns the raw
// tensor ("weight_orig", the analog of parametrizations.weight.original) and
// recomputes the effective weight with differentiable tensor ops on every
// Forward. This file generalizes that pattern to arbitrary user-supplied
// parametrizations.
//
// DELIBERATE DEVIATIONS from PyTorch (documented loudly on purpose):
//
//   - No property injection: wrap the layer (NewParametrizedLinear /
//     NewParametrizedConv2d) and use the wrapper in place of the original
//     layer from then on. The wrapped layer's own registered weight is never
//     mutated and is NOT part of the wrapper's parameters; the wrapper's
//     Parameters() are what the optimizer should train.
//   - right_inverse IS supported via the optional ParametrizationWithInverse
//     interface: when the initial parametrization implements it, weight_orig
//     is initialized to RightInverse(weight) so Apply(weight_orig) == weight
//     and wrapping PRESERVES the layer's function (PyTorch semantics). A
//     parametrization without an inverse initializes weight_orig to the
//     weight verbatim, which generally changes the function — documented on
//     the constructors. SetEffectiveWeight assigns a new effective value
//     through the chain's inverses (PyTorch's `module.weight = value`).
//   - Parametrizations must preserve shape. PyTorch allows chains that change
//     shape between links (only the final shape must fit the module); here
//     every link must map (shape) -> (same shape), validated on each Forward.
//   - parametrize.cached() has an explicit analog: (m).Cached(fn) computes
//     the effective weight/bias once and reuses them for every Forward inside
//     fn (see Cached; ParametrizeCachedAll nests several modules' windows).
//   - No unsafe flag: the only consistency check is the shape check above.
//   - RemoveParametrizations returns a NEW plain layer instead of mutating
//     the module in place (concrete Go structs cannot change type).
//   - IsParametrized reports true only for the generic wrappers in this
//     file; the fixed-function wrappers in parametrize.go (WeightNorm*,
//     SpectralNorm*) predate the generic mechanism and report false.

import (
	"fmt"

	"gonn/tensor"
)

// Parametrization transforms a raw ("original") tensor into the effective
// tensor a layer actually uses — the analog of a torch.nn.Module passed to
// register_parametrization. Apply MUST be built from differentiable tensor
// ops (autograd must flow back to orig) and MUST return a tensor with the
// same shape as orig.
type Parametrization interface {
	Apply(orig *tensor.Tensor) *tensor.Tensor
}

// ParametrizationFunc adapts a plain function to the Parametrization
// interface:
//
//	positive := nn.ParametrizationFunc((*tensor.Tensor).Softplus)
type ParametrizationFunc func(*tensor.Tensor) *tensor.Tensor

// Apply implements Parametrization.
func (f ParametrizationFunc) Apply(orig *tensor.Tensor) *tensor.Tensor { return f(orig) }

// ParametrizationWithInverse is the optional right_inverse extension of
// Parametrization (torch's parametrization.right_inverse): RightInverse maps
// an effective value back to a raw one such that Apply(RightInverse(v)) ≈ v.
// It is used for function-preserving initialization, for AddParametrization's
// effective-weight preservation, and for SetEffectiveWeight assignment.
// RightInverse runs OUTSIDE autograd (plain values; PyTorch likewise runs it
// under no_grad) and must return a tensor of the same shape.
type ParametrizationWithInverse interface {
	Parametrization
	RightInverse(value *tensor.Tensor) *tensor.Tensor
}

// InvertibleParametrization bundles two plain functions into a
// ParametrizationWithInverse:
//
//	positive := nn.InvertibleParametrization{
//	    Fwd: (*tensor.Tensor).Softplus,
//	    Inv: func(v *tensor.Tensor) *tensor.Tensor { // log(exp(v)-1)
//	        return v.Exp().SubScalar(1).Log()
//	    },
//	}
type InvertibleParametrization struct {
	Fwd func(*tensor.Tensor) *tensor.Tensor
	Inv func(*tensor.Tensor) *tensor.Tensor
}

// Apply implements Parametrization.
func (p InvertibleParametrization) Apply(orig *tensor.Tensor) *tensor.Tensor { return p.Fwd(orig) }

// RightInverse implements ParametrizationWithInverse.
func (p InvertibleParametrization) RightInverse(v *tensor.Tensor) *tensor.Tensor { return p.Inv(v) }

// initialOrig computes the raw tensor a wrapper should own for a layer whose
// current effective value is `value`: RightInverse(value) when p supports it
// (function-preserving), else a verbatim copy.
func initialOrig(p Parametrization, value *tensor.Tensor) *tensor.Tensor {
	if pi, ok := p.(ParametrizationWithInverse); ok {
		inv := pi.RightInverse(value)
		if inv == nil || !intsEqual(inv.Shape, value.Shape) {
			panic("nn: RightInverse changed shape; parametrizations must preserve shape")
		}
		return inv.Copy().SetRequiresGrad(true)
	}
	return value.Copy().SetRequiresGrad(true)
}

// inverseChain maps an effective value back through ps' inverses in REVERSE
// order (orig = p1⁻¹(p2⁻¹(...pn⁻¹(value)))). Panics (with the failing link's
// index) unless every link implements ParametrizationWithInverse.
func inverseChain(ps []Parametrization, value *tensor.Tensor) *tensor.Tensor {
	v := value
	for i := len(ps) - 1; i >= 0; i-- {
		pi, ok := ps[i].(ParametrizationWithInverse)
		if !ok {
			panic(fmt.Sprintf("nn: parametrization %d does not implement ParametrizationWithInverse; cannot assign through the chain", i))
		}
		v = pi.RightInverse(v)
		if v == nil || !intsEqual(v.Shape, value.Shape) {
			panic(fmt.Sprintf("nn: RightInverse of parametrization %d changed shape", i))
		}
	}
	return v
}

// allInvertible reports whether every link supports RightInverse.
func allInvertible(ps []Parametrization) bool {
	for _, p := range ps {
		if _, ok := p.(ParametrizationWithInverse); !ok {
			return false
		}
	}
	return true
}

// Cacheable is implemented by the parametrized wrappers: Cached(fn) computes
// the effective tensors once and reuses them for every Forward inside fn —
// the analog of torch.nn.utils.parametrize.cached().
//
// Gradient contract inside a window: each Backward must complete before the
// next Forward (the usual train-step shape). The cached weight is an interior
// node shared across the graphs; a completed Backward leaves its consumed
// gradient on the node, so the next cache-hit Forward clears it — otherwise
// the stale value would be re-propagated and double-counted. Building several
// graphs first and backwarding them afterwards is NOT supported inside a
// window (combine them into one loss and call Backward once instead).
type Cacheable interface{ Cached(fn func()) }

// clearConsumedGrad resets an interior cached node's gradient between
// forward+backward pairs (see the Cacheable gradient contract).
func clearConsumedGrad(t *tensor.Tensor) {
	if t != nil {
		t.Grad = nil
	}
}

// ParametrizeCachedAll nests the Cached windows of several parametrized
// modules around fn.
func ParametrizeCachedAll(fn func(), ms ...Cacheable) {
	if len(ms) == 0 {
		fn()
		return
	}
	ms[0].Cached(func() { ParametrizeCachedAll(fn, ms[1:]...) })
}

// applyParametrizationChain applies ps in registration order (ps[0] first,
// like PyTorch's ParametrizationList) and validates that every link
// preserves the shape.
func applyParametrizationChain(ps []Parametrization, orig *tensor.Tensor) *tensor.Tensor {
	w := orig
	for i, p := range ps {
		w = p.Apply(w)
		if w == nil || !intsEqual(w.Shape, orig.Shape) {
			panic(fmt.Sprintf(
				"nn: parametrization %d changed shape %v -> %v; parametrizations must preserve shape",
				i, orig.Shape, shapeOrNil(w)))
		}
	}
	return w
}

func shapeOrNil(t *tensor.Tensor) []int {
	if t == nil {
		return nil
	}
	return t.Shape
}

// parametrizedModule is the marker interface behind IsParametrized.
type parametrizedModule interface{ isParametrizedModule() }

// IsParametrized reports whether c is one of the generic parametrized
// wrappers (the analog of torch.nn.utils.parametrize.is_parametrized).
// Deviation: the fixed-function WeightNorm*/SpectralNorm* wrappers report
// false — they predate the generic mechanism.
func IsParametrized(c Child) bool {
	_, ok := c.(parametrizedModule)
	return ok
}

// ---- ParametrizedLinear -------------------------------------------------------

// ParametrizedLinear wraps a Linear layer with a chain of weight
// parametrizations: the wrapper owns weight_orig (initialized from the
// wrapped layer's current weight) and computes
//
//	W = p_n(...p_1(weight_orig)),   y = x @ W^T + b
//
// on every Forward (same math as Linear.Forward — leading dims flattened).
// If a bias parametrization is given the wrapper also owns bias_orig and
// recomputes the effective bias the same way; otherwise the wrapped layer's
// bias tensor is shared (registered here as "bias").
type ParametrizedLinear struct {
	Base
	InFeatures  int
	OutFeatures int
	WeightOrig  *tensor.Tensor // owned raw weight ("weight_orig"), (Out, In)
	BiasOrig    *tensor.Tensor // owned raw bias ("bias_orig") when the bias is parametrized, else nil
	Bias        *tensor.Tensor // shared with the wrapped layer when the bias is NOT parametrized, else nil
	weightPs    []Parametrization
	biasPs      []Parametrization

	cachedW, cachedB *tensor.Tensor // non-nil only inside a Cached window
}

// NewParametrizedLinear wraps l with weightP. When weightP implements
// ParametrizationWithInverse, weight_orig is initialized to
// RightInverse(l.Weight) so the wrapper computes the SAME function as l
// (PyTorch's right_inverse initialization); otherwise weight_orig is a
// verbatim copy and the function generally changes. Chain further weight
// parametrizations with AddParametrization. At most one optional bias
// parametrization may be given as a trailing argument; it turns the bias into
// an owned bias_orig as well (same right_inverse rule).
func NewParametrizedLinear(l *Linear, weightP Parametrization, biasP ...Parametrization) *ParametrizedLinear {
	if weightP == nil {
		panic("nn: NewParametrizedLinear requires a non-nil weight parametrization")
	}
	if len(biasP) > 1 {
		panic("nn: NewParametrizedLinear accepts at most one bias parametrization")
	}
	m := &ParametrizedLinear{InFeatures: l.InFeatures, OutFeatures: l.OutFeatures}
	m.WeightOrig = m.reg("weight_orig", initialOrig(weightP, l.Weight))
	m.weightPs = []Parametrization{weightP}
	if len(biasP) == 1 && biasP[0] != nil {
		if l.Bias == nil {
			panic("nn: bias parametrization given but the wrapped Linear has no bias")
		}
		m.BiasOrig = m.reg("bias_orig", initialOrig(biasP[0], l.Bias))
		m.biasPs = []Parametrization{biasP[0]}
	} else if l.Bias != nil {
		m.Bias = m.reg("bias", l.Bias)
	}
	return m
}

func (m *ParametrizedLinear) isParametrizedModule() {}

// AddParametrization appends p to the WEIGHT parametrization chain. Chains
// apply in registration order (first registered runs first), mirroring
// PyTorch's ParametrizationList. When p AND every existing link implement
// ParametrizationWithInverse, weight_orig is re-derived through the new
// chain's inverses so the module's CURRENT EFFECTIVE WEIGHT — and therefore
// its function — is preserved across the append; otherwise weight_orig is
// left unchanged and the function changes (documented deviation: PyTorch's
// append rule differs in detail). Returns m for call chaining.
func (m *ParametrizedLinear) AddParametrization(p Parametrization) *ParametrizedLinear {
	if p == nil {
		panic("nn: AddParametrization: nil parametrization")
	}
	newChain := append(append([]Parametrization(nil), m.weightPs...), p)
	if allInvertible(newChain) {
		eff := m.EffectiveWeight()
		copy(m.WeightOrig.Data, inverseChain(newChain, eff).Data)
	}
	m.weightPs = newChain
	return m
}

// SetEffectiveWeight assigns a new EFFECTIVE weight (the analog of PyTorch's
// `module.weight = value` on a parametrized module): value is mapped through
// the chain's right inverses in reverse order and written into weight_orig
// in place (no autograd — assignment, not an op). Every link must implement
// ParametrizationWithInverse; the shape must match.
func (m *ParametrizedLinear) SetEffectiveWeight(value *tensor.Tensor) {
	if !intsEqual(value.Shape, m.WeightOrig.Shape) {
		panic(fmt.Sprintf("nn: SetEffectiveWeight: shape %v != weight shape %v", value.Shape, m.WeightOrig.Shape))
	}
	copy(m.WeightOrig.Data, inverseChain(m.weightPs, value).Data)
}

// Cached computes the effective weight (and parametrized bias) ONCE and
// reuses them for every Forward inside fn — the analog of
// torch.nn.utils.parametrize.cached(). The cache is invalidated when fn
// returns (defer-safe). Autograd-safe across multiple forward+backward
// passes inside the window: each Backward re-walks the cached tensors'
// creators and accumulates into weight_orig/bias_orig.
func (m *ParametrizedLinear) Cached(fn func()) {
	m.cachedW = applyParametrizationChain(m.weightPs, m.WeightOrig)
	if m.BiasOrig != nil {
		m.cachedB = applyParametrizationChain(m.biasPs, m.BiasOrig)
	}
	defer func() { m.cachedW, m.cachedB = nil, nil }()
	fn()
}

// EffectiveWeight recomputes the parametrized weight with differentiable ops
// (inside a Cached window it returns the cached tensor).
func (m *ParametrizedLinear) EffectiveWeight() *tensor.Tensor {
	if m.cachedW != nil {
		return m.cachedW
	}
	return applyParametrizationChain(m.weightPs, m.WeightOrig)
}

// EffectiveBias returns the parametrized bias (chain over bias_orig) when the
// bias is parametrized, the shared raw bias otherwise, or nil for no bias.
func (m *ParametrizedLinear) EffectiveBias() *tensor.Tensor {
	if m.BiasOrig != nil {
		if m.cachedB != nil {
			return m.cachedB
		}
		return applyParametrizationChain(m.biasPs, m.BiasOrig)
	}
	return m.Bias
}

// Forward computes x @ W^T + b with the recomputed effective weight/bias.
func (m *ParametrizedLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	if m.cachedW != nil {
		clearConsumedGrad(m.cachedW)
		clearConsumedGrad(m.cachedB)
	}
	return applyLinear(x, m.EffectiveWeight(), m.EffectiveBias(), m.InFeatures, m.OutFeatures)
}

// hasBias reports whether the wrapper carries any bias (owned or shared).
func (m *ParametrizedLinear) hasBias() bool { return m.BiasOrig != nil || m.Bias != nil }

// RemoveParametrizationsLinear dissolves the wrapper into a fresh plain
// Linear — the analog of torch.nn.utils.parametrize.remove_parametrizations.
// leaveParametrized=true bakes the CURRENT EFFECTIVE weight/bias in (the
// returned layer computes the same function the wrapper does right now);
// leaveParametrized=false copies the raw weight_orig/bias_orig instead,
// discarding the parametrization.
func RemoveParametrizationsLinear(m *ParametrizedLinear, leaveParametrized bool) *Linear {
	l := NewLinear(m.InFeatures, m.OutFeatures, m.hasBias())
	w := m.WeightOrig
	if leaveParametrized {
		w = m.EffectiveWeight()
	}
	copy(l.Weight.Data, w.Data)
	if l.Bias != nil {
		b := m.EffectiveBias()
		if !leaveParametrized {
			if m.BiasOrig != nil {
				b = m.BiasOrig
			} else {
				b = m.Bias
			}
		}
		copy(l.Bias.Data, b.Data)
	}
	return l
}

// ---- ParametrizedConv2d -------------------------------------------------------

// ParametrizedConv2d wraps a Conv2d the same way ParametrizedLinear wraps a
// Linear: the wrapper owns weight_orig ((OutC, InC/groups, KH, KW), derived
// from the conv's current weight — via RightInverse when the first
// parametrization provides one), recomputes W = chain(weight_orig) each
// Forward, and runs the wrapped conv's im2col + GEMM math (its unfold
// machinery, gather cache, padding_mode pre-pad, and grouped path) with that
// weight. The wrapped conv's own registered weight is never touched.
type ParametrizedConv2d struct {
	Base
	inner      *Conv2d
	WeightOrig *tensor.Tensor // owned raw weight ("weight_orig"), (OutC, InC/groups, KH, KW)
	BiasOrig   *tensor.Tensor // owned raw bias ("bias_orig") when the bias is parametrized, else nil
	Bias       *tensor.Tensor // shared with the wrapped layer when the bias is NOT parametrized, else nil
	weightPs   []Parametrization
	biasPs     []Parametrization

	cachedW, cachedB *tensor.Tensor // non-nil only inside a Cached window
}

// NewParametrizedConv2d wraps c with weightP (right_inverse initialization
// when available — see NewParametrizedLinear). Chain further weight
// parametrizations with AddParametrization. At most one optional bias
// parametrization may be given as a trailing argument. Grouped convolutions
// are supported.
func NewParametrizedConv2d(c *Conv2d, weightP Parametrization, biasP ...Parametrization) *ParametrizedConv2d {
	if weightP == nil {
		panic("nn: NewParametrizedConv2d requires a non-nil weight parametrization")
	}
	if len(biasP) > 1 {
		panic("nn: NewParametrizedConv2d accepts at most one bias parametrization")
	}
	m := &ParametrizedConv2d{inner: c}
	m.WeightOrig = m.reg("weight_orig", initialOrig(weightP, c.Weight))
	m.weightPs = []Parametrization{weightP}
	if len(biasP) == 1 && biasP[0] != nil {
		if c.Bias == nil {
			panic("nn: bias parametrization given but the wrapped Conv2d has no bias")
		}
		m.BiasOrig = m.reg("bias_orig", initialOrig(biasP[0], c.Bias))
		m.biasPs = []Parametrization{biasP[0]}
	} else if c.Bias != nil {
		m.Bias = m.reg("bias", c.Bias)
	}
	return m
}

func (m *ParametrizedConv2d) isParametrizedModule() {}

// AddParametrization appends p to the WEIGHT parametrization chain (applied
// in registration order), preserving the current effective weight when the
// whole new chain is invertible — see ParametrizedLinear.AddParametrization.
// Returns m for call chaining.
func (m *ParametrizedConv2d) AddParametrization(p Parametrization) *ParametrizedConv2d {
	if p == nil {
		panic("nn: AddParametrization: nil parametrization")
	}
	newChain := append(append([]Parametrization(nil), m.weightPs...), p)
	if allInvertible(newChain) {
		eff := m.EffectiveWeight()
		copy(m.WeightOrig.Data, inverseChain(newChain, eff).Data)
	}
	m.weightPs = newChain
	return m
}

// SetEffectiveWeight assigns a new effective weight through the chain's
// right inverses — see ParametrizedLinear.SetEffectiveWeight.
func (m *ParametrizedConv2d) SetEffectiveWeight(value *tensor.Tensor) {
	if !intsEqual(value.Shape, m.WeightOrig.Shape) {
		panic(fmt.Sprintf("nn: SetEffectiveWeight: shape %v != weight shape %v", value.Shape, m.WeightOrig.Shape))
	}
	copy(m.WeightOrig.Data, inverseChain(m.weightPs, value).Data)
}

// Cached computes the effective weight/bias once for every Forward inside fn
// — see ParametrizedLinear.Cached.
func (m *ParametrizedConv2d) Cached(fn func()) {
	m.cachedW = applyParametrizationChain(m.weightPs, m.WeightOrig)
	if m.BiasOrig != nil {
		m.cachedB = applyParametrizationChain(m.biasPs, m.BiasOrig)
	}
	defer func() { m.cachedW, m.cachedB = nil, nil }()
	fn()
}

// EffectiveWeight recomputes the parametrized weight with differentiable ops
// (inside a Cached window it returns the cached tensor).
func (m *ParametrizedConv2d) EffectiveWeight() *tensor.Tensor {
	if m.cachedW != nil {
		return m.cachedW
	}
	return applyParametrizationChain(m.weightPs, m.WeightOrig)
}

// EffectiveBias returns the parametrized bias when the bias is parametrized,
// the shared raw bias otherwise, or nil for no bias.
func (m *ParametrizedConv2d) EffectiveBias() *tensor.Tensor {
	if m.BiasOrig != nil {
		if m.cachedB != nil {
			return m.cachedB
		}
		return applyParametrizationChain(m.biasPs, m.BiasOrig)
	}
	return m.Bias
}

// Forward runs the wrapped conv's full forward math — padding_mode pre-pad,
// gather cache, unfold, grouped GEMMs — with the recomputed effective weight.
func (m *ParametrizedConv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if m.cachedW != nil {
		clearConsumedGrad(m.cachedW)
		clearConsumedGrad(m.cachedB)
	}
	return convForwardWithWeightFull(&m.inner.convNd, x, m.EffectiveWeight(), m.EffectiveBias())
}

// convForwardWithWeightFull mirrors convNd.Forward with an explicit
// (OutC, InC/groups, K...) weight: padding_mode pre-pad, single or grouped
// unfold + GEMM, bias, channels-first restore. c's own weight is untouched;
// its gather/pad caches are shared.
func convForwardWithWeightFull(c *convNd, x, w, bias *tensor.Tensor) *tensor.Tensor {
	rank := len(c.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: conv expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	if x.Shape[1] != c.InC {
		panic(fmt.Sprintf("nn: conv input channels %d != InC %d", x.Shape[1], c.InC))
	}
	if c.PaddingMode != "" && c.PaddingMode != "zeros" && !allZeroInts(c.Pad) {
		x = c.prePad(x)
	}
	N := x.Shape[0]
	g, out, numWin, winSize := c.gatherFor(x.Shape[2:])

	var y *tensor.Tensor
	if c.Groups == 1 {
		col := unfold(x, g, numWin, winSize)
		y = col.MatMul(w.Reshape(c.OutC, c.InC*winSize).Transpose())
	} else {
		inCg, outCg := c.InC/c.Groups, c.OutC/c.Groups
		parts := make([]*tensor.Tensor, c.Groups)
		for i := 0; i < c.Groups; i++ {
			xi := x.IndexSelect(1, rangeIndex(i*inCg, inCg))
			col := unfold(xi, g, numWin, winSize)
			wi := w.IndexSelect(0, rangeIndex(i*outCg, outCg)).Reshape(outCg, inCg*winSize)
			parts[i] = col.MatMul(wi.Transpose())
		}
		y = tensor.Concat(1, parts...)
	}
	if bias != nil {
		y = y.Add(bias)
	}
	shape := append(append([]int{N}, out...), c.OutC)
	perm := make([]int, rank+2)
	perm[0], perm[1] = 0, rank+1
	for d := 0; d < rank; d++ {
		perm[2+d] = 1 + d
	}
	return y.Reshape(shape...).Permute(perm...)
}

func (m *ParametrizedConv2d) hasBias() bool { return m.BiasOrig != nil || m.Bias != nil }

// RemoveParametrizationsConv2d dissolves the wrapper into a fresh plain
// Conv2d with the same geometry. See RemoveParametrizationsLinear for the
// leaveParametrized semantics.
func RemoveParametrizationsConv2d(m *ParametrizedConv2d, leaveParametrized bool) *Conv2d {
	c := newConv2dLike(m.inner, m.hasBias())
	w := m.WeightOrig
	if leaveParametrized {
		w = m.EffectiveWeight()
	}
	copy(c.Weight.Data, w.Data)
	if c.Bias != nil {
		b := m.EffectiveBias()
		if !leaveParametrized {
			if m.BiasOrig != nil {
				b = m.BiasOrig
			} else {
				b = m.Bias
			}
		}
		copy(c.Bias.Data, b.Data)
	}
	return c
}

// RemoveParametrizations dissolves a generic parametrized wrapper into a
// plain layer — the analog of
// torch.nn.utils.parametrize.remove_parametrizations(module, "weight",
// leave_parametrized=...). c must be a *ParametrizedLinear or
// *ParametrizedConv2d; anything else panics. Go deviation: PyTorch mutates
// the module in place, Go returns a NEW layer (type-assert the result, or
// use the typed RemoveParametrizationsLinear / RemoveParametrizationsConv2d
// directly).
func RemoveParametrizations(c Child, leaveParametrized bool) Module {
	switch m := c.(type) {
	case *ParametrizedLinear:
		return RemoveParametrizationsLinear(m, leaveParametrized)
	case *ParametrizedConv2d:
		return RemoveParametrizationsConv2d(m, leaveParametrized)
	default:
		panic(fmt.Sprintf("nn: RemoveParametrizations: %T is not a generic parametrized module", c))
	}
}
