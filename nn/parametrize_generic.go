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
//   - No right_inverse: weight_orig is initialized to the wrapped layer's
//     current weight VERBATIM. PyTorch calls right_inverse(weight) when
//     available so that forward(original) == weight and wrapping preserves
//     the layer's function; here wrapping generally CHANGES the function
//     (softplus(weight) != weight). Callers who need function-preserving
//     wrapping must pre-invert the weight themselves before wrapping.
//   - Parametrizations must preserve shape. PyTorch allows chains that change
//     shape between links (only the final shape must fit the module); here
//     every link must map (shape) -> (same shape), validated on each Forward.
//   - No parametrize.cached() context manager: the chain is recomputed on
//     every Forward (PyTorch's default behavior outside cached()).
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
}

// NewParametrizedLinear wraps l with weightP applied to an owned copy of its
// current weight. Chain further weight parametrizations with
// AddParametrization. At most one optional bias parametrization may be given
// as a trailing argument; it turns the bias into an owned bias_orig as well.
func NewParametrizedLinear(l *Linear, weightP Parametrization, biasP ...Parametrization) *ParametrizedLinear {
	if weightP == nil {
		panic("nn: NewParametrizedLinear requires a non-nil weight parametrization")
	}
	if len(biasP) > 1 {
		panic("nn: NewParametrizedLinear accepts at most one bias parametrization")
	}
	m := &ParametrizedLinear{InFeatures: l.InFeatures, OutFeatures: l.OutFeatures}
	m.WeightOrig = m.reg("weight_orig", l.Weight.Copy().SetRequiresGrad(true))
	m.weightPs = []Parametrization{weightP}
	if len(biasP) == 1 && biasP[0] != nil {
		if l.Bias == nil {
			panic("nn: bias parametrization given but the wrapped Linear has no bias")
		}
		m.BiasOrig = m.reg("bias_orig", l.Bias.Copy().SetRequiresGrad(true))
		m.biasPs = []Parametrization{biasP[0]}
	} else if l.Bias != nil {
		m.Bias = m.reg("bias", l.Bias)
	}
	return m
}

func (m *ParametrizedLinear) isParametrizedModule() {}

// AddParametrization appends p to the WEIGHT parametrization chain. Chains
// apply in registration order (first registered runs first), mirroring
// PyTorch's ParametrizationList. Returns m for call chaining.
func (m *ParametrizedLinear) AddParametrization(p Parametrization) *ParametrizedLinear {
	if p == nil {
		panic("nn: AddParametrization: nil parametrization")
	}
	m.weightPs = append(m.weightPs, p)
	return m
}

// EffectiveWeight recomputes the parametrized weight with differentiable ops.
func (m *ParametrizedLinear) EffectiveWeight() *tensor.Tensor {
	return applyParametrizationChain(m.weightPs, m.WeightOrig)
}

// EffectiveBias returns the parametrized bias (chain over bias_orig) when the
// bias is parametrized, the shared raw bias otherwise, or nil for no bias.
func (m *ParametrizedLinear) EffectiveBias() *tensor.Tensor {
	if m.BiasOrig != nil {
		return applyParametrizationChain(m.biasPs, m.BiasOrig)
	}
	return m.Bias
}

// Forward computes x @ W^T + b with the recomputed effective weight/bias.
func (m *ParametrizedLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
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
// Linear: the wrapper owns weight_orig (a copy of the conv's current weight,
// (OutC, InC, KH, KW)), recomputes W = chain(weight_orig) each Forward, and
// runs the wrapped conv's im2col + GEMM math (its unfold machinery and
// gather cache) with that weight. The wrapped conv's own registered weight
// is never touched.
//
// Limitation (deliberate, documented): grouped convolutions are not
// supported — the shared explicit-weight forward path handles groups == 1
// only, so the constructor panics on Groups > 1.
type ParametrizedConv2d struct {
	Base
	inner      *Conv2d
	WeightOrig *tensor.Tensor // owned raw weight ("weight_orig"), (OutC, InC, KH, KW)
	BiasOrig   *tensor.Tensor // owned raw bias ("bias_orig") when the bias is parametrized, else nil
	Bias       *tensor.Tensor // shared with the wrapped layer when the bias is NOT parametrized, else nil
	weightPs   []Parametrization
	biasPs     []Parametrization
}

// NewParametrizedConv2d wraps c with weightP applied to an owned copy of its
// current weight. Chain further weight parametrizations with
// AddParametrization. At most one optional bias parametrization may be given
// as a trailing argument. Panics if c uses groups > 1.
func NewParametrizedConv2d(c *Conv2d, weightP Parametrization, biasP ...Parametrization) *ParametrizedConv2d {
	if weightP == nil {
		panic("nn: NewParametrizedConv2d requires a non-nil weight parametrization")
	}
	if len(biasP) > 1 {
		panic("nn: NewParametrizedConv2d accepts at most one bias parametrization")
	}
	if c.Groups != 1 {
		panic("nn: ParametrizedConv2d supports groups == 1 only")
	}
	m := &ParametrizedConv2d{inner: c}
	m.WeightOrig = m.reg("weight_orig", c.Weight.Copy().SetRequiresGrad(true))
	m.weightPs = []Parametrization{weightP}
	if len(biasP) == 1 && biasP[0] != nil {
		if c.Bias == nil {
			panic("nn: bias parametrization given but the wrapped Conv2d has no bias")
		}
		m.BiasOrig = m.reg("bias_orig", c.Bias.Copy().SetRequiresGrad(true))
		m.biasPs = []Parametrization{biasP[0]}
	} else if c.Bias != nil {
		m.Bias = m.reg("bias", c.Bias)
	}
	return m
}

func (m *ParametrizedConv2d) isParametrizedModule() {}

// AddParametrization appends p to the WEIGHT parametrization chain (applied
// in registration order). Returns m for call chaining.
func (m *ParametrizedConv2d) AddParametrization(p Parametrization) *ParametrizedConv2d {
	if p == nil {
		panic("nn: AddParametrization: nil parametrization")
	}
	m.weightPs = append(m.weightPs, p)
	return m
}

// EffectiveWeight recomputes the parametrized weight with differentiable ops.
func (m *ParametrizedConv2d) EffectiveWeight() *tensor.Tensor {
	return applyParametrizationChain(m.weightPs, m.WeightOrig)
}

// EffectiveBias returns the parametrized bias when the bias is parametrized,
// the shared raw bias otherwise, or nil for no bias.
func (m *ParametrizedConv2d) EffectiveBias() *tensor.Tensor {
	if m.BiasOrig != nil {
		return applyParametrizationChain(m.biasPs, m.BiasOrig)
	}
	return m.Bias
}

// Forward runs the wrapped conv's im2col + GEMM math (gather cache and
// unfold from the inner conv) with the recomputed effective weight.
func (m *ParametrizedConv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return convForwardWithWeight(&m.inner.convNd, x, m.EffectiveWeight(), m.EffectiveBias())
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
