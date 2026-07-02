package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// softplusParam is the guaranteed-positive-weights parametrization used
// throughout these tests.
var softplusParam = ParametrizationFunc(func(w *tensor.Tensor) *tensor.Tensor { return w.Softplus() })

// spRef mirrors tensor's Softplus forward exactly (including the large-input
// branch) so manual expectations are bit-identical.
func spRef(v float64) float64 {
	if v > 20 {
		return v
	}
	return math.Log(1 + math.Exp(v))
}

// ---- ParametrizedLinear -------------------------------------------------------

func TestParametrizedLinearForwardMatchesManual(t *testing.T) {
	l := NewLinear(5, 3, true)
	pl := NewParametrizedLinear(l, softplusParam)
	x := seededRandn(301, 4, 5)
	got := pl.Forward(x)

	// Manual: plain Linear whose weight is softplus(weight_orig).
	manual := NewLinear(5, 3, true)
	for i, w := range l.Weight.Data {
		manual.Weight.Data[i] = spRef(w)
	}
	copy(manual.Bias.Data, l.Bias.Data)
	want := manual.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d != 0 {
		t.Errorf("parametrized forward != manual softplus-weight forward: max diff %g", d)
	}

	// The whole point of the softplus parametrization: weights stay positive.
	for i, w := range pl.EffectiveWeight().Data {
		if w <= 0 {
			t.Errorf("effective weight[%d] = %g, want > 0", i, w)
		}
	}

	// The wrapped layer's own weight must be untouched and unregistered here.
	if d := maxAbsDiff64(t, l.Weight.Data, pl.WeightOrig.Data); d != 0 {
		t.Errorf("weight_orig was not initialized from the wrapped layer's weight: max diff %g", d)
	}
	for _, p := range pl.Parameters() {
		if p == l.Weight {
			t.Errorf("wrapped layer's weight leaked into the wrapper's parameters")
		}
	}
}

func TestGradCheckParametrizedLinear(t *testing.T) {
	pl := NewParametrizedLinear(NewLinear(4, 3, true), softplusParam)
	x := seededRandn(302, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return pl.Forward(x).Square().Mean() }
	gradCheck(t, "ParametrizedLinear", loss, append(pl.Parameters(), x), gcEps, gcTol, 0)
}

func TestParametrizedLinearChainOrder(t *testing.T) {
	l := NewLinear(4, 2, false)
	pl := NewParametrizedLinear(l, softplusParam).
		AddParametrization(ParametrizationFunc(func(w *tensor.Tensor) *tensor.Tensor {
			return w.AddScalar(1)
		}))

	// Registration order: softplus first, then +1 -> softplus(w)+1.
	eff := pl.EffectiveWeight()
	sameAsReversed := true
	for i, w := range l.Weight.Data {
		want := spRef(w) + 1
		if math.Abs(eff.Data[i]-want) > 1e-12 {
			t.Errorf("chained weight[%d] = %g, want softplus(w)+1 = %g", i, eff.Data[i], want)
		}
		if math.Abs(eff.Data[i]-spRef(w+1)) > 1e-6 {
			sameAsReversed = false
		}
	}
	// Guard the test itself: the two orders must actually differ on this data.
	if sameAsReversed {
		t.Fatalf("test is vacuous: softplus(w)+1 == softplus(w+1) for all weights")
	}

	// The chained wrapper must still be fully differentiable.
	x := seededRandn(303, 3, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return pl.Forward(x).Square().Mean() }
	gradCheck(t, "ParametrizedLinearChain", loss, append(pl.Parameters(), x), gcEps, gcTol, 0)
}

func TestParametrizedLinearBiasParametrization(t *testing.T) {
	l := NewLinear(4, 3, true)
	pl := NewParametrizedLinear(l, softplusParam, softplusParam)
	if pl.BiasOrig == nil || pl.Bias != nil {
		t.Fatalf("bias parametrization did not switch the wrapper to an owned bias_orig")
	}

	x := seededRandn(304, 2, 4)
	got := pl.Forward(x)

	manual := NewLinear(4, 3, true)
	for i, w := range l.Weight.Data {
		manual.Weight.Data[i] = spRef(w)
	}
	for i, b := range l.Bias.Data {
		manual.Bias.Data[i] = spRef(b)
	}
	want := manual.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d != 0 {
		t.Errorf("bias-parametrized forward != manual forward: max diff %g", d)
	}

	xg := seededRandn(305, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return pl.Forward(xg).Square().Mean() }
	gradCheck(t, "ParametrizedLinearBias", loss, append(pl.Parameters(), xg), gcEps, gcTol, 0)
}

func TestRemoveParametrizationsLinear(t *testing.T) {
	l := NewLinear(5, 3, true)
	pl := NewParametrizedLinear(l, softplusParam)
	// Move weight_orig off its init so baked vs raw genuinely differ.
	for i := range pl.WeightOrig.Data {
		pl.WeightOrig.Data[i] += 0.05 * float64(i%7)
	}
	x := seededRandn(306, 4, 5)
	want := pl.Forward(x)

	// leave_parametrized=true: the plain layer computes the same function.
	baked := RemoveParametrizationsLinear(pl, true)
	if d := maxAbsDiff64(t, baked.Forward(x).Data, want.Data); d != 0 {
		t.Errorf("baked linear differs from parametrized forward: max diff %g", d)
	}
	if d := maxAbsDiff64(t, baked.Weight.Data, pl.EffectiveWeight().Data); d != 0 {
		t.Errorf("baked weight != effective weight: max diff %g", d)
	}

	// leave_parametrized=false: the plain layer gets the raw weight_orig back.
	raw := RemoveParametrizationsLinear(pl, false)
	if d := maxAbsDiff64(t, raw.Weight.Data, pl.WeightOrig.Data); d != 0 {
		t.Errorf("raw removal did not restore weight_orig: max diff %g", d)
	}
	if d := maxAbsDiff64(t, raw.Bias.Data, l.Bias.Data); d != 0 {
		t.Errorf("raw removal changed the bias: max diff %g", d)
	}
}

// ---- ParametrizedConv2d -------------------------------------------------------

func TestParametrizedConv2dForwardMatchesManual(t *testing.T) {
	c := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
	pc := NewParametrizedConv2d(c, softplusParam)
	x := seededRandn(307, 2, 2, 6, 7)
	got := pc.Forward(x)

	// Manual: a plain conv with identical geometry whose weight Data is
	// overwritten with the transformed values; outputs must match EXACTLY
	// (identical im2col + GEMM code path on identical numbers).
	manual := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
	for i, w := range c.Weight.Data {
		manual.Weight.Data[i] = spRef(w)
	}
	copy(manual.Bias.Data, c.Bias.Data)
	want := manual.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d != 0 {
		t.Errorf("parametrized conv forward != manual transformed-weight conv: max diff %g", d)
	}
}

func TestGradCheckParametrizedConv2d(t *testing.T) {
	pc := NewParametrizedConv2d(NewConv2d(2, 3, 3, WithPad(1)), softplusParam)
	x := seededRandn(308, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return pc.Forward(x).Square().Mean() }
	gradCheck(t, "ParametrizedConv2d", loss, append(pc.Parameters(), x), gcEps, gcTol, 40)
}

func TestRemoveParametrizationsConv2d(t *testing.T) {
	c := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
	pc := NewParametrizedConv2d(c, softplusParam)
	x := seededRandn(309, 2, 2, 6, 6)
	want := pc.Forward(x)

	baked := RemoveParametrizationsConv2d(pc, true)
	if d := maxAbsDiff64(t, baked.Forward(x).Data, want.Data); d != 0 {
		t.Errorf("baked conv differs from parametrized forward: max diff %g", d)
	}

	raw := RemoveParametrizationsConv2d(pc, false)
	if d := maxAbsDiff64(t, raw.Weight.Data, pc.WeightOrig.Data); d != 0 {
		t.Errorf("raw removal did not restore weight_orig: max diff %g", d)
	}
}

// Grouped convolutions are supported since the right_inverse/cached upgrade;
// TestParametrizedConv2dGrouped (parametrize_finish_test.go) covers them.
func TestParametrizedConv2dGroupsConstructs(t *testing.T) {
	m := NewParametrizedConv2d(NewConv2d(4, 4, 3, WithGroups(2)), softplusParam)
	if m == nil || !IsParametrized(m) {
		t.Fatal("grouped ParametrizedConv2d should construct")
	}
}

// ---- generic helpers ----------------------------------------------------------

func TestRemoveParametrizationsDispatch(t *testing.T) {
	pl := NewParametrizedLinear(NewLinear(3, 2, true), softplusParam)
	if _, ok := RemoveParametrizations(pl, true).(*Linear); !ok {
		t.Errorf("RemoveParametrizations(ParametrizedLinear) did not return *Linear")
	}
	pc := NewParametrizedConv2d(NewConv2d(2, 2, 3), softplusParam)
	if _, ok := RemoveParametrizations(pc, false).(*Conv2d); !ok {
		t.Errorf("RemoveParametrizations(ParametrizedConv2d) did not return *Conv2d")
	}
	mustPanic(t, "non-parametrized module", func() {
		RemoveParametrizations(NewLinear(2, 2, true), true)
	})
}

func TestIsParametrized(t *testing.T) {
	pl := NewParametrizedLinear(NewLinear(3, 2, true), softplusParam)
	pc := NewParametrizedConv2d(NewConv2d(2, 2, 3), softplusParam)
	if !IsParametrized(pl) || !IsParametrized(pc) {
		t.Errorf("IsParametrized is false for a generic parametrized wrapper")
	}
	if IsParametrized(NewLinear(3, 2, true)) {
		t.Errorf("IsParametrized is true for a plain Linear")
	}
}

func TestParametrizationShapeChangePanics(t *testing.T) {
	pl := NewParametrizedLinear(NewLinear(3, 2, false),
		ParametrizationFunc(func(w *tensor.Tensor) *tensor.Tensor { return w.Reshape(1, 6) }))
	mustPanic(t, "shape-changing parametrization", func() { pl.EffectiveWeight() })
}
