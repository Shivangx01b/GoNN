package nn

import (
	"math"
	"math/rand"
	"testing"

	"gonn/tensor"
)

// positiveParam is softplus with its exact right inverse log(exp(v)-1):
// softplus(log(exp(v)-1)) = log(1 + exp(log(exp(v)-1))) = log(exp(v)) = v.
// Valid for strictly positive v.
func positiveParam() InvertibleParametrization {
	return InvertibleParametrization{
		Fwd: (*tensor.Tensor).Softplus,
		Inv: func(v *tensor.Tensor) *tensor.Tensor { return v.Exp().SubScalar(1).Log() },
	}
}

// scaleParam multiplies by c (inverse: divide by c).
func scaleParam(c float64) InvertibleParametrization {
	return InvertibleParametrization{
		Fwd: func(w *tensor.Tensor) *tensor.Tensor { return w.MulScalar(c) },
		Inv: func(v *tensor.Tensor) *tensor.Tensor { return v.MulScalar(1 / c) },
	}
}

// positiveTensor fills shape with values in (0.5, 1.5) so log(exp(v)-1) is safe.
func positiveTensor(rng *rand.Rand, shape ...int) *tensor.Tensor {
	n := 1
	for _, s := range shape {
		n *= s
	}
	d := make([]float64, n)
	for i := range d {
		d[i] = 0.5 + rng.Float64()
	}
	return tensor.New(d, shape...)
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// TestRightInverseFunctionPreservation: wrapping with an inverse-equipped
// parametrization preserves the layer's function (PyTorch right_inverse
// initialization) — but only when the current weight is in the
// parametrization's image (positive, for softplus).
func TestRightInverseFunctionPreservation(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	l := NewLinear(4, 3, true)
	copy(l.Weight.Data, positiveTensor(rng, 3, 4).Data) // softplus image

	x := tensor.Randn(5, 4)
	want := l.Forward(x)

	m := NewParametrizedLinear(l, positiveParam())
	got := m.Forward(x)
	if d := maxAbsDiff(got.Data, want.Data); d > 1e-9 {
		t.Fatalf("right_inverse wrap changed the function: max diff %g", d)
	}
	// And the effective weight reproduces the original weight.
	if d := maxAbsDiff(m.EffectiveWeight().Data, l.Weight.Data); d > 1e-9 {
		t.Fatalf("EffectiveWeight != original weight: %g", d)
	}
}

// TestSetEffectiveWeight: assignment through the chain's inverses.
func TestSetEffectiveWeight(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	l := NewLinear(3, 3, false)
	m := NewParametrizedLinear(l, positiveParam())

	target := positiveTensor(rng, 3, 3)
	m.SetEffectiveWeight(target)
	if d := maxAbsDiff(m.EffectiveWeight().Data, target.Data); d > 1e-12 {
		t.Fatalf("SetEffectiveWeight round trip: max diff %g", d)
	}
}

// TestSetEffectiveWeightNonInvertiblePanics: a chain link without
// RightInverse must reject assignment.
func TestSetEffectiveWeightNonInvertiblePanics(t *testing.T) {
	l := NewLinear(2, 2, false)
	m := NewParametrizedLinear(l, ParametrizationFunc((*tensor.Tensor).Softplus))
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for non-invertible chain")
		}
	}()
	m.SetEffectiveWeight(tensor.Ones(2, 2))
}

// TestAddParametrizationPreservesFunction: appending to a fully invertible
// chain re-derives weight_orig so the effective weight is unchanged.
func TestAddParametrizationPreservesFunction(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	l := NewLinear(4, 2, true)
	copy(l.Weight.Data, positiveTensor(rng, 2, 4).Data)

	m := NewParametrizedLinear(l, positiveParam())
	x := tensor.Randn(3, 4)
	before := m.Forward(x)

	m.AddParametrization(scaleParam(2.0))
	after := m.Forward(x)
	if d := maxAbsDiff(after.Data, before.Data); d > 1e-9 {
		t.Fatalf("AddParametrization changed the function: max diff %g", d)
	}

	// A non-invertible append leaves weight_orig alone (function changes).
	m2 := NewParametrizedLinear(NewLinear(2, 2, false), scaleParam(3.0))
	origData := append([]float64(nil), m2.WeightOrig.Data...)
	m2.AddParametrization(ParametrizationFunc((*tensor.Tensor).Softplus))
	if d := maxAbsDiff(m2.WeightOrig.Data, origData); d != 0 {
		t.Fatalf("non-invertible append should not touch weight_orig (diff %g)", d)
	}
}

// countingParam wraps a Parametrization and counts Apply calls.
type countingParam struct {
	inner Parametrization
	n     int
}

func (c *countingParam) Apply(orig *tensor.Tensor) *tensor.Tensor {
	c.n++
	return c.inner.Apply(orig)
}

// TestCachedWindow: inside Cached the chain runs once; gradients through the
// cached weight accumulate exactly like two uncached passes; the cache is
// invalidated on exit.
func TestCachedWindow(t *testing.T) {
	build := func() (*ParametrizedLinear, *countingParam) {
		rand.Seed(42)
		l := NewLinear(3, 2, false)
		cp := &countingParam{inner: ParametrizationFunc((*tensor.Tensor).Softplus)}
		return NewParametrizedLinear(l, cp), cp
	}
	x1 := tensor.Randn(4, 3)
	x2 := tensor.Randn(4, 3)

	// Uncached reference: two forward+backward passes.
	ref, _ := build()
	ref.Forward(x1).Sum().Backward()
	ref.Forward(x2).Sum().Backward()
	wantGrad := append([]float64(nil), ref.WeightOrig.Grad.Data...)

	m, cp := build()
	m.Cached(func() {
		if cp.n != 1 {
			t.Fatalf("cache setup should apply the chain once, got %d", cp.n)
		}
		m.Forward(x1).Sum().Backward()
		m.Forward(x2).Sum().Backward()
		if cp.n != 1 {
			t.Fatalf("forwards inside Cached must not recompute (applies=%d)", cp.n)
		}
	})
	if d := maxAbsDiff(m.WeightOrig.Grad.Data, wantGrad); d > 1e-12 {
		t.Fatalf("cached gradients differ from uncached: max diff %g", d)
	}
	// Cache invalidated: next Forward recomputes.
	m.Forward(x1)
	if cp.n != 2 {
		t.Fatalf("cache not invalidated after window (applies=%d)", cp.n)
	}
}

// TestParametrizeCachedAll nests windows over several modules.
func TestParametrizeCachedAll(t *testing.T) {
	rand.Seed(43)
	c1 := &countingParam{inner: ParametrizationFunc((*tensor.Tensor).Softplus)}
	c2 := &countingParam{inner: ParametrizationFunc((*tensor.Tensor).Tanh)}
	m1 := NewParametrizedLinear(NewLinear(3, 3, false), c1)
	m2 := NewParametrizedLinear(NewLinear(3, 2, false), c2)
	x := tensor.Randn(2, 3)
	ParametrizeCachedAll(func() {
		for i := 0; i < 3; i++ {
			m2.Forward(m1.Forward(x))
		}
	}, m1, m2)
	if c1.n != 1 || c2.n != 1 {
		t.Fatalf("expected one apply per module, got %d and %d", c1.n, c2.n)
	}
}

// TestParametrizedConv2dGrouped: grouped convs now work — identity
// parametrization reproduces the inner conv exactly, and a real
// parametrization equals a plain grouped conv with the transformed weight.
func TestParametrizedConv2dGrouped(t *testing.T) {
	rand.Seed(44)
	c := NewConv2d(4, 6, 3, WithGroups(2), WithPad(1))
	x := tensor.Randn(2, 4, 5, 5)

	ident := ParametrizationFunc(func(w *tensor.Tensor) *tensor.Tensor { return w.MulScalar(1) })
	m := NewParametrizedConv2d(c, ident)
	if d := maxAbsDiff(m.Forward(x).Data, c.Forward(x).Data); d != 0 {
		t.Fatalf("identity-parametrized grouped conv != inner conv (diff %g)", d)
	}

	// Softplus parametrization vs a manually transformed plain conv.
	mp := NewParametrizedConv2d(c, ParametrizationFunc((*tensor.Tensor).Softplus))
	ref := NewConv2d(4, 6, 3, WithGroups(2), WithPad(1))
	sp := c.Weight.Softplus()
	copy(ref.Weight.Data, sp.Data)
	copy(ref.Bias.Data, c.Bias.Data)
	if d := maxAbsDiff(mp.Forward(x).Data, ref.Forward(x).Data); d > 1e-12 {
		t.Fatalf("grouped parametrized conv mismatch: %g", d)
	}

	// Gradcheck through weight_orig on a small grouped conv.
	small := NewConv2d(2, 2, 2, WithGroups(2))
	msm := NewParametrizedConv2d(small, ParametrizationFunc((*tensor.Tensor).Softplus))
	xs := tensor.Randn(1, 2, 3, 3)
	gradCheck(t, "ParametrizedConv2dGrouped", func() *tensor.Tensor {
		return msm.Forward(xs).Pow(2).Sum()
	}, []*tensor.Tensor{msm.WeightOrig}, 1e-5, 1e-4, 0)

	// RemoveParametrizations keeps groups (newConv2dLike carries them now).
	baked := RemoveParametrizationsConv2d(mp, true)
	if baked.Groups != 2 {
		t.Fatalf("RemoveParametrizations dropped groups: got %d", baked.Groups)
	}
	if d := maxAbsDiff(baked.Forward(x).Data, mp.Forward(x).Data); d > 1e-12 {
		t.Fatalf("baked grouped conv differs: %g", d)
	}
}
