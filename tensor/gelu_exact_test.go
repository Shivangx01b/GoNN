package tensor

import (
	"math"
	"testing"
)

// geluExactRef is the independent reference for the exact-erf GELU.
func geluExactRef(x float64) float64 {
	return 0.5 * x * (1 + math.Erf(x/math.Sqrt2))
}

func TestGELUExactRegistered(t *testing.T) {
	d, ok := LookupUnary("geluexact")
	if !ok {
		t.Fatalf("\"geluexact\" not in registry; have %v", UnaryOpNames())
	}
	if d.Kind != UnaryGELUExact {
		t.Fatalf("geluexact Kind = %v, want UnaryGELUExact (%v)", d.Kind, UnaryGELUExact)
	}
	// Unary(name) matches the fluent method exactly.
	x := New([]float64{-2, -0.5, 0, 0.5, 2}, 5)
	byName := x.Unary("geluexact")
	fluent := x.GELUExact()
	for i := range byName.Data {
		if byName.Data[i] != fluent.Data[i] {
			t.Fatalf("Unary(\"geluexact\")[%d]=%v != fluent %v", i, byName.Data[i], fluent.Data[i])
		}
	}
}

func TestGELUExactForward(t *testing.T) {
	xs := []float64{0, 0.5, -0.5, 2, -2, 5, -5}
	x := New(xs, len(xs))
	y := x.GELUExact()
	want := make([]float64, len(xs))
	for i, v := range xs {
		want[i] = geluExactRef(v)
	}
	approxEq(t, y.Data, want, 1e-15)
	// Spot values: GELU(0) = 0 and far-tail behavior GELU(5) ~= 5, GELU(-5) ~= 0.
	if y.Data[0] != 0 {
		t.Fatalf("GELUExact(0) = %v, want 0", y.Data[0])
	}
	if math.Abs(y.Data[5]-5) > 1e-5 {
		t.Fatalf("GELUExact(5) = %v, want ~5", y.Data[5])
	}
	if math.Abs(y.Data[6]) > 1e-5 {
		t.Fatalf("GELUExact(-5) = %v, want ~0", y.Data[6])
	}
}

func TestGELUExactGradCheck(t *testing.T) {
	x := New([]float64{-2.5, -1, -0.3, 0.2, 0.9, 1.7, 3}, 7)
	w := fixedWeights(7)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.GELUExact().Mul(w).Sum() })
}

// TestGELUTanhApproxVsExact sanity-checks that the tanh approximation (the
// registered "gelu") stays within 1e-3 of the exact-erf GELU for |x| < 3.
func TestGELUTanhApproxVsExact(t *testing.T) {
	for v := -2.99; v < 3; v += 0.01 {
		x := New([]float64{v}, 1)
		exact := x.GELUExact().Data[0]
		approx := x.GELU().Data[0]
		if d := math.Abs(exact - approx); d > 1e-3 {
			t.Fatalf("tanh approx vs exact at x=%v: |diff|=%g > 1e-3", v, d)
		}
	}
}

func TestSoftplusBetaForward(t *testing.T) {
	beta, threshold := 2.0, 20.0
	xs := []float64{-3, -0.5, 0, 0.7, 4}
	x := New(xs, len(xs))
	y := x.SoftplusBeta(beta, threshold)
	want := make([]float64, len(xs))
	for i, v := range xs {
		want[i] = math.Log(1+math.Exp(beta*v)) / beta
	}
	approxEq(t, y.Data, want, 1e-12)
}

// TestSoftplusBetaLinearRegion checks the exact identity (and pass-through
// gradient) where beta*x > threshold.
func TestSoftplusBetaLinearRegion(t *testing.T) {
	x := New([]float64{15, 30, 100}, 3).SetRequiresGrad(true)
	y := x.SoftplusBeta(2, 20) // beta*x in {30, 60, 200}, all > 20
	for i, v := range x.Data {
		if y.Data[i] != v {
			t.Fatalf("linear region: SoftplusBeta[%d] = %v, want exactly %v", i, y.Data[i], v)
		}
	}
	y.Sum().Backward()
	approxEq(t, x.Grad.Data, []float64{1, 1, 1}, 0)
}

// TestSoftplusBetaMatchesDefaultSoftplus: beta=1, threshold=20 is exactly the
// pinned zero-arg Softplus.
func TestSoftplusBetaMatchesDefaultSoftplus(t *testing.T) {
	x := New([]float64{-5, -1, 0, 1, 5, 25}, 6)
	a := x.SoftplusBeta(1, 20)
	b := x.Softplus()
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("SoftplusBeta(1,20)[%d]=%v != Softplus() %v", i, a.Data[i], b.Data[i])
		}
	}
}

func TestSoftplusBetaGradCheck(t *testing.T) {
	// Points kept away from the beta*x == threshold seam (the linear switch is
	// only approximately continuous, which would break central differences).
	x := New([]float64{-2, -0.6, 0.1, 0.8, 2.5}, 5)
	w := fixedWeights(5)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.SoftplusBeta(2, 20).Mul(w).Sum() })
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.SoftplusBeta(0.5, 20).Mul(w).Sum() })
}

func TestSoftplusBetaZeroBetaPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for beta == 0")
		}
	}()
	Zeros(3).SoftplusBeta(0, 20)
}
