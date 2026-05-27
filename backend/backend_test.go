package backend

import (
	"math"
	"testing"
)

// approxEq reports |a-b| <= tol. Used for floating-point equality.
func approxEq(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) == math.IsNaN(b)
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		return a == b
	}
	return math.Abs(a-b) <= tol
}

func sliceApproxEq(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if !approxEq(got[i], want[i], tol) {
			t.Fatalf("%s[%d] = %v, want %v (tol %v)", name, i, got[i], want[i], tol)
		}
	}
}

func TestCPUBackend_NewOps(t *testing.T) {
	be := cpuBackend{}

	if be.Name() != CPU {
		t.Fatalf("Name() = %q, want %q", be.Name(), CPU)
	}

	a := []float64{1, 2, 3, 4}
	b := []float64{4, 3, 2, 1}

	// Sub / Div / Scale -----------------------------------------------------
	sliceApproxEq(t, "Sub", be.Sub(a, b), []float64{-3, -1, 1, 3}, 1e-12)
	sliceApproxEq(t, "Div", be.Div(a, b), []float64{0.25, 2.0 / 3.0, 1.5, 4.0}, 1e-12)
	sliceApproxEq(t, "Scale", be.Scale(a, 0.5), []float64{0.5, 1, 1.5, 2}, 1e-12)

	// AxpyInto: out += alpha*x ---------------------------------------------
	out := []float64{10, 20, 30, 40}
	be.AxpyInto(out, []float64{1, 1, 1, 1}, 2.0)
	sliceApproxEq(t, "AxpyInto", out, []float64{12, 22, 32, 42}, 1e-12)

	// Reductions -----------------------------------------------------------
	if got, want := be.Sum(a), 10.0; !approxEq(got, want, 1e-12) {
		t.Fatalf("Sum = %v, want %v", got, want)
	}
	if got, want := be.Max([]float64{-1, 5, 3, 5, 2}), 5.0; !approxEq(got, want, 1e-12) {
		t.Fatalf("Max = %v, want %v", got, want)
	}
	// Max of empty is -Inf.
	if got := be.Max(nil); !math.IsInf(got, -1) {
		t.Fatalf("Max(nil) = %v, want -Inf", got)
	}

	// Activations ----------------------------------------------------------
	relu := be.ReLU([]float64{-2, -0.5, 0, 0.5, 2})
	sliceApproxEq(t, "ReLU", relu, []float64{0, 0, 0, 0.5, 2}, 1e-12)

	// Sigmoid: known values at 0 and symmetry around 0.5.
	sig := be.Sigmoid([]float64{0, 1, -1})
	want := []float64{0.5, 1.0 / (1 + math.Exp(-1)), 1.0 / (1 + math.Exp(1))}
	sliceApproxEq(t, "Sigmoid", sig, want, 1e-12)

	// Tanh
	th := be.Tanh([]float64{0, 1, -1})
	sliceApproxEq(t, "Tanh", th, []float64{0, math.Tanh(1), math.Tanh(-1)}, 1e-12)

	// Exp / Log round-trip on positive inputs.
	x := []float64{0.1, 1, 2, 5}
	roundTrip := be.Log(be.Exp(x))
	sliceApproxEq(t, "Log(Exp(x))", roundTrip, x, 1e-10)

	// GELU spot-checks. f(0)=0; for large positive x, GELU(x) ~ x;
	// for large negative x, GELU(x) ~ 0.
	gelu := be.GELU([]float64{-5, 0, 1, 5})
	if !approxEq(gelu[1], 0, 1e-12) {
		t.Fatalf("GELU(0) = %v, want 0", gelu[1])
	}
	if !approxEq(gelu[2], 0.8411919906082768, 1e-6) {
		t.Fatalf("GELU(1) = %v, want ~0.84119", gelu[2])
	}
	if !approxEq(gelu[0], 0, 1e-4) {
		t.Fatalf("GELU(-5) = %v, want ~0", gelu[0])
	}
	if !approxEq(gelu[3], 5, 1e-4) {
		t.Fatalf("GELU(5) = %v, want ~5", gelu[3])
	}

	// SiLU(x) = x * sigmoid(x); SiLU(0)=0, SiLU(1)=sigmoid(1).
	silu := be.SiLU([]float64{0, 1, -1})
	if !approxEq(silu[0], 0, 1e-12) {
		t.Fatalf("SiLU(0) = %v, want 0", silu[0])
	}
	if !approxEq(silu[1], 1.0/(1+math.Exp(-1)), 1e-12) {
		t.Fatalf("SiLU(1) = %v, want sigmoid(1)", silu[1])
	}
	if !approxEq(silu[2], -1.0/(1+math.Exp(1)), 1e-12) {
		t.Fatalf("SiLU(-1) = %v, want -sigmoid(-1)", silu[2])
	}
}

// TestBackend_InterfaceSatisfied is a compile-time check that cpuBackend
// implements the (now larger) Backend interface — and that Current()
// returns it by default.
func TestBackend_InterfaceSatisfied(t *testing.T) {
	var _ Backend = cpuBackend{}
	if Current().Name() != CPU {
		t.Fatalf("default backend = %q, want %q", Current().Name(), CPU)
	}
}
