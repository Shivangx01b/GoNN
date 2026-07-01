package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

func TestGELUExactModuleForward(t *testing.T) {
	x := tensor.New([]float64{-2, -0.5, 0, 0.5, 2}, 5)
	got := GELUExact().Forward(x)
	want := x.GELUExact()
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("GELUExact module [%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
	// Spot value against the erf formula.
	ref := 0.5 * 2 * (1 + math.Erf(2/math.Sqrt2))
	if math.Abs(got.Data[4]-ref) > 1e-15 {
		t.Fatalf("GELUExact(2) = %v, want %v", got.Data[4], ref)
	}
}

func TestGELUApprox(t *testing.T) {
	x := tensor.New([]float64{-1.5, -0.2, 0.4, 1.1}, 4)
	none := GELUApprox("none").Forward(x)
	wantNone := x.GELUExact()
	tanh := GELUApprox("tanh").Forward(x)
	wantTanh := x.GELU()
	for i := range none.Data {
		if none.Data[i] != wantNone.Data[i] {
			t.Fatalf("GELUApprox(\"none\") [%d] = %v, want %v", i, none.Data[i], wantNone.Data[i])
		}
		if tanh.Data[i] != wantTanh.Data[i] {
			t.Fatalf("GELUApprox(\"tanh\") [%d] = %v, want %v", i, tanh.Data[i], wantTanh.Data[i])
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for unknown approximate value")
		}
	}()
	GELUApprox("bogus")
}

func TestSoftplusWithModuleForward(t *testing.T) {
	x := tensor.New([]float64{-3, -0.5, 0, 0.7, 15}, 5)
	got := SoftplusWith(2, 20).Forward(x)
	want := x.SoftplusBeta(2, 20)
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("SoftplusWith module [%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
	// Linear region: beta*x = 30 > 20 -> identity.
	if got.Data[4] != 15 {
		t.Fatalf("SoftplusWith linear region = %v, want exactly 15", got.Data[4])
	}
	// Formula spot check below threshold.
	ref := math.Log(1+math.Exp(2*0.7)) / 2
	if math.Abs(got.Data[3]-ref) > 1e-12 {
		t.Fatalf("SoftplusWith(2,20)(0.7) = %v, want %v", got.Data[3], ref)
	}
}

func TestRReLUModuleEvalMidpoint(t *testing.T) {
	r := RReLU(0.1, 0.3)
	r.Eval()
	x := tensor.New([]float64{-2, -1, 0, 1, 2}, 5)
	got := r.Forward(x)
	// Midpoint slope 0.2, matching tensor.RReLU.
	want := []float64{-0.4, -0.2, 0, 1, 2}
	for i := range got.Data {
		if math.Abs(got.Data[i]-want[i]) > 1e-12 {
			t.Fatalf("eval RReLU [%d] = %v, want %v", i, got.Data[i], want[i])
		}
	}
}

func TestRReLUModuleTrainingSlope(t *testing.T) {
	const lower, upper = 0.1, 0.3
	r := RReLU(lower, upper)
	r.Seed(42)
	if !r.Training() {
		t.Fatal("new module should default to training mode")
	}
	x := tensor.New([]float64{-1, 1}, 2)
	const trials = 500
	sum := 0.0
	seen := map[float64]bool{}
	for i := 0; i < trials; i++ {
		y := r.Forward(x)
		slope := -y.Data[0] // x=-1 -> y = -slope
		if slope < lower || slope > upper {
			t.Fatalf("trial %d: slope %v outside [%v, %v]", i, slope, lower, upper)
		}
		if y.Data[1] != 1 {
			t.Fatalf("trial %d: positive input changed: %v", i, y.Data[1])
		}
		sum += slope
		seen[slope] = true
	}
	// Statistical: mean of U(0.1, 0.3) is 0.2; stddev of the mean over 500
	// draws is ~0.0026, so +/-0.02 is ~8 sigma.
	mean := sum / trials
	if math.Abs(mean-0.2) > 0.02 {
		t.Fatalf("mean slope %v, want ~0.2 (fixed seed 42)", mean)
	}
	if len(seen) < trials/2 {
		t.Fatalf("slopes not varying: only %d distinct values in %d trials", len(seen), trials)
	}
}

func TestRReLUInvalidRangePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for lower > upper")
		}
	}()
	RReLU(0.5, 0.1)
}
