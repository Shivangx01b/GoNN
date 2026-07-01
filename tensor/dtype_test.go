package tensor

import (
	"math"
	"testing"
)

// TestFloat16KnownValues checks the IEEE-754 binary16 rounding against values
// with well-known half-precision encodings.
func TestFloat16KnownValues(t *testing.T) {
	cases := []struct {
		in   float64
		want float64 // exact float16 value, as float64
	}{
		{0, 0},
		{1, 1},
		{-2, -2},
		{0.5, 0.5},
		{65504, 65504},                       // max normal half
		{0.1, 0.0999755859375},               // nearest half to 0.1
		{math.Pow(2, -24), math.Pow(2, -24)}, // smallest positive subnormal
	}
	for _, c := range cases {
		got := roundTo(c.in, Float16)
		if got != c.want {
			t.Fatalf("float16(%v) = %v, want %v", c.in, got, c.want)
		}
	}
	// Overflow above max-normal rounds to +Inf.
	if got := roundTo(70000, Float16); !math.IsInf(got, 1) {
		t.Fatalf("float16(70000) = %v, want +Inf", got)
	}
	if got := roundTo(-70000, Float16); !math.IsInf(got, -1) {
		t.Fatalf("float16(-70000) = %v, want -Inf", got)
	}
	// Half of the smallest subnormal rounds to 0 (ties to even).
	if got := roundTo(math.Pow(2, -25), Float16); got != 0 {
		t.Fatalf("float16(2^-25) = %v, want 0", got)
	}
	// NaN stays NaN.
	if got := roundTo(math.NaN(), Float16); !math.IsNaN(got) {
		t.Fatalf("float16(NaN) not NaN: %v", got)
	}
}

// TestFloat32Rounding checks fp32 precision loss is exactly Go's float32.
func TestFloat32Rounding(t *testing.T) {
	for _, v := range []float64{math.Pi, 0.1, 1.0 / 3.0, 1e30, 123456789.123} {
		if got, want := roundTo(v, Float32), float64(float32(v)); got != want {
			t.Fatalf("float32(%v) = %v, want %v", v, got, want)
		}
	}
	// fp32 cannot represent this f64-distinct value; round-trip must lose it.
	x := 1.0 + math.Pow(2, -40)
	if roundTo(x, Float32) != 1.0 {
		t.Fatalf("expected fp32(1+2^-40) == 1")
	}
}

// TestDTypePropagation checks ops carry and round to the right dtype.
func TestDTypePropagation(t *testing.T) {
	a := NewTyped([]float64{0.1, 0.2, 0.3}, Float16, 3)
	if a.DType() != Float16 {
		t.Fatalf("DType = %v, want float16", a.DType())
	}
	// every element must be a representable half
	for i, v := range a.Data {
		if v != roundTo(v, Float16) {
			t.Fatalf("a[%d]=%v not a float16 value", i, v)
		}
	}
	// f16 + f16 -> f16 (result rounded to half)
	b := NewTyped([]float64{0.1, 0.1, 0.1}, Float16, 3)
	c := a.Add(b)
	if c.DType() != Float16 {
		t.Fatalf("f16+f16 dtype = %v, want float16", c.DType())
	}
	for _, v := range c.Data {
		if v != roundTo(v, Float16) {
			t.Fatalf("sum element %v not a float16 value", v)
		}
	}
	// promotion: f16 + f64 -> f64
	d := New([]float64{1, 1, 1}, 3) // float64
	if got := a.Add(d).DType(); got != Float64 {
		t.Fatalf("f16+f64 dtype = %v, want float64", got)
	}
	// f32 + f16 -> f32
	e := NewTyped([]float64{1, 1, 1}, Float32, 3)
	if got := e.Add(a).DType(); got != Float32 {
		t.Fatalf("f32+f16 dtype = %v, want float32", got)
	}
}

// TestAsTypeAutograd checks casts are differentiable (straight-through).
func TestAsTypeAutograd(t *testing.T) {
	x := New([]float64{1, 2, 3}, 3).SetRequiresGrad(true)
	// loss = sum( (x.to(f32))^2 ); d/dx = 2x (cast grad passes through)
	x.To(Float32).Square().Sum().Backward()
	approxEq(t, x.Grad.Data, []float64{2, 4, 6}, 1e-9)
}

// TestFloat64DefaultUnchanged ensures default (float64) tensors are bit-identical
// to before (no rounding applied).
func TestFloat64DefaultUnchanged(t *testing.T) {
	a := New([]float64{0.1, 0.2, 0.3}, 3)
	if a.DType() != Float64 {
		t.Fatalf("default dtype = %v, want float64", a.DType())
	}
	c := a.Add(a)
	want := []float64{0.1 + 0.1, 0.2 + 0.2, 0.3 + 0.3} // exact float64, no rounding
	approxEq(t, c.Data, want, 0)
}
