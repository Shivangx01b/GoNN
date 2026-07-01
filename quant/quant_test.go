package quant

import (
	"math"
	"math/rand"
	"testing"

	"gonn/nn"
	"gonn/tensor"
)

// seededTensor returns a deterministic tensor with values drawn from rng by f.
func seededTensor(seed int64, f func(*rand.Rand) float64, shape ...int) *tensor.Tensor {
	rng := rand.New(rand.NewSource(seed))
	t := tensor.Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = f(rng)
	}
	return t
}

func randn(seed int64, shape ...int) *tensor.Tensor {
	return seededTensor(seed, (*rand.Rand).NormFloat64, shape...)
}

func maxAbs(xs []float64) float64 {
	m := 0.0
	for _, v := range xs {
		if a := math.Abs(v); a > m {
			m = a
		}
	}
	return m
}

func TestQuantizeDequantizeRoundTrip(t *testing.T) {
	x := seededTensor(1, func(r *rand.Rand) float64 { return -2 + 4*r.Float64() }, 16, 16)
	var obs MinMaxObserver
	obs.Observe(x)
	scale, zp := obs.ComputeQParams()

	q := Quantize(x, scale, zp)
	d := Dequantize(q)
	maxErr := 0.0
	for i := range x.Data {
		if e := math.Abs(d.Data[i] - x.Data[i]); e > maxErr {
			maxErr = e
		}
	}
	// Values inside the observed range round-trip within half a quantization
	// step.
	if maxErr > scale/2+1e-12 {
		t.Errorf("round-trip max error %g exceeds scale/2 = %g", maxErr, scale/2)
	}
}

func TestObserverQParamsSane(t *testing.T) {
	x := tensor.New([]float64{-1.5, 0.25, 3.0, 0.75}, 4)
	var obs MinMaxObserver
	obs.Observe(x)
	scale, zp := obs.ComputeQParams()

	if scale <= 0 {
		t.Fatalf("scale = %g, want > 0", scale)
	}
	if zp < QMin || zp > QMax {
		t.Fatalf("zero point %d outside int8 range", zp)
	}
	// Real zero must be exactly representable (affine nudge-to-zero).
	z := Dequantize(Quantize(tensor.New([]float64{0}, 1), scale, zp))
	if z.Data[0] != 0 {
		t.Errorf("real 0.0 does not quantize exactly: got %g", z.Data[0])
	}
	// The observed extremes reconstruct within half a step (no saturation).
	ext := Dequantize(Quantize(tensor.New([]float64{obs.Min, obs.Max}, 2), scale, zp))
	if math.Abs(ext.Data[0]-obs.Min) > scale/2+1e-12 || math.Abs(ext.Data[1]-obs.Max) > scale/2+1e-12 {
		t.Errorf("range extremes saturate: got %v for [%g, %g]", ext.Data, obs.Min, obs.Max)
	}

	// Degenerate (unobserved) observer must still return usable params.
	var empty MinMaxObserver
	s0, z0 := empty.ComputeQParams()
	if s0 != 1 || z0 != 0 {
		t.Errorf("empty observer qparams = (%g, %d), want (1, 0)", s0, z0)
	}
}

func TestDynamicLinearMatchesFloat(t *testing.T) {
	l := nn.NewLinear(64, 64, true)
	x := randn(2, 32, 64)
	want := l.Forward(x)

	d := NewDynamicLinearFrom(l)
	got := d.Forward(x)

	relErr := 0.0
	scaleRef := maxAbs(want.Data)
	for i := range want.Data {
		if e := math.Abs(got.Data[i]-want.Data[i]) / scaleRef; e > relErr {
			relErr = e
		}
	}
	if relErr > 0.05 {
		t.Errorf("dynamic quantized linear max relative error %g exceeds 0.05", relErr)
	}
	if got.RequiresGrad {
		t.Errorf("quantized forward must not carry autograd state")
	}
}

func TestDynamicLinearMultiDimInput(t *testing.T) {
	l := nn.NewLinear(16, 8, true)
	d := NewDynamicLinearFrom(l)
	x3 := randn(3, 2, 4, 16)
	got := d.Forward(x3)
	if len(got.Shape) != 3 || got.Shape[0] != 2 || got.Shape[1] != 4 || got.Shape[2] != 8 {
		t.Fatalf("output shape %v, want [2 4 8]", got.Shape)
	}
	// Must equal the flattened-batch computation exactly.
	flat := d.Forward(x3.Reshape(8, 16))
	for i := range got.Data {
		if got.Data[i] != flat.Data[i] {
			t.Fatalf("multi-dim forward diverges from flattened forward at %d", i)
		}
	}
}

func TestDynamicLinearNoBias(t *testing.T) {
	l := nn.NewLinear(16, 8, false)
	x := randn(4, 8, 16)
	want := l.Forward(x)
	got := NewDynamicLinearFrom(l).Forward(x)
	scaleRef := maxAbs(want.Data)
	for i := range want.Data {
		if math.Abs(got.Data[i]-want.Data[i])/scaleRef > 0.05 {
			t.Fatalf("bias-free dynamic linear error too large at %d: got %g want %g",
				i, got.Data[i], want.Data[i])
		}
	}
}

func TestStaticLinearWithCalibration(t *testing.T) {
	l := nn.NewLinear(64, 64, true)
	calib := randn(5, 32, 64)
	var obs MinMaxObserver
	obs.Observe(calib)

	s := NewStaticLinearFrom(l, &obs)
	x := randn(6, 32, 64) // same distribution as calibration data
	want := l.Forward(x)
	got := s.Forward(x)

	relErr := 0.0
	scaleRef := maxAbs(want.Data)
	for i := range want.Data {
		if e := math.Abs(got.Data[i]-want.Data[i]) / scaleRef; e > relErr {
			relErr = e
		}
	}
	// Static qparams saturate unseen tails, so the bound is looser than the
	// dynamic case but must stay in the same ballpark.
	if relErr > 0.1 {
		t.Errorf("static quantized linear max relative error %g exceeds 0.1", relErr)
	}
}
