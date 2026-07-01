package quant

import (
	"math"
	"math/rand"
	"testing"

	"gonn/nn"
	"gonn/tensor"
)

// wildRowMatrix returns an (out, in) weight matrix whose rows have wildly
// different magnitudes — the case per-tensor symmetric quantization handles
// badly (one scale must span all rows) and per-channel handles well.
func wildRowMatrix(seed int64, out, in int, rowScales []float64) *tensor.Tensor {
	rng := rand.New(rand.NewSource(seed))
	w := tensor.Zeros(out, in)
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			// values in [0.5, 1.0) * rowScale, random sign: every row fully
			// exercises its own magnitude band.
			v := (0.5 + 0.5*rng.Float64()) * rowScales[o]
			if rng.Intn(2) == 0 {
				v = -v
			}
			w.Data[o*in+i] = v
		}
	}
	return w
}

func TestPerChannelRoundTrip(t *testing.T) {
	rowScales := []float64{100, 1, 0.01, 0.5}
	w := wildRowMatrix(11, 4, 8, rowScales)

	q := QuantizePerChannel(w, 0)
	if q.Axis != 0 || len(q.Scales) != 4 || len(q.ZeroPoints) != 4 {
		t.Fatalf("per-channel metadata wrong: axis=%d scales=%d zps=%d", q.Axis, len(q.Scales), len(q.ZeroPoints))
	}
	for c, zp := range q.ZeroPoints {
		if zp != 0 {
			t.Errorf("symmetric per-channel zero point for row %d = %d, want 0", c, zp)
		}
	}
	d := DequantizePerChannel(q)
	if len(d.Shape) != 2 || d.Shape[0] != 4 || d.Shape[1] != 8 {
		t.Fatalf("dequantized shape %v, want [4 8]", d.Shape)
	}
	// Round-trip error of every element in row c is bounded by Scales[c]/2.
	for o := 0; o < 4; o++ {
		for i := 0; i < 8; i++ {
			idx := o*8 + i
			if e := math.Abs(d.Data[idx] - w.Data[idx]); e > q.Scales[o]/2+1e-12 {
				t.Errorf("row %d elem %d: round-trip error %g exceeds scale/2 = %g", o, i, e, q.Scales[o]/2)
			}
		}
	}
}

func TestPerChannelAxis1(t *testing.T) {
	// Columns with different magnitudes, quantized along axis 1.
	w := tensor.New([]float64{
		100, 0.01,
		-50, 0.005,
		75, -0.008,
	}, 3, 2)
	q := QuantizePerChannel(w, 1)
	if len(q.Scales) != 2 {
		t.Fatalf("axis=1 should give 2 channels, got %d", len(q.Scales))
	}
	d := DequantizePerChannel(q)
	for r := 0; r < 3; r++ {
		for c := 0; c < 2; c++ {
			idx := r*2 + c
			if e := math.Abs(d.Data[idx] - w.Data[idx]); e > q.Scales[c]/2+1e-12 {
				t.Errorf("[%d,%d]: error %g exceeds column scale/2 = %g", r, c, e, q.Scales[c]/2)
			}
		}
	}
}

// maxRowRelErr returns the max over elements of |deq - orig| relative to the
// element's row max magnitude (so tiny rows are not drowned out by big ones).
func maxRowRelErr(orig, deq *tensor.Tensor) float64 {
	out, in := orig.Shape[0], orig.Shape[1]
	worst := 0.0
	for o := 0; o < out; o++ {
		ref := maxAbs(orig.Data[o*in : (o+1)*in])
		if ref == 0 {
			continue
		}
		for i := 0; i < in; i++ {
			idx := o*in + i
			if e := math.Abs(deq.Data[idx]-orig.Data[idx]) / ref; e > worst {
				worst = e
			}
		}
	}
	return worst
}

func TestPerChannelBeatsPerTensor(t *testing.T) {
	rowScales := []float64{100, 1, 0.01, 0.5}
	w := wildRowMatrix(12, 4, 8, rowScales)

	// Per-tensor symmetric: one scale spans the 100-magnitude row and the
	// 0.01-magnitude row.
	ptScale := symmetricQParams(w.Data)
	ptDeq := Dequantize(Quantize(w, ptScale, 0))
	ptErr := maxRowRelErr(w, ptDeq)

	pcDeq := DequantizePerChannel(QuantizePerChannel(w, 0))
	pcErr := maxRowRelErr(w, pcDeq)

	t.Logf("max per-row relative error: per-tensor=%g per-channel=%g", ptErr, pcErr)
	if !(pcErr < ptErr) {
		t.Errorf("per-channel max relative error %g not lower than per-tensor %g", pcErr, ptErr)
	}
	// The gap must be structural, not luck: per-tensor loses the small rows
	// entirely (relative error ~1), per-channel keeps every row within
	// scale_i/2 of one row-max (<= 1/254 per element).
	if pcErr > 10*1.0/254 {
		t.Errorf("per-channel relative error %g larger than a few quantization steps", pcErr)
	}
}

func TestDynamicLinearPerChannelCloserToFloat(t *testing.T) {
	const in, out = 8, 4
	rowScales := []float64{100, 1, 0.01, 0.5}
	w := wildRowMatrix(13, out, in, rowScales)

	l := nn.NewLinear(in, out, false)
	copy(l.Weight.Data, w.Data)

	x := seededTensor(14, func(r *rand.Rand) float64 { return -1 + 2*r.Float64() }, 16, in)
	want := l.Forward(x)

	pt := NewDynamicLinearFrom(l).Forward(x)
	pc := NewDynamicLinearFrom(l, WithPerChannelWeights()).Forward(x)

	// Compare per output channel, relative to that channel's float magnitude:
	// per-tensor wipes out the small-weight rows' outputs.
	batch := 16
	chanRef := make([]float64, out)
	for b := 0; b < batch; b++ {
		for o := 0; o < out; o++ {
			if a := math.Abs(want.Data[b*out+o]); a > chanRef[o] {
				chanRef[o] = a
			}
		}
	}
	relErr := func(got *tensor.Tensor) float64 {
		worst := 0.0
		for b := 0; b < batch; b++ {
			for o := 0; o < out; o++ {
				idx := b*out + o
				if chanRef[o] == 0 {
					continue
				}
				if e := math.Abs(got.Data[idx]-want.Data[idx]) / chanRef[o]; e > worst {
					worst = e
				}
			}
		}
		return worst
	}
	ptErr, pcErr := relErr(pt), relErr(pc)
	t.Logf("dynamic linear max per-channel relative error: per-tensor=%g per-channel=%g", ptErr, pcErr)
	if !(pcErr < ptErr) {
		t.Errorf("per-channel dynamic linear error %g not lower than per-tensor %g", pcErr, ptErr)
	}
	if pcErr > 0.05 {
		t.Errorf("per-channel dynamic linear relative error %g exceeds 0.05", pcErr)
	}
}

func TestPerChannelDefaultUnchanged(t *testing.T) {
	// Without the option the core must behave exactly like the historical
	// per-tensor path: WScale set, PerChannel false, and every row scale equal.
	l := nn.NewLinear(16, 8, true)
	d := NewDynamicLinearFrom(l)
	if d.PerChannel {
		t.Fatal("default DynamicLinear must be per-tensor")
	}
	if d.WScale <= 0 {
		t.Fatalf("per-tensor WScale = %g, want > 0", d.WScale)
	}
	for o, s := range d.WScales {
		if s != d.WScale {
			t.Fatalf("row %d scale %g != per-tensor scale %g", o, s, d.WScale)
		}
	}
	pc := NewDynamicLinearFrom(l, WithPerChannelWeights())
	if !pc.PerChannel {
		t.Fatal("WithPerChannelWeights must set PerChannel")
	}
	if pc.WScale != 0 {
		t.Fatalf("per-channel WScale = %g, want 0 (unused)", pc.WScale)
	}
}
