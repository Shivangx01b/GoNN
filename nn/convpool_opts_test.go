package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// Tests for conv padding_mode and the pool padding/ceil_mode/dilation/
// count_include_pad options. Reuses tensorsEqualExact (attention_masked_test)
// and expectPanic (conv_groups_test).

func tensorsAlmostEqual(t *testing.T, name string, a, b *tensor.Tensor, tol float64) {
	t.Helper()
	if !intsEqual(a.Shape, b.Shape) {
		t.Fatalf("%s: shape %v != %v", name, a.Shape, b.Shape)
	}
	for i := range a.Data {
		if math.Abs(a.Data[i]-b.Data[i]) > tol {
			t.Fatalf("%s: data[%d]: %.17g != %.17g", name, i, a.Data[i], b.Data[i])
		}
	}
}

// ---- Conv padding_mode ------------------------------------------------------

// A padding_mode conv must equal: manually pre-pad the input with the
// matching pad layer, then run the SAME weights with zero padding. Both paths
// perform identical gather matmuls, so equality is exact.
func TestConvPaddingModeCircularMatchesManualPrePad(t *testing.T) {
	x := seededRandn(200, 2, 2, 4, 5)
	cm := NewConv2d(2, 3, 3, WithPad(1), WithPaddingMode("circular"))
	cz := NewConv2d(2, 3, 3) // zero internal padding
	copy(cz.Weight.Data, cm.Weight.Data)
	copy(cz.Bias.Data, cm.Bias.Data)

	pre := NewCircularPad2d(1, 1, 1, 1).Forward(x)
	got, want := cm.Forward(x), cz.Forward(pre)
	if !intsEqual(got.Shape, want.Shape) {
		t.Fatalf("shape %v != %v", got.Shape, want.Shape)
	}
	tensorsEqualExact(t, "circular conv2d", got, want)

	// Output size must match the zeros-mode formula.
	ref := NewConv2d(2, 3, 3, WithPad(1))
	if !intsEqual(got.Shape, ref.Forward(x).Shape) {
		t.Fatalf("circular conv output shape %v != zeros-mode shape", got.Shape)
	}
}

func TestConvPaddingModeCircular1d(t *testing.T) {
	x := seededRandn(201, 2, 2, 5)
	cm := NewConv1d(2, 3, 3, WithPad(2), WithPaddingMode("circular"))
	cz := NewConv1d(2, 3, 3)
	copy(cz.Weight.Data, cm.Weight.Data)
	copy(cz.Bias.Data, cm.Bias.Data)
	pre := NewCircularPad1d(2, 2).Forward(x)
	tensorsEqualExact(t, "circular conv1d", cm.Forward(x), cz.Forward(pre))
}

func TestConvPaddingModeReflectReplicateMatchManualPrePad(t *testing.T) {
	x := seededRandn(202, 2, 2, 4, 5)

	cr := NewConv2d(2, 3, 3, WithPad(1), WithPaddingMode("reflect"))
	cz := NewConv2d(2, 3, 3)
	copy(cz.Weight.Data, cr.Weight.Data)
	copy(cz.Bias.Data, cr.Bias.Data)
	pre := NewReflectionPad2d(1, 1, 1, 1).Forward(x)
	tensorsEqualExact(t, "reflect conv2d", cr.Forward(x), cz.Forward(pre))

	cp := NewConv2d(2, 3, 3, WithPad(1), WithPaddingMode("replicate"))
	copy(cz.Weight.Data, cp.Weight.Data)
	copy(cz.Bias.Data, cp.Bias.Data)
	pre = NewReplicationPad2d(1, 1, 1, 1).Forward(x)
	tensorsEqualExact(t, "replicate conv2d", cp.Forward(x), cz.Forward(pre))
}

// padding_mode must compose with groups: pre-pad happens once, before the
// per-group channel slicing.
func TestConvPaddingModeWithGroups(t *testing.T) {
	x := seededRandn(203, 2, 4, 5, 5)
	cm := NewConv2d(4, 6, 3, WithPad(1), WithGroups(2), WithPaddingMode("circular"))
	cz := NewConv2d(4, 6, 3, WithGroups(2))
	copy(cz.Weight.Data, cm.Weight.Data)
	copy(cz.Bias.Data, cm.Bias.Data)
	pre := NewCircularPad2d(1, 1, 1, 1).Forward(x)
	tensorsEqualExact(t, "circular grouped conv2d", cm.Forward(x), cz.Forward(pre))
}

func TestConvPaddingModeZerosBitIdentical(t *testing.T) {
	x := seededRandn(204, 2, 2, 6, 7)
	a := NewConv2d(2, 3, 3, WithStride(2), WithPad(1), WithPaddingMode("zeros"))
	b := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
	copy(b.Weight.Data, a.Weight.Data)
	copy(b.Bias.Data, a.Bias.Data)
	tensorsEqualExact(t, "explicit zeros mode", a.Forward(x), b.Forward(x))
}

func TestConvPaddingModeValidation(t *testing.T) {
	expectPanic(t, "invalid mode string", func() {
		NewConv2d(2, 3, 3, WithPaddingMode("bogus"))
	})
	expectPanic(t, "ConvTranspose1d non-zeros", func() {
		NewConvTranspose1d(2, 3, 3, WithPaddingMode("reflect"))
	})
	expectPanic(t, "ConvTranspose2d non-zeros", func() {
		NewConvTranspose2d(2, 3, 3, WithPaddingMode("circular"))
	})
	expectPanic(t, "ConvTranspose3d non-zeros", func() {
		NewConvTranspose3d(2, 2, 2, WithPaddingMode("replicate"))
	})
	// reflect requires pad < input size (checked when the input is seen).
	expectPanic(t, "reflect pad too large", func() {
		c := NewConv1d(1, 1, 3, WithPad(2), WithPaddingMode("reflect"))
		c.Forward(tensor.New([]float64{1, 2}, 1, 1, 2))
	})
}

func TestGradCheckConvReflectPaddingMode(t *testing.T) {
	c := NewConv2d(2, 3, 3, WithPad(1), WithPaddingMode("reflect"))
	x := seededRandn(205, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "ConvReflectPaddingMode", loss, append(c.Parameters(), x), gcEps, gcTol, 40)
}

// ---- Pool padding -----------------------------------------------------------

// Max pooling pads with a huge negative sentinel, NOT zero: on an all-negative
// input every window must pick the real maximum, never a pad cell.
func TestMaxPoolPaddingSentinel(t *testing.T) {
	x := tensor.New([]float64{-1, -2, -3, -4}, 1, 1, 2, 2).SetRequiresGrad(true)
	p := NewMaxPool2d(2, WithPoolPadding(1))
	y := p.Forward(x)
	want := tensor.New([]float64{-1, -2, -3, -4}, 1, 1, 2, 2)
	tensorsEqualExact(t, "max pool sentinel", y, want)
	if !intsEqual(y.Shape, []int{1, 1, 2, 2}) {
		t.Fatalf("shape %v", y.Shape)
	}

	// Gradient flows only through the real cells (each selected exactly once).
	y.Sum().Backward()
	for i, g := range x.Grad.Data {
		if g != 1 {
			t.Fatalf("grad[%d] = %g, want 1", i, g)
		}
	}
}

// PyTorch hand-check: AvgPool2d(2, stride=2, padding=1) on [[1,2],[3,4]].
// count_include_pad=true divides by the full window (4); false divides by the
// number of real cells (1 per corner window).
func TestAvgPoolCountIncludePad(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2)

	inc := NewAvgPool2d(2, WithPoolPadding(1))
	tensorsEqualExact(t, "count_include_pad=true",
		inc.Forward(x), tensor.New([]float64{0.25, 0.5, 0.75, 1}, 1, 1, 2, 2))

	exc := NewAvgPool2d(2, WithPoolPadding(1), WithCountIncludePad(false))
	tensorsEqualExact(t, "count_include_pad=false",
		exc.Forward(x), tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2))
}

// ---- Pool ceil_mode ---------------------------------------------------------

// Output sizes: out = ceil((in + 2p - d*(k-1) - 1)/s) + 1, minus 1 where the
// last window would start inside the right padding ((out-1)*s >= in + p).
func TestPoolCeilModeShapes(t *testing.T) {
	cases := []struct {
		in, k, s, pad int
		ceil          bool
		want          int
	}{
		{5, 2, 2, 0, false, 2},
		{5, 2, 2, 0, true, 3},
		{6, 2, 2, 0, true, 3}, // stride divides evenly: ceil == floor
		{4, 2, 3, 1, true, 2}, // ceil gives 3; clip rule ((3-1)*3 >= 4+1) trims to 2
		{3, 2, 2, 1, true, 2}, // ceil gives 3; clip rule ((3-1)*2 >= 3+1) trims to 2
		{4, 3, 2, 1, true, 3}, // last window overhangs the padded end by 1
	}
	for _, c := range cases {
		opts := []PoolOpt{WithPoolStride(c.s), WithPoolPadding(c.pad)}
		if c.ceil {
			opts = append(opts, WithPoolCeilMode())
		}
		x := seededRandn(206, 1, 1, c.in)
		for _, layer := range []interface {
			Forward(*tensor.Tensor) *tensor.Tensor
		}{
			NewMaxPool1d(c.k, opts...),
			NewAvgPool1d(c.k, opts...),
		} {
			got := layer.Forward(x).Shape[2]
			if got != c.want {
				t.Errorf("in=%d k=%d s=%d pad=%d ceil=%v: out=%d, want %d",
					c.in, c.k, c.s, c.pad, c.ceil, got, c.want)
			}
		}
	}
}

// PyTorch hand-check (avg_pool1d([1,2,3,4], kernel=3, stride=2, padding=1,
// ceil_mode=True)): the last window starts at 3, covers only cell 3 plus the
// right pad, and overhangs the padded end by 1, so with count_include_pad the
// divisor is clipped to the PADDED extent (2, not 3): [1, 3, 2]. With
// count_include_pad=false the divisors are the real-cell counts [2, 3, 1]:
// [1.5, 3, 4].
func TestAvgPoolCeilModeDivisors(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 4)

	inc := NewAvgPool1d(3, WithPoolStride(2), WithPoolPadding(1), WithPoolCeilMode())
	tensorsAlmostEqual(t, "ceil include_pad",
		inc.Forward(x), tensor.New([]float64{1, 3, 2}, 1, 1, 3), 1e-12)

	exc := NewAvgPool1d(3, WithPoolStride(2), WithPoolPadding(1), WithPoolCeilMode(),
		WithCountIncludePad(false))
	tensorsAlmostEqual(t, "ceil exclude_pad",
		exc.Forward(x), tensor.New([]float64{1.5, 3, 4}, 1, 1, 3), 1e-12)
}

// No padding, ceil mode: windows clipped at the input end divide by the
// clipped extent (both count_include_pad settings agree with no pad).
func TestPoolCeilModeValues2d(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)

	avg := NewAvgPool2d(2, WithPoolCeilMode())
	tensorsAlmostEqual(t, "avg ceil 3x3",
		avg.Forward(x), tensor.New([]float64{3, 4.5, 7.5, 9}, 1, 1, 2, 2), 1e-12)

	max := NewMaxPool2d(2, WithPoolCeilMode())
	tensorsEqualExact(t, "max ceil 3x3",
		max.Forward(x), tensor.New([]float64{5, 6, 8, 9}, 1, 1, 2, 2))
}

// ---- Pool dilation ----------------------------------------------------------

func TestMaxPoolDilation(t *testing.T) {
	// 1D: kernel 2 dilation 2 taps (i, i+2).
	x1 := tensor.New([]float64{1, 5, 2, 4, 3}, 1, 1, 5)
	p1 := NewMaxPool1d(2, WithPoolStride(1), WithPoolDilation(2))
	tensorsEqualExact(t, "dilated max 1d",
		p1.Forward(x1), tensor.New([]float64{2, 5, 3}, 1, 1, 3))

	// 2D: kernel 2 dilation 2 on 3x3 taps the four corners.
	x2 := tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	p2 := NewMaxPool2d(2, WithPoolDilation(2))
	tensorsEqualExact(t, "dilated max 2d",
		p2.Forward(x2), tensor.New([]float64{9}, 1, 1, 1, 1))
}

// ---- ForwardWithIndices with the new options --------------------------------

// Indices always refer to the UNPADDED input's flat spatial positions, so a
// padded pool round-trips through MaxUnpool (with an explicit output size,
// since unpool has no padding).
func TestMaxPoolIndicesPaddingRoundTrip(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	p := NewMaxPool2d(2, WithPoolPadding(1))
	y, idx := p.ForwardWithIndices(x)
	tensorsEqualExact(t, "padded indices output vs Forward", y, p.Forward(x))
	tensorsEqualExact(t, "padded indices", idx, tensor.New([]float64{0, 1, 2, 3}, 1, 1, 2, 2))

	u := NewMaxUnpool2d(2)
	rec := u.Forward(y, idx, 2, 2)
	tensorsEqualExact(t, "unpool round trip", rec, x)
}

func TestMaxPoolIndicesPaddingDilationCeil(t *testing.T) {
	x := seededRandn(207, 1, 2, 6, 5)
	p := NewMaxPool2d(2, WithPoolStride(2), WithPoolPadding(1),
		WithPoolDilation(2), WithPoolCeilMode())
	y1 := p.Forward(x)
	y2, idx := p.ForwardWithIndices(x)
	tensorsEqualExact(t, "indices output vs Forward", y2, y1)
	if !intsEqual(y2.Shape, []int{1, 2, 4, 3}) {
		t.Fatalf("output shape %v, want [1 2 4 3]", y2.Shape)
	}

	// Every index is a valid unpadded flat position whose value is the max.
	H, W := 6, 5
	numWin := 4 * 3
	for i := 0; i < 2; i++ { // (n, c) planes
		for w := 0; w < numWin; w++ {
			fi := int(idx.Data[i*numWin+w])
			if fi < 0 || fi >= H*W {
				t.Fatalf("plane %d win %d: index %d out of range [0,%d)", i, w, fi, H*W)
			}
			if got, want := x.Data[i*H*W+fi], y2.Data[i*numWin+w]; got != want {
				t.Fatalf("plane %d win %d: x[%d]=%g != pooled %g", i, w, fi, got, want)
			}
		}
	}
}

// ---- Option validation --------------------------------------------------------

func TestPoolOptionValidation(t *testing.T) {
	expectPanic(t, "pad > effective kernel / 2", func() {
		NewMaxPool2d(2, WithPoolPadding(2))
	})
	expectPanic(t, "avg pool dilation", func() {
		NewAvgPool2d(2, WithPoolDilation(2))
	})
	expectPanic(t, "count_include_pad on max pool", func() {
		NewMaxPool2d(2, WithCountIncludePad(false))
	})
	// Dilation raises the effective kernel, admitting a larger pad.
	NewMaxPool2d(2, WithPoolPadding(1), WithPoolDilation(2)) // eff 3: pad 1 ok
}

// ---- Gradchecks through the opt-in paths -------------------------------------

func TestGradCheckMaxPoolPaddedCeil(t *testing.T) {
	p := NewMaxPool2d(2, WithPoolStride(2), WithPoolPadding(1), WithPoolCeilMode())
	x := seededRandn(208, 2, 2, 5, 5).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "MaxPoolPaddedCeil", loss, []*tensor.Tensor{x}, gcEps, gcTol, 40)
}

func TestGradCheckAvgPoolPaddedNoIncludePad(t *testing.T) {
	p := NewAvgPool2d(3, WithPoolStride(2), WithPoolPadding(1), WithCountIncludePad(false))
	x := seededRandn(209, 2, 2, 5, 5).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "AvgPoolPaddedNoIncludePad", loss, []*tensor.Tensor{x}, gcEps, gcTol, 40)
}

func TestGradCheckMaxPoolDilated(t *testing.T) {
	p := NewMaxPool2d(2, WithPoolStride(1), WithPoolDilation(2))
	x := seededRandn(210, 1, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "MaxPoolDilated", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}
