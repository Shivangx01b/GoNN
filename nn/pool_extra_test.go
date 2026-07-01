package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// ---- LPPool -----------------------------------------------------------------

func TestLPPool1dHandValues(t *testing.T) {
	// p=2, kernel 2, default stride = kernel: windows [1,2] and [3,4].
	p := NewLPPool1d(2, 2)
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 4)
	out := p.Forward(x)
	if !intsEqual(out.Shape, []int{1, 1, 2}) {
		t.Fatalf("shape = %v, want [1 1 2]", out.Shape)
	}
	want := []float64{math.Sqrt(5), 5} // sqrt(1+4), sqrt(9+16)
	for i, w := range want {
		if math.Abs(out.Data[i]-w) > 1e-12 {
			t.Errorf("out[%d] = %g, want %g", i, out.Data[i], w)
		}
	}

	// Overlapping windows via stride override.
	po := NewLPPool1d(2, 2, WithPoolStride(1))
	out = po.Forward(x)
	want = []float64{math.Sqrt(5), math.Sqrt(13), 5}
	if !intsEqual(out.Shape, []int{1, 1, 3}) {
		t.Fatalf("stride-1 shape = %v, want [1 1 3]", out.Shape)
	}
	for i, w := range want {
		if math.Abs(out.Data[i]-w) > 1e-12 {
			t.Errorf("stride-1 out[%d] = %g, want %g", i, out.Data[i], w)
		}
	}

	// p=1 is sum pooling.
	ps := NewLPPool1d(1, 2)
	out = ps.Forward(x)
	want = []float64{3, 7}
	for i, w := range want {
		if math.Abs(out.Data[i]-w) > 1e-12 {
			t.Errorf("p=1 out[%d] = %g, want %g", i, out.Data[i], w)
		}
	}
}

// lpPool2dBrute computes (sum x^p)^(1/p) over each window directly.
func lpPool2dBrute(x *tensor.Tensor, p float64, kh, kw, sh, sw int) *tensor.Tensor {
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	oh := (H-kh)/sh + 1
	ow := (W-kw)/sw + 1
	out := tensor.Zeros(N, C, oh, ow)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for i := 0; i < oh; i++ {
				for j := 0; j < ow; j++ {
					s := 0.0
					for a := 0; a < kh; a++ {
						for b := 0; b < kw; b++ {
							v := x.Data[((n*C+c)*H+i*sh+a)*W+j*sw+b]
							s += math.Pow(v, p)
						}
					}
					out.Data[((n*C+c)*oh+i)*ow+j] = math.Pow(s, 1/p)
				}
			}
		}
	}
	return out
}

func TestLPPool2dVsBrute(t *testing.T) {
	// p=3 needs positive inputs (fractional 1/p on a negative sum is NaN,
	// matching PyTorch); shift a random draw to be strictly positive.
	x := seededRandn(31, 2, 3, 5, 6)
	for i := range x.Data {
		x.Data[i] = math.Abs(x.Data[i]) + 0.5
	}
	p := NewLPPool2d(3, 2, WithPoolStride(1, 2))
	got := p.Forward(x)
	want := lpPool2dBrute(x, 3, 2, 2, 1, 2)
	if !intsEqual(got.Shape, want.Shape) {
		t.Fatalf("shape = %v, want %v", got.Shape, want.Shape)
	}
	for i := range want.Data {
		if math.Abs(got.Data[i]-want.Data[i]) > 1e-10 {
			t.Fatalf("out[%d] = %g, want %g", i, got.Data[i], want.Data[i])
		}
	}
}

func TestLPPool3dShape(t *testing.T) {
	p := NewLPPool3d(2, 2)
	x := seededRandn(32, 1, 2, 4, 4, 4)
	out := p.Forward(x)
	if !intsEqual(out.Shape, []int{1, 2, 2, 2, 2}) {
		t.Fatalf("shape = %v, want [1 2 2 2 2]", out.Shape)
	}
	// Spot-check one cell: window (0,0,0) of (n=0,c=0).
	s := 0.0
	for d := 0; d < 2; d++ {
		for h := 0; h < 2; h++ {
			for w := 0; w < 2; w++ {
				v := x.Data[(d*4+h)*4+w]
				s += v * v
			}
		}
	}
	if math.Abs(out.Data[0]-math.Sqrt(s)) > 1e-12 {
		t.Errorf("out[0] = %g, want %g", out.Data[0], math.Sqrt(s))
	}
}

func TestLPPoolInvalidNormType(t *testing.T) {
	for _, p := range []float64{0, -1, math.Inf(1)} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("NewLPPool1d(%v, 2) did not panic", p)
				}
			}()
			NewLPPool1d(p, 2)
		}()
	}
}

func TestGradCheckLPPool(t *testing.T) {
	// p=2 handles negative inputs (x^2 is smooth, window sums are positive
	// almost surely for random data).
	p1 := NewLPPool1d(2, 2)
	x1 := seededRandn(33, 2, 2, 6).SetRequiresGrad(true)
	gradCheck(t, "LPPool1d(p=2)",
		func() *tensor.Tensor { return p1.Forward(x1).Square().Mean() },
		[]*tensor.Tensor{x1}, gcEps, gcTol, 0)

	// Fractional 1/p: strictly positive input.
	p2 := NewLPPool2d(3, 2, WithPoolStride(1))
	x2 := seededRandn(34, 1, 2, 4, 5)
	for i := range x2.Data {
		x2.Data[i] = math.Abs(x2.Data[i]) + 0.5
	}
	x2.SetRequiresGrad(true)
	gradCheck(t, "LPPool2d(p=3)",
		func() *tensor.Tensor { return p2.Forward(x2).Square().Mean() },
		[]*tensor.Tensor{x2}, gcEps, gcTol, 0)
}

// ---- MaxPool ForwardWithIndices ----------------------------------------------

// arange returns a tensor with values 0..n-1 in the given shape.
func arange(shape ...int) *tensor.Tensor {
	x := tensor.Zeros(shape...)
	for i := range x.Data {
		x.Data[i] = float64(i)
	}
	return x
}

func TestMaxPool1dForwardWithIndices(t *testing.T) {
	x := arange(1, 1, 6)
	p := NewMaxPool1d(2)
	out, idx := p.ForwardWithIndices(x)
	wantOut := []float64{1, 3, 5}
	wantIdx := []float64{1, 3, 5}
	for i := range wantOut {
		if out.Data[i] != wantOut[i] || idx.Data[i] != wantIdx[i] {
			t.Errorf("cell %d: out=%g idx=%g, want out=%g idx=%g",
				i, out.Data[i], idx.Data[i], wantOut[i], wantIdx[i])
		}
	}

	// Overlapping windows: kernel 3, stride 2 -> windows [0..2], [2..4].
	po := NewMaxPool1d(3, WithPoolStride(2))
	out, idx = po.ForwardWithIndices(x)
	wantOut = []float64{2, 4}
	wantIdx = []float64{2, 4}
	for i := range wantOut {
		if out.Data[i] != wantOut[i] || idx.Data[i] != wantIdx[i] {
			t.Errorf("k3s2 cell %d: out=%g idx=%g, want out=%g idx=%g",
				i, out.Data[i], idx.Data[i], wantOut[i], wantIdx[i])
		}
	}
}

func TestMaxPool2dForwardWithIndices(t *testing.T) {
	// Channel 0 ascending 0..15 (max at bottom-right of each window),
	// channel 1 descending 15..0 (max at top-left). Indices are flat within
	// each (n, c) plane: h*W + w.
	x := tensor.Zeros(1, 2, 4, 4)
	for i := 0; i < 16; i++ {
		x.Data[i] = float64(i)
		x.Data[16+i] = float64(15 - i)
	}
	p := NewMaxPool2d(2)
	out, idx := p.ForwardWithIndices(x)
	if !intsEqual(idx.Shape, []int{1, 2, 2, 2}) {
		t.Fatalf("indices shape = %v, want [1 2 2 2]", idx.Shape)
	}
	wantOut := []float64{5, 7, 13, 15 /* ch0 */, 15, 13, 7, 5 /* ch1 */}
	wantIdx := []float64{5, 7, 13, 15 /* ch0 */, 0, 2, 8, 10 /* ch1 */}
	for i := range wantOut {
		if out.Data[i] != wantOut[i] || idx.Data[i] != wantIdx[i] {
			t.Errorf("cell %d: out=%g idx=%g, want out=%g idx=%g",
				i, out.Data[i], idx.Data[i], wantOut[i], wantIdx[i])
		}
	}

	// The pooled output must match plain Forward exactly.
	fwd := p.Forward(x)
	for i := range fwd.Data {
		if fwd.Data[i] != out.Data[i] {
			t.Fatalf("ForwardWithIndices output diverges from Forward at %d: %g != %g",
				i, out.Data[i], fwd.Data[i])
		}
	}
}

func TestMaxPool3dForwardWithIndices(t *testing.T) {
	x := arange(1, 1, 2, 4, 4)
	p := NewMaxPool3d(2)
	out, idx := p.ForwardWithIndices(x)
	// Windows start at (0, 0/2, 0/2); max is the (1,1,1) corner:
	// flat = (d*4+h)*4 + w with d=1, h=start_h+1, w=start_w+1.
	wantOut := []float64{21, 23, 29, 31}
	wantIdx := []float64{21, 23, 29, 31}
	for i := range wantOut {
		if out.Data[i] != wantOut[i] || idx.Data[i] != wantIdx[i] {
			t.Errorf("cell %d: out=%g idx=%g, want out=%g idx=%g",
				i, out.Data[i], idx.Data[i], wantOut[i], wantIdx[i])
		}
	}
}

// ---- MaxUnpool ----------------------------------------------------------------

func TestMaxUnpoolRoundTrip2d(t *testing.T) {
	// Distinct random values (continuous draw: ties have measure zero).
	x := seededRandn(41, 2, 2, 4, 6)
	p := NewMaxPool2d(2)
	u := NewMaxUnpool2d(2)
	out, idx := p.ForwardWithIndices(x)
	rec := u.Forward(out, idx)
	if !intsEqual(rec.Shape, x.Shape) {
		t.Fatalf("round-trip shape = %v, want %v", rec.Shape, x.Shape)
	}
	// Expected: zero everywhere except each selected max at its position.
	want := tensor.Zeros(x.Shape...)
	cells := len(out.Data) / (2 * 2) // out cells per plane
	plane := 4 * 6
	for i := 0; i < 2*2; i++ {
		for j := 0; j < cells; j++ {
			want.Data[i*plane+int(idx.Data[i*cells+j])] = out.Data[i*cells+j]
		}
	}
	for i := range want.Data {
		if rec.Data[i] != want.Data[i] {
			t.Fatalf("rec[%d] = %g, want %g", i, rec.Data[i], want.Data[i])
		}
	}
	// Every non-zero cell of rec must equal the input there (maxima preserved).
	nz := 0
	for i := range rec.Data {
		if rec.Data[i] != 0 {
			nz++
			if rec.Data[i] != x.Data[i] {
				t.Fatalf("rec[%d] = %g, input has %g", i, rec.Data[i], x.Data[i])
			}
		}
	}
	if nz != len(out.Data) {
		t.Fatalf("non-zero cells = %d, want %d", nz, len(out.Data))
	}
}

func TestMaxUnpoolRoundTrip1dAnd3d(t *testing.T) {
	x1 := seededRandn(42, 1, 3, 8)
	p1 := NewMaxPool1d(2)
	u1 := NewMaxUnpool1d(2)
	o1, i1 := p1.ForwardWithIndices(x1)
	r1 := u1.Forward(o1, i1)
	if !intsEqual(r1.Shape, x1.Shape) {
		t.Fatalf("1d round-trip shape = %v, want %v", r1.Shape, x1.Shape)
	}
	for i := range r1.Data {
		if r1.Data[i] != 0 && r1.Data[i] != x1.Data[i] {
			t.Fatalf("1d rec[%d] = %g, input has %g", i, r1.Data[i], x1.Data[i])
		}
	}

	x3 := seededRandn(43, 1, 2, 2, 4, 4)
	p3 := NewMaxPool3d(2)
	u3 := NewMaxUnpool3d(2)
	o3, i3 := p3.ForwardWithIndices(x3)
	r3 := u3.Forward(o3, i3)
	if !intsEqual(r3.Shape, x3.Shape) {
		t.Fatalf("3d round-trip shape = %v, want %v", r3.Shape, x3.Shape)
	}
	for i := range r3.Data {
		if r3.Data[i] != 0 && r3.Data[i] != x3.Data[i] {
			t.Fatalf("3d rec[%d] = %g, input has %g", i, r3.Data[i], x3.Data[i])
		}
	}
}

func TestMaxUnpoolExplicitOutputSize(t *testing.T) {
	// Input length 5, kernel 2 -> pooled length 2; default inverse is 4, so
	// the true size must be passed explicitly.
	x := seededRandn(44, 1, 1, 5)
	p := NewMaxPool1d(2)
	u := NewMaxUnpool1d(2)
	out, idx := p.ForwardWithIndices(x)
	rec := u.Forward(out, idx, 5)
	if !intsEqual(rec.Shape, []int{1, 1, 5}) {
		t.Fatalf("explicit-size shape = %v, want [1 1 5]", rec.Shape)
	}
	def := u.Forward(out, idx)
	if !intsEqual(def.Shape, []int{1, 1, 4}) {
		t.Fatalf("default-size shape = %v, want [1 1 4]", def.Shape)
	}
}

func TestGradCheckMaxUnpool(t *testing.T) {
	// Fixed indices from a real pooling; gradcheck w.r.t. the pooled values.
	base := seededRandn(45, 1, 2, 4, 4)
	p := NewMaxPool2d(2)
	_, idx := p.ForwardWithIndices(base)
	u := NewMaxUnpool2d(2)
	x := seededRandn(46, 1, 2, 2, 2).SetRequiresGrad(true)
	gradCheck(t, "MaxUnpool2d",
		func() *tensor.Tensor { return u.Forward(x, idx).Square().Mean() },
		[]*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckMaxPoolThroughUnpool(t *testing.T) {
	// End-to-end: gradient flows through pool (with indices) and unpool.
	x := seededRandn(47, 1, 1, 4, 4).SetRequiresGrad(true)
	p := NewMaxPool2d(2)
	u := NewMaxUnpool2d(2)
	loss := func() *tensor.Tensor {
		out, idx := p.ForwardWithIndices(x)
		return u.Forward(out, idx).Square().Mean()
	}
	gradCheck(t, "MaxPool+MaxUnpool", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

// ---- FractionalMaxPool ---------------------------------------------------------

func TestFractionalMaxPool2dDeterministicRegions(t *testing.T) {
	// in=5, kernel=2, out=3, u=0.5: alpha = (5-2)/(3-1) = 1.5,
	// start_0 = floor(0.75) - floor(0.75) = 0, start_1 = floor(2.25) - 0 = 2,
	// start_2 = in - kernel = 3 -> starts [0, 2, 3] on both dims. On an
	// ascending plane the max of each 2x2 window is its bottom-right corner.
	x := arange(1, 1, 5, 5)
	p := NewFractionalMaxPool2d(2, WithOutputSize(3), WithFractionalSamples(0.5))
	out := p.Forward(x)
	if !intsEqual(out.Shape, []int{1, 1, 3, 3}) {
		t.Fatalf("shape = %v, want [1 1 3 3]", out.Shape)
	}
	want := []float64{6, 8, 9, 16, 18, 19, 21, 23, 24}
	for i, w := range want {
		if out.Data[i] != w {
			t.Errorf("out[%d] = %g, want %g", i, out.Data[i], w)
		}
	}

	// Injected samples make repeated forwards identical.
	again := p.Forward(x)
	for i := range out.Data {
		if out.Data[i] != again.Data[i] {
			t.Fatalf("deterministic forward diverged at %d", i)
		}
	}
}

func TestFractionalMaxPoolShapes(t *testing.T) {
	// Ratio form: out = floor(in * ratio).
	pr := NewFractionalMaxPool2d(2, WithOutputRatio(0.5))
	x := seededRandn(51, 1, 2, 10, 10)
	out := pr.Forward(x)
	if !intsEqual(out.Shape, []int{1, 2, 5, 5}) {
		t.Fatalf("ratio shape = %v, want [1 2 5 5]", out.Shape)
	}

	// Per-dim ratios.
	pr2 := NewFractionalMaxPool2d(2, WithOutputRatio(0.5, 0.4))
	out = pr2.Forward(x)
	if !intsEqual(out.Shape, []int{1, 2, 5, 4}) {
		t.Fatalf("per-dim ratio shape = %v, want [1 2 5 4]", out.Shape)
	}

	// 3d size form.
	p3 := NewFractionalMaxPool3d(2, WithOutputSize(3), WithFractionalSamples(0.25))
	x3 := seededRandn(52, 1, 1, 6, 6, 6)
	out3 := p3.Forward(x3)
	if !intsEqual(out3.Shape, []int{1, 1, 3, 3, 3}) {
		t.Fatalf("3d shape = %v, want [1 1 3 3 3]", out3.Shape)
	}
}

func TestFractionalMaxPoolValidation(t *testing.T) {
	cases := []func(){
		func() { NewFractionalMaxPool2d(2) },                                                // neither size nor ratio
		func() { NewFractionalMaxPool2d(2, WithOutputSize(3), WithOutputRatio(0.5)) },       // both
		func() { NewFractionalMaxPool2d(2, WithOutputRatio(1.5)) },                          // ratio out of range
		func() { NewFractionalMaxPool2d(2, WithOutputSize(3), WithFractionalSamples(1.5)) }, // sample out of range
	}
	for i, c := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("case %d did not panic", i)
				}
			}()
			c()
		}()
	}
	// Infeasible geometry (kernel + out - 1 > in) panics at Forward.
	func() {
		defer func() {
			if recover() == nil {
				t.Errorf("infeasible geometry did not panic")
			}
		}()
		p := NewFractionalMaxPool2d(3, WithOutputSize(4), WithFractionalSamples(0))
		p.Forward(seededRandn(53, 1, 1, 5, 5))
	}()
}

func TestGradCheckFractionalMaxPool(t *testing.T) {
	// Injected samples make the layer deterministic, so gradcheck is valid.
	p := NewFractionalMaxPool2d(2, WithOutputSize(3), WithFractionalSamples(0.3))
	x := seededRandn(54, 1, 2, 5, 5).SetRequiresGrad(true)
	gradCheck(t, "FractionalMaxPool2d",
		func() *tensor.Tensor { return p.Forward(x).Square().Mean() },
		[]*tensor.Tensor{x}, gcEps, gcTol, 0)
}
