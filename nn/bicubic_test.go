package nn

import (
	"fmt"
	"math"
	"testing"

	"gonn/tensor"
)

// TestBicubicRowSumsToOne checks the partition-of-unity property of the Keys
// kernel: the four per-axis weights sum to 1 at every output position even
// when border clamping folds taps together, so every row of the assembled
// (outH*outW, H*W) gather matrix sums to 1 (the 16 separable products
// factorize as (sum_y)(sum_x) = 1). Verified directly on cubicWeights and
// end-to-end via constant-input invariance.
func TestBicubicRowSumsToOne(t *testing.T) {
	for _, ac := range []bool{false, true} {
		for _, tc := range []struct{ in, r int }{{2, 2}, {3, 2}, {4, 3}, {5, 2}, {7, 4}} {
			out := tc.in * tc.r
			_, w := cubicWeights(tc.in, out, tc.r, ac)
			for o := 0; o < out; o++ {
				sum := w[o][0] + w[o][1] + w[o][2] + w[o][3]
				if math.Abs(sum-1) > 1e-12 {
					t.Errorf("cubicWeights(in=%d, r=%d, align=%v): weights at o=%d sum to %.17g, want 1",
						tc.in, tc.r, ac, o, sum)
				}
			}
		}
		// End-to-end: a constant image must stay constant under upsampling.
		for _, hw := range [][2]int{{2, 3}, {4, 4}, {5, 3}} {
			u := NewUpsample(2, "bicubic", WithAlignCorners(ac))
			y := u.Forward(tensor.Full(2.5, 1, 2, hw[0], hw[1]))
			if !shapeEq(y.Shape, []int{1, 2, 2 * hw[0], 2 * hw[1]}) {
				t.Fatalf("bicubic(align=%v) %dx%d shape: got %v", ac, hw[0], hw[1], y.Shape)
			}
			for i, v := range y.Data {
				if math.Abs(v-2.5) > 1e-12 {
					t.Fatalf("bicubic(align=%v) constant %dx%d: y[%d]=%g, want 2.5", ac, hw[0], hw[1], i, v)
				}
			}
		}
	}
}

// TestBicubicMatchesPyTorch2x2 pins the full 4x4 output of
//
//	F.interpolate([[1,2],[3,4]], scale_factor=2, mode='bicubic',
//	              align_corners=False)
//
// against torch 2.7.1 (float64). All 16 values are exact binary fractions
// (the a=-0.75 weights for scale 2 are multiples of 1/256), e.g. the corner:
// the 1D collapsed border weights are [283, -27]/256, so
// y(0,0) = (283^2*1 + 283*(-27)*(2+3) + 27^2*4)/65536 = 44800/65536.
func TestBicubicMatchesPyTorch2x2(t *testing.T) {
	u := NewUpsample(2, "bicubic")
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	y := u.Forward(x)
	want := []float64{
		0.68359375, 1.015625, 1.5625, 1.89453125,
		1.34765625, 1.6796875, 2.2265625, 2.55859375,
		2.44140625, 2.7734375, 3.3203125, 3.65234375,
		3.10546875, 3.4375, 3.984375, 4.31640625,
	}
	if !shapeEq(y.Shape, []int{1, 1, 4, 4}) {
		t.Fatalf("bicubic 2x2 shape: got %v", y.Shape)
	}
	if !dataClose(y.Data, want, 1e-12) {
		t.Fatalf("bicubic 2x2: got %v, want %v", y.Data, want)
	}
}

// TestBicubicMatchesPyTorch3x3AlignCorners pins align_corners=true numerics
// (source coords o*(in-1)/(out-1), here o*2/5) against torch 2.7.1 float64
// output for a non-separable input with a cross term. Corners must coincide
// with the input corners exactly.
func TestBicubicMatchesPyTorch3x3AlignCorners(t *testing.T) {
	// Input: g(k) = 0.5*k^2 - k for k = 3*i + j.
	x := tensor.New([]float64{0, -0.5, 0, 1.5, 4, 7.5, 12, 17.5, 24}, 1, 1, 3, 3)
	u := NewUpsample(2, "bicubic", WithAlignCorners(true))
	y := u.Forward(x)
	want := []float64{
		0, -0.23, -0.46, -0.46, -0.23, 0,
		-0.174, -0.104432, 0.056144, 0.571856, 1.192432, 1.722,
		0.228, 0.688144, 1.357952, 2.546048, 3.675856, 4.596,
		3.492, 4.467856, 5.810048, 7.885952, 9.688144, 11.124,
		8.034, 9.400432, 11.251856, 14.000144, 16.311568, 18.138,
		12, 13.666, 15.908, 19.172, 21.874, 24,
	}
	if !dataClose(y.Data, want, 1e-12) {
		t.Fatalf("bicubic 3x3 align_corners: got %v, want %v", y.Data, want)
	}
	// align_corners=true preserves the four corner samples bit-exactly.
	for _, c := range [][2]int{{0, 0}, {0, 5}, {5, 0}, {5, 5}} {
		got := y.Data[c[0]*6+c[1]]
		wantC := x.Data[(c[0]/5*2)*3+(c[1]/5*2)]
		if got != wantC {
			t.Errorf("bicubic align_corners corner (%d,%d): got %g, want %g", c[0], c[1], got, wantC)
		}
	}
}

// TestBicubicInteriorSpotCheck upsamples the 4x4 ramp x[i][j] = 4i + j by 2
// (align_corners=false) and verifies two fully-interior outputs against
// hand-computed 16-tap sums.
//
// Output (3,3): sy = sx = 3/2 - 0.25 = 1.25, floor = 1, frac = 0.25, taps
// {0,1,2,3} per axis (no clamping). The 1D weights [W(1.25), W(0.25),
// W(0.75), W(1.75)] = [-27, 225, 67, -9]/256 (sum 256/256 = 1). Because the
// ramp is affine (4*row + col), the 16-tap double sum collapses to
// 4*T + T = 5*T with the weighted tap position
// T = (0*(-27) + 1*225 + 2*67 + 3*(-9))/256 = 332/256 = 1.296875, giving
// y(3,3) = 6.484375.
//
// Output (4,4): s = 1.75, frac = 0.75, weights [-9, 67, 225, -27]/256,
// T' = (0*(-9) + 1*67 + 2*225 + 3*(-27))/256 = 436/256 = 1.703125, giving
// y(4,4) = 5*T' = 8.515625. Both match torch 2.7.1 exactly.
func TestBicubicInteriorSpotCheck(t *testing.T) {
	data := make([]float64, 16)
	for i := range data {
		data[i] = float64(i)
	}
	u := NewUpsample(2, "bicubic")
	y := u.Forward(tensor.New(data, 1, 1, 4, 4))
	if !shapeEq(y.Shape, []int{1, 1, 8, 8}) {
		t.Fatalf("bicubic ramp shape: got %v", y.Shape)
	}
	if got := y.Data[3*8+3]; math.Abs(got-6.484375) > 1e-12 {
		t.Errorf("bicubic ramp y(3,3): got %.17g, want 6.484375", got)
	}
	if got := y.Data[4*8+4]; math.Abs(got-8.515625) > 1e-12 {
		t.Errorf("bicubic ramp y(4,4): got %.17g, want 8.515625", got)
	}
}

// bicubicMoments returns the first and second moments M1 = sum_k w_k p_k and
// M2 = sum_k w_k p_k^2 of the four Keys a=-0.75 weights about the source
// coordinate s (tap offsets p_k = tap_k - s), assuming no clamping. Summing
// the kernel branches analytically gives, with f = s - floor(s) and
// e = f(1-f):
//
//	M1 = -(2a+1) e (1-2f)      M2 = 4(2a+1) e^2
//
// Both vanish only for Keys' third-order choice a = -1/2 (Catmull-Rom);
// PyTorch's a = -3/4 is NOT exact on linear or quadratic inputs, but the
// Taylor expansion sum_k w_k f(s+p_k) = f(s) + M1 f'(s) + (M2/2) f”(s) is
// EXACT for any quadratic f, so interior outputs are pinned in closed form.
func bicubicMoments(s float64) (m1, m2 float64) {
	const a = -0.75
	f := s - math.Floor(s)
	e := f * (1 - f)
	return -(2*a + 1) * e * (1 - 2*f), 4 * (2*a + 1) * e * e
}

// TestBicubicQuadraticClosedForm samples the quadratic surface
//
//	F(x,y) = 2 + 0.5x - y + 0.25x^2 - 0.2y^2 + 0.3xy
//
// on an 8x8 grid, upsamples x2, and checks every fully-interior output cell
// (all four taps unclamped on both axes) against the closed-form separable
// interpolant built from the exact kernel moments:
//
//	I[1](s) = 1,  I[x](s) = s + M1,  I[x^2](s) = s^2 + 2 s M1 + M2
//
// and, since the 2D operator is the tensor product of the 1D operators,
// I[xy](sx,sy) = I[x](sx) * I[y](sy). Verified to agree with torch 2.7.1 to
// ~1e-14 for both align_corners settings.
func TestBicubicQuadraticClosedForm(t *testing.T) {
	const in = 8
	const r = 2
	const out = in * r
	F := func(x, y float64) float64 {
		return 2 + 0.5*x - y + 0.25*x*x - 0.2*y*y + 0.3*x*y
	}
	data := make([]float64, in*in)
	for i := 0; i < in; i++ {
		for j := 0; j < in; j++ {
			data[i*in+j] = F(float64(j), float64(i))
		}
	}
	x := tensor.New(data, 1, 1, in, in)

	for _, ac := range []bool{false, true} {
		u := NewUpsample(r, "bicubic", WithAlignCorners(ac))
		y := u.Forward(x)
		checked := 0
		for oy := 0; oy < out; oy++ {
			sy := srcCoord(oy, in, out, r, ac)
			ly := int(math.Floor(sy))
			if ly < 1 || ly+2 > in-1 {
				continue // taps would clamp on the y axis
			}
			m1y, m2y := bicubicMoments(sy)
			iy1 := sy + m1y
			iy2 := sy*sy + 2*sy*m1y + m2y
			for ox := 0; ox < out; ox++ {
				sx := srcCoord(ox, in, out, r, ac)
				lx := int(math.Floor(sx))
				if lx < 1 || lx+2 > in-1 {
					continue
				}
				m1x, m2x := bicubicMoments(sx)
				ix1 := sx + m1x
				ix2 := sx*sx + 2*sx*m1x + m2x
				want := 2 + 0.5*ix1 - iy1 + 0.25*ix2 - 0.2*iy2 + 0.3*ix1*iy1
				got := y.Data[oy*out+ox]
				if math.Abs(got-want) > 1e-10 {
					t.Errorf("bicubic quadratic (align=%v) out(%d,%d): got %.15g, want %.15g",
						ac, oy, ox, got, want)
				}
				checked++
			}
		}
		if checked != 100 { // 10x10 interior cells for in=8, r=2
			t.Fatalf("bicubic quadratic (align=%v): checked %d interior cells, want 100", ac, checked)
		}
	}
}

func TestGradCheckUpsampleBicubic(t *testing.T) {
	for _, ac := range []bool{false, true} {
		u := NewUpsample(2, "bicubic", WithAlignCorners(ac))
		x := seededRandn(150, 1, 2, 3, 3).SetRequiresGrad(true)
		loss := func() *tensor.Tensor { return u.Forward(x).Square().Mean() }
		gradCheck(t, fmt.Sprintf("UpsampleBicubic(align=%v)", ac), loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
	}
}

// TestBicubicRankMismatchPanics: like PyTorch, mode "bicubic" only supports
// 4D (N, C, H, W) input.
func TestBicubicRankMismatchPanics(t *testing.T) {
	for _, shape := range [][]int{{1, 1, 4}, {1, 1, 2, 2, 2}} {
		shape := shape
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("bicubic: expected panic for %dD input", len(shape))
				}
			}()
			NewUpsample(2, "bicubic").Forward(tensor.Zeros(shape...))
		}()
	}
}
