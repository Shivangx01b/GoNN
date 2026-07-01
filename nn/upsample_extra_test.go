package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

func TestUpsampleNearest1d(t *testing.T) {
	u := NewUpsample(2, "nearest")
	x := tensor.New([]float64{1, 2, 3}, 1, 1, 3)
	y := u.Forward(x)
	if !shapeEq(y.Shape, []int{1, 1, 6}) {
		t.Fatalf("nearest 1d shape: got %v", y.Shape)
	}
	want := []float64{1, 1, 2, 2, 3, 3}
	if !dataClose(y.Data, want, 0) {
		t.Fatalf("nearest 1d: got %v, want %v", y.Data, want)
	}
}

func TestUpsampleLinear1d(t *testing.T) {
	// torch.nn.functional.interpolate([1, 2], scale_factor=2, mode='linear',
	// align_corners=False) -> [1.0, 1.25, 1.75, 2.0].
	u := NewUpsample(2, "linear")
	x := tensor.New([]float64{1, 2}, 1, 1, 2)
	y := u.Forward(x)
	want := []float64{1, 1.25, 1.75, 2}
	if !dataClose(y.Data, want, 1e-12) {
		t.Fatalf("linear 1d: got %v, want %v", y.Data, want)
	}
}

func TestUpsampleLinear1dAlignCorners(t *testing.T) {
	// align_corners=True: src = o*(in-1)/(out-1) -> [1, 4/3, 5/3, 2].
	u := NewUpsample(2, "linear", WithAlignCorners(true))
	x := tensor.New([]float64{1, 2}, 1, 1, 2)
	y := u.Forward(x)
	want := []float64{1, 4.0 / 3, 5.0 / 3, 2}
	if !dataClose(y.Data, want, 1e-12) {
		t.Fatalf("linear 1d align_corners: got %v, want %v", y.Data, want)
	}
}

func TestUpsampleNearest3d(t *testing.T) {
	u := NewUpsample(2, "nearest")
	x := tensor.New([]float64{1, 2}, 1, 1, 2, 1, 1)
	y := u.Forward(x)
	if !shapeEq(y.Shape, []int{1, 1, 4, 2, 2}) {
		t.Fatalf("nearest 3d shape: got %v", y.Shape)
	}
	// Depth doubles first: two 2x2 planes of 1s, then two 2x2 planes of 2s.
	want := []float64{1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}
	if !dataClose(y.Data, want, 0) {
		t.Fatalf("nearest 3d: got %v, want %v", y.Data, want)
	}
}

func TestUpsampleTrilinearConstant(t *testing.T) {
	// Interpolation weights sum to 1, so constant input stays constant under
	// both align_corners settings.
	for _, ac := range []bool{false, true} {
		u := NewUpsample(2, "trilinear", WithAlignCorners(ac))
		x := tensor.Full(3.5, 1, 2, 2, 2, 2)
		y := u.Forward(x)
		if !shapeEq(y.Shape, []int{1, 2, 4, 4, 4}) {
			t.Fatalf("trilinear shape: got %v", y.Shape)
		}
		for i, v := range y.Data {
			if math.Abs(v-3.5) > 1e-12 {
				t.Fatalf("trilinear(align=%v)[%d]=%g, want 3.5", ac, i, v)
			}
		}
	}
}

func TestUpsamplingBilinear2dAlignCorners(t *testing.T) {
	// UpsamplingBilinear2d == Upsample(bilinear, align_corners=True).
	// For x = [[1,2],[3,4]] the surface is f(y,x) = 1 + x + 2y sampled at
	// {0, 1/3, 2/3, 1} per axis.
	u := NewUpsamplingBilinear2d(2)
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	y := u.Forward(x)
	want := []float64{
		1, 4.0 / 3, 5.0 / 3, 2,
		5.0 / 3, 2, 7.0 / 3, 8.0 / 3,
		7.0 / 3, 8.0 / 3, 3, 10.0 / 3,
		3, 10.0 / 3, 11.0 / 3, 4,
	}
	if !dataClose(y.Data, want, 1e-12) {
		t.Fatalf("UpsamplingBilinear2d: got %v, want %v", y.Data, want)
	}
}

func TestUpsamplingNearest2dAlias(t *testing.T) {
	x := seededRandn(140, 1, 2, 3, 3)
	ya := NewUpsamplingNearest2d(2).Forward(x)
	yb := NewUpsample(2, "nearest").Forward(x)
	if !dataClose(ya.Data, yb.Data, 0) {
		t.Fatalf("UpsamplingNearest2d differs from Upsample(nearest)")
	}
}

func TestGradCheckUpsampleTrilinear(t *testing.T) {
	u := NewUpsample(2, "trilinear")
	x := seededRandn(141, 1, 2, 2, 2, 2).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return u.Forward(x).Square().Mean() }
	gradCheck(t, "UpsampleTrilinear", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckUpsampleLinearAlignCorners(t *testing.T) {
	u := NewUpsample(3, "linear", WithAlignCorners(true))
	x := seededRandn(142, 2, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return u.Forward(x).Square().Mean() }
	gradCheck(t, "UpsampleLinearAlign", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestUpsampleModeRankMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("Upsample: expected panic for mode 'linear' on 4D input")
		}
	}()
	NewUpsample(2, "linear").Forward(tensor.Zeros(1, 1, 2, 2))
}
