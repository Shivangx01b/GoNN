package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// Tests for the N-d padding catalogue (padding_nd.go): hand-computed value
// tests per mode/rank, gradchecks (one per mode — grads must accumulate when
// several output cells read the same input cell), shape tests, and the
// PyTorch pad-size constraints.

func checkVals(t *testing.T, name string, got *tensor.Tensor, wantShape []int, want []float64) {
	t.Helper()
	if !shapeEq(got.Shape, wantShape) {
		t.Fatalf("%s: shape got %v want %v", name, got.Shape, wantShape)
	}
	if len(got.Data) != len(want) {
		t.Fatalf("%s: numel got %d want %d", name, len(got.Data), len(want))
	}
	for i := range want {
		if math.Abs(got.Data[i]-want[i]) > 1e-12 {
			t.Fatalf("%s: data[%d] got %v want %v\n  got:  %v\n  want: %v",
				name, i, got.Data[i], want[i], got.Data, want)
		}
	}
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	f()
}

// --- 1d value tests -------------------------------------------------------

func TestZeroPad1dValues(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3}, 1, 1, 3)
	y := NewZeroPad1d(1, 2).Forward(x)
	checkVals(t, "ZeroPad1d(1,2)", y, []int{1, 1, 6}, []float64{0, 1, 2, 3, 0, 0})
}

func TestConstantPad1dValues(t *testing.T) {
	x := tensor.New([]float64{1, 2}, 1, 1, 2)
	y := NewConstantPad1d(2, 1, 9).Forward(x)
	checkVals(t, "ConstantPad1d(2,1,9)", y, []int{1, 1, 5}, []float64{9, 9, 1, 2, 9})
}

func TestReflectionPad1dValues(t *testing.T) {
	// PyTorch doc example: ReflectionPad1d((3, 1)) of [0 1 2 3]
	// -> [3 2 1 0 1 2 3 2].
	x := tensor.New([]float64{0, 1, 2, 3}, 1, 1, 4)
	y := NewReflectionPad1d(3, 1).Forward(x)
	checkVals(t, "ReflectionPad1d(3,1)", y, []int{1, 1, 8},
		[]float64{3, 2, 1, 0, 1, 2, 3, 2})
}

func TestReplicationPad1dValues(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3}, 1, 1, 3)
	y := NewReplicationPad1d(2, 1).Forward(x)
	checkVals(t, "ReplicationPad1d(2,1)", y, []int{1, 1, 6},
		[]float64{1, 1, 1, 2, 3, 3})

	// Replication allows padding larger than the input.
	x2 := tensor.New([]float64{7, 9}, 1, 1, 2)
	y2 := NewReplicationPad1d(3, 5).Forward(x2)
	checkVals(t, "ReplicationPad1d(3,5)", y2, []int{1, 1, 10},
		[]float64{7, 7, 7, 7, 9, 9, 9, 9, 9, 9})
}

func TestCircularPad1dValues(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3}, 1, 1, 3)
	y := NewCircularPad1d(2, 1).Forward(x)
	checkVals(t, "CircularPad1d(2,1)", y, []int{1, 1, 6},
		[]float64{2, 3, 1, 2, 3, 1})

	// PyTorch doc example: CircularPad1d(2) of [0 1 2 3]
	// -> [2 3 0 1 2 3 0 1].
	x2 := tensor.New([]float64{0, 1, 2, 3}, 1, 1, 4)
	y2 := NewCircularPad1d(2, 2).Forward(x2)
	checkVals(t, "CircularPad1d(2,2)", y2, []int{1, 1, 8},
		[]float64{2, 3, 0, 1, 2, 3, 0, 1})
}

func TestPad1dPerChannel(t *testing.T) {
	// Padding is applied per (N, C) plane independently.
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 2, 2)
	y := NewZeroPad1d(1, 0).Forward(x)
	checkVals(t, "ZeroPad1d per-channel", y, []int{1, 2, 3},
		[]float64{0, 1, 2, 0, 3, 4})
}

// --- 2d value tests (CircularPad2d is new; others are golden-pinned) -------

func TestCircularPad2dValues(t *testing.T) {
	// 3x3 arange, circular pad 1 on every side.
	x := tensor.New([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, 1, 1, 3, 3)
	y := NewCircularPad2d(1, 1, 1, 1).Forward(x)
	checkVals(t, "CircularPad2d(1,1,1,1)", y, []int{1, 1, 5, 5}, []float64{
		8, 6, 7, 8, 6,
		2, 0, 1, 2, 0,
		5, 3, 4, 5, 3,
		8, 6, 7, 8, 6,
		2, 0, 1, 2, 0,
	})

	// Asymmetric: left=2, right=0, top=0, bottom=1 on a 2x2 input.
	x2 := tensor.New([]float64{0, 1, 2, 3}, 1, 1, 2, 2)
	y2 := NewCircularPad2d(2, 0, 0, 1).Forward(x2)
	checkVals(t, "CircularPad2d(2,0,0,1)", y2, []int{1, 1, 3, 4}, []float64{
		0, 1, 0, 1,
		2, 3, 2, 3,
		0, 1, 0, 1,
	})
}

// --- 3d value tests ---------------------------------------------------------

func TestZeroPad3dValues(t *testing.T) {
	// (1,1,1,1,2): left=1 (W), back=1 (D).
	x := tensor.New([]float64{1, 2}, 1, 1, 1, 1, 2)
	y := NewZeroPad3d(1, 0, 0, 0, 0, 1).Forward(x)
	checkVals(t, "ZeroPad3d(1,0,0,0,0,1)", y, []int{1, 1, 2, 1, 3},
		[]float64{0, 1, 2, 0, 0, 0})
}

func TestConstantPad3dValues(t *testing.T) {
	// (1,1,1,1,2): right=1 (W), top=1 (H), value 9.
	x := tensor.New([]float64{1, 2}, 1, 1, 1, 1, 2)
	y := NewConstantPad3d(0, 1, 1, 0, 0, 0, 9).Forward(x)
	checkVals(t, "ConstantPad3d(0,1,1,0,0,0,9)", y, []int{1, 1, 1, 2, 3},
		[]float64{9, 9, 9, 1, 2, 9})
}

func TestReflectionPad3dValues(t *testing.T) {
	// (1,1,1,2,2): left=1 (W), bottom=1 (H). Reflection of a size-2 dim with
	// pad 1 mirrors the opposite element.
	x := tensor.New([]float64{0, 1, 2, 3}, 1, 1, 1, 2, 2)
	y := NewReflectionPad3d(1, 0, 0, 1, 0, 0).Forward(x)
	checkVals(t, "ReflectionPad3d(1,0,0,1,0,0)", y, []int{1, 1, 1, 3, 3},
		[]float64{1, 0, 1, 3, 2, 3, 1, 0, 1})
}

func TestReplicationPad3dValues(t *testing.T) {
	// (1,1,2,1,2): right=1 (W), front=1 (D).
	x := tensor.New([]float64{0, 1, 2, 3}, 1, 1, 2, 1, 2)
	y := NewReplicationPad3d(0, 1, 0, 0, 1, 0).Forward(x)
	checkVals(t, "ReplicationPad3d(0,1,0,0,1,0)", y, []int{1, 1, 3, 1, 3},
		[]float64{0, 1, 1, 0, 1, 1, 2, 3, 3})
}

func TestCircularPad3dValues(t *testing.T) {
	// (1,1,2,2,1): top=1 (H), back=1 (D).
	x := tensor.New([]float64{0, 1, 2, 3}, 1, 1, 2, 2, 1)
	y := NewCircularPad3d(0, 0, 1, 0, 0, 1).Forward(x)
	checkVals(t, "CircularPad3d(0,0,1,0,0,1)", y, []int{1, 1, 3, 3, 1},
		[]float64{1, 0, 1, 3, 2, 3, 1, 0, 1})
}

// --- shape tests ------------------------------------------------------------

func TestPadNdShapes(t *testing.T) {
	y := NewZeroPad3d(1, 2, 3, 4, 5, 6).Forward(tensor.Randn(2, 3, 4, 5, 6))
	if !shapeEq(y.Shape, []int{2, 3, 15, 12, 9}) {
		t.Fatalf("ZeroPad3d shape: got %v", y.Shape)
	}
	y = NewConstantPad3d(2, 1, 0, 3, 1, 1, -1).Forward(tensor.Randn(1, 2, 2, 3, 4))
	if !shapeEq(y.Shape, []int{1, 2, 4, 6, 7}) {
		t.Fatalf("ConstantPad3d shape: got %v", y.Shape)
	}
	y = NewReflectionPad3d(1, 1, 2, 2, 1, 1).Forward(tensor.Randn(1, 2, 3, 4, 5))
	if !shapeEq(y.Shape, []int{1, 2, 5, 8, 7}) {
		t.Fatalf("ReflectionPad3d shape: got %v", y.Shape)
	}
	y = NewReplicationPad3d(1, 2, 3, 4, 5, 6).Forward(tensor.Randn(1, 1, 2, 2, 2))
	if !shapeEq(y.Shape, []int{1, 1, 13, 9, 5}) {
		t.Fatalf("ReplicationPad3d shape: got %v", y.Shape)
	}
	// Circular allows pad == input size.
	y = NewCircularPad1d(4, 4).Forward(tensor.Randn(2, 3, 4))
	if !shapeEq(y.Shape, []int{2, 3, 12}) {
		t.Fatalf("CircularPad1d shape: got %v", y.Shape)
	}
	y = NewCircularPad3d(1, 1, 2, 2, 1, 0).Forward(tensor.Randn(2, 1, 3, 2, 4))
	if !shapeEq(y.Shape, []int{2, 1, 4, 6, 6}) {
		t.Fatalf("CircularPad3d shape: got %v", y.Shape)
	}
	y = NewConstantPad1d(3, 0, 0.5).Forward(tensor.Randn(2, 4, 5))
	if !shapeEq(y.Shape, []int{2, 4, 8}) {
		t.Fatalf("ConstantPad1d shape: got %v", y.Shape)
	}
}

// --- constraint / rank panics ------------------------------------------------

func TestPadNdConstraints(t *testing.T) {
	mustPanic(t, "ReflectionPad1d pad==size", func() {
		NewReflectionPad1d(3, 0).Forward(tensor.Randn(1, 1, 3))
	})
	mustPanic(t, "ReflectionPad3d pad==size", func() {
		NewReflectionPad3d(0, 0, 4, 0, 0, 0).Forward(tensor.Randn(1, 1, 2, 4, 3))
	})
	mustPanic(t, "CircularPad1d pad>size", func() {
		NewCircularPad1d(4, 0).Forward(tensor.Randn(1, 1, 3))
	})
	mustPanic(t, "CircularPad2d pad>size", func() {
		NewCircularPad2d(0, 0, 3, 0).Forward(tensor.Randn(1, 1, 2, 4))
	})
	mustPanic(t, "CircularPad3d pad>size", func() {
		NewCircularPad3d(0, 0, 0, 0, 3, 0).Forward(tensor.Randn(1, 1, 2, 4, 4))
	})
	mustPanic(t, "ZeroPad1d wrong rank", func() {
		NewZeroPad1d(1, 1).Forward(tensor.Randn(1, 1, 2, 2))
	})
	mustPanic(t, "ZeroPad3d wrong rank", func() {
		NewZeroPad3d(1, 1, 1, 1, 1, 1).Forward(tensor.Randn(1, 1, 2, 2))
	})

	// pad == size-1 is the largest legal reflection pad; pad == size is the
	// largest legal circular pad. Both must not panic.
	NewReflectionPad1d(2, 2).Forward(tensor.Randn(1, 1, 3))
	NewCircularPad1d(3, 3).Forward(tensor.Randn(1, 1, 3))
}

// --- gradchecks (one per mode) ------------------------------------------------
//
// Replicate/reflect/circular map several output cells onto the same input
// cell, so the input gradient must accumulate contributions; the finite-
// difference check catches any scatter mistake.

func TestGradCheckZeroPad3d(t *testing.T) {
	p := NewZeroPad3d(1, 0, 0, 1, 1, 0)
	x := seededRandn(31, 2, 2, 2, 3, 2).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "ZeroPad3d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckConstantPad1d(t *testing.T) {
	p := NewConstantPad1d(2, 1, 1.5)
	x := seededRandn(32, 2, 3, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "ConstantPad1d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckReflectionPad1d(t *testing.T) {
	p := NewReflectionPad1d(3, 2)
	x := seededRandn(33, 2, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "ReflectionPad1d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckReplicationPad3d(t *testing.T) {
	p := NewReplicationPad3d(1, 2, 1, 0, 0, 1)
	x := seededRandn(34, 1, 2, 2, 2, 2).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "ReplicationPad3d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckCircularPad2d(t *testing.T) {
	p := NewCircularPad2d(2, 1, 1, 2)
	x := seededRandn(35, 2, 1, 2, 3).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "CircularPad2d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}
