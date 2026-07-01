package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// dataClose reports whether a and b match elementwise within tol.
func dataClose(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func TestBatchNorm3dMatchesBatchNorm1dOnFlattenedSpatial(t *testing.T) {
	// BatchNorm3d over (N, C, D, H, W) must equal BatchNorm1d over
	// (N, C, D*H*W): both normalize per channel over everything else.
	x := seededRandn(101, 2, 3, 2, 2, 2)
	b3 := NewBatchNorm3d(3)
	b1 := NewBatchNorm1d(3)
	y3 := b3.Forward(x)
	y1 := b1.Forward(x.Reshape(2, 3, 8))
	if !shapeEq(y3.Shape, []int{2, 3, 2, 2, 2}) {
		t.Fatalf("BatchNorm3d shape: got %v", y3.Shape)
	}
	if !dataClose(y3.Data, y1.Data, 1e-12) {
		t.Fatalf("BatchNorm3d output differs from BatchNorm1d on flattened input")
	}
	if !dataClose(b3.RunMean.Data, b1.RunMean.Data, 1e-12) ||
		!dataClose(b3.RunVar.Data, b1.RunVar.Data, 1e-12) {
		t.Fatalf("BatchNorm3d running stats differ from BatchNorm1d")
	}
}

func TestBatchNorm3dEvalUsesRunningStats(t *testing.T) {
	bn := NewBatchNorm3d(2)
	bn.Eval()
	// Fresh running stats are mean 0 / var 1, so eval output is
	// x / sqrt(1 + eps) exactly (weight 1, bias 0).
	x := seededRandn(102, 1, 2, 2, 2, 2)
	y := bn.Forward(x)
	scale := 1.0 / math.Sqrt(1+1e-5)
	for i := range x.Data {
		if math.Abs(y.Data[i]-x.Data[i]*scale) > 1e-12 {
			t.Fatalf("eval output[%d]=%g, want %g", i, y.Data[i], x.Data[i]*scale)
		}
	}
}

func TestBatchNorm3dRejectsWrongRank(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("BatchNorm3d: expected panic on 4D input")
		}
	}()
	NewBatchNorm3d(2).Forward(tensor.Zeros(1, 2, 3, 3))
}

func TestGradCheckBatchNorm3d(t *testing.T) {
	bn := NewBatchNorm3d(2)
	x := seededRandn(103, 2, 2, 2, 3, 2).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return bn.Forward(x).Square().Mean() }
	gradCheck(t, "BatchNorm3d", loss, append(bn.Parameters(), x), gcEps, gcTol, 40)
}

func TestInstanceNorm3dPerInstanceStats(t *testing.T) {
	in := NewInstanceNorm3d(3)
	x := seededRandn(104, 2, 3, 2, 3, 2)
	y := in.Forward(x)
	if !shapeEq(y.Shape, x.Shape) {
		t.Fatalf("InstanceNorm3d shape: got %v", y.Shape)
	}
	// Every (n, c) slice must have mean ~0 and biased variance ~1.
	N, C, S := 2, 3, 12
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			mean, sq := 0.0, 0.0
			for s := 0; s < S; s++ {
				v := y.Data[(n*C+c)*S+s]
				mean += v
				sq += v * v
			}
			mean /= float64(S)
			variance := sq/float64(S) - mean*mean
			if math.Abs(mean) > 1e-9 {
				t.Errorf("slice (%d,%d): mean=%g, want ~0", n, c, mean)
			}
			if math.Abs(variance-1) > 1e-3 {
				t.Errorf("slice (%d,%d): var=%g, want ~1", n, c, variance)
			}
		}
	}
}

func TestGradCheckInstanceNorm3d(t *testing.T) {
	in := NewInstanceNorm3d(2, WithAffine(true))
	x := seededRandn(105, 2, 2, 2, 2, 3).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return in.Forward(x).Square().Mean() }
	gradCheck(t, "InstanceNorm3d", loss, append(in.Parameters(), x), gcEps, gcTol, 40)
}
