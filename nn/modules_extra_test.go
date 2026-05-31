package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// gradFlows is a local helper: runs loss.Backward() and asserts param.Grad is
// non-nil and has at least one non-zero entry.
func gradFlows(t *testing.T, name string, loss, param *tensor.Tensor) {
	t.Helper()
	loss.Backward()
	if param.Grad == nil {
		t.Fatalf("%s: grad is nil on param", name)
	}
	for _, v := range param.Grad.Data {
		if v != 0 {
			return
		}
	}
	t.Fatalf("%s: all gradients are zero", name)
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// ---- Pooling: 1d / 3d ----

func TestMaxPool1dForward(t *testing.T) {
	// (1,1,4): [1,3,2,5], k=2,s=2 -> [max(1,3), max(2,5)] = [3,5]
	x := tensor.New([]float64{1, 3, 2, 5}, 1, 1, 4)
	y := NewMaxPool1d(2, 2).Forward(x)
	if got := y.Shape; len(got) != 3 || got[0] != 1 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("MaxPool1d shape: got %v", got)
	}
	if !approx(y.Data[0], 3) || !approx(y.Data[1], 5) {
		t.Fatalf("MaxPool1d: got %v want [3 5]", y.Data)
	}
}

func TestAvgPool1dForward(t *testing.T) {
	// (1,1,4): [1,3,2,6], k=2,s=2 -> [2, 4]
	x := tensor.New([]float64{1, 3, 2, 6}, 1, 1, 4)
	y := NewAvgPool1d(2, 2).Forward(x)
	if !approx(y.Data[0], 2) || !approx(y.Data[1], 4) {
		t.Fatalf("AvgPool1d: got %v want [2 4]", y.Data)
	}
}

func TestMaxPool1dGrad(t *testing.T) {
	x := tensor.New([]float64{1, 3, 2, 5}, 1, 1, 4).SetRequiresGrad(true)
	y := NewMaxPool1d(2, 2).Forward(x)
	gradFlows(t, "MaxPool1d", y.Sum(), x)
}

func TestMaxPool3dForward(t *testing.T) {
	// (1,1,2,2,2) full window k=2 -> single max over the 8 values.
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	x := tensor.New(data, 1, 1, 2, 2, 2)
	y := NewMaxPool3d(2, 2).Forward(x)
	if got := y.Shape; len(got) != 5 || got[2] != 1 || got[3] != 1 || got[4] != 1 {
		t.Fatalf("MaxPool3d shape: got %v", got)
	}
	if !approx(y.Data[0], 8) {
		t.Fatalf("MaxPool3d: got %v want 8", y.Data[0])
	}
}

func TestAvgPool3dForward(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8} // mean = 4.5
	x := tensor.New(data, 1, 1, 2, 2, 2)
	y := NewAvgPool3d(2, 2).Forward(x)
	if !approx(y.Data[0], 4.5) {
		t.Fatalf("AvgPool3d: got %v want 4.5", y.Data[0])
	}
}

func TestAvgPool3dGrad(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	x := tensor.New(data, 1, 1, 2, 2, 2).SetRequiresGrad(true)
	y := NewAvgPool3d(2, 2).Forward(x)
	gradFlows(t, "AvgPool3d", y.Sum(), x)
}

// ---- Flatten / Unflatten ----

func TestFlattenForward(t *testing.T) {
	x := tensor.New(make([]float64, 2*3*4), 2, 3, 4)
	y := NewFlatten(1, -1).Forward(x)
	if got := y.Shape; len(got) != 2 || got[0] != 2 || got[1] != 12 {
		t.Fatalf("Flatten shape: got %v want [2 12]", got)
	}
}

func TestUnflattenForward(t *testing.T) {
	x := tensor.New(make([]float64, 2*12), 2, 12)
	y := NewUnflatten(1, 3, 4).Forward(x)
	if got := y.Shape; len(got) != 3 || got[0] != 2 || got[1] != 3 || got[2] != 4 {
		t.Fatalf("Unflatten shape: got %v want [2 3 4]", got)
	}
}

func TestFlattenUnflattenRoundtrip(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 3).SetRequiresGrad(true)
	f := NewFlatten(1, -1).Forward(x)     // (1,6)
	u := NewUnflatten(1, 2, 3).Forward(f) // (1,2,3)
	for i := range x.Data {
		if !approx(u.Data[i], x.Data[i]) {
			t.Fatalf("roundtrip[%d]: got %v want %v", i, u.Data[i], x.Data[i])
		}
	}
	gradFlows(t, "Flatten/Unflatten", u.Sum(), x)
}

// ---- Bilinear ----

func TestBilinearForwardValue(t *testing.T) {
	// in1=2, in2=2, out=1. Set W = identity-ish so y = x1 . (W x2).
	b := NewBilinear(2, 2, 1, false)
	// Override weight to a known value: W[0] = [[1,0],[0,2]]
	copy(b.Weight.Data, []float64{1, 0, 0, 2})
	x1 := tensor.New([]float64{1, 1}, 1, 2)
	x2 := tensor.New([]float64{3, 5}, 1, 2)
	// tmp = x1 @ W = [1*1+1*0, 1*0+1*2] = [1, 2]; prod with x2 = [3, 10]; sum = 13
	y := b.Forward(x1, x2)
	if got := y.Shape; len(got) != 2 || got[0] != 1 || got[1] != 1 {
		t.Fatalf("Bilinear shape: got %v", got)
	}
	if !approx(y.Data[0], 13) {
		t.Fatalf("Bilinear: got %v want 13", y.Data[0])
	}
}

func TestBilinearGrad(t *testing.T) {
	b := NewBilinear(3, 2, 4, true)
	x1 := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	x2 := tensor.New([]float64{1, 1, 2, 2}, 2, 2)
	y := b.Forward(x1, x2)
	if got := y.Shape; got[0] != 2 || got[1] != 4 {
		t.Fatalf("Bilinear batch shape: got %v want [2 4]", got)
	}
	loss := y.Sum()
	loss.Backward()
	if b.Weight.Grad == nil {
		t.Fatalf("Bilinear: weight grad nil")
	}
	nz := false
	for _, v := range b.Weight.Grad.Data {
		if v != 0 {
			nz = true
			break
		}
	}
	if !nz {
		t.Fatalf("Bilinear: weight grad all zero")
	}
	if b.Bias.Grad == nil {
		t.Fatalf("Bilinear: bias grad nil")
	}
}

// ---- Softmin / Softmax2d ----

func TestSoftminForward(t *testing.T) {
	// Softmin should put the most mass on the smallest element.
	x := tensor.New([]float64{1, 2, 3}, 1, 3)
	y := Softmin{Axis: 1}.Forward(x)
	var sum float64
	for _, v := range y.Data {
		sum += v
	}
	if !approx(sum, 1) {
		t.Fatalf("Softmin: sum got %v want 1", sum)
	}
	if !(y.Data[0] > y.Data[1] && y.Data[1] > y.Data[2]) {
		t.Fatalf("Softmin: expected decreasing weights, got %v", y.Data)
	}
}

func TestSoftmax2dForward(t *testing.T) {
	// (1,2,1,1): channel softmax over [0,0] -> [0.5, 0.5]
	x := tensor.New([]float64{0, 0}, 1, 2, 1, 1)
	y := Softmax2d{}.Forward(x)
	if !approx(y.Data[0], 0.5) || !approx(y.Data[1], 0.5) {
		t.Fatalf("Softmax2d: got %v want [0.5 0.5]", y.Data)
	}
}

// ---- CosineSimilarity / PairwiseDistance ----

func TestCosineSimilarityForward(t *testing.T) {
	// Identical vectors -> cos = 1; orthogonal -> 0.
	x1 := tensor.New([]float64{1, 0, 0, 1}, 2, 2)
	x2 := tensor.New([]float64{1, 0, 1, 0}, 2, 2)
	y := NewCosineSimilarity(1, 1e-8).Forward(x1, x2)
	if y.Shape[0] != 2 {
		t.Fatalf("CosineSimilarity shape: got %v", y.Shape)
	}
	if !(math.Abs(y.Data[0]-1) < 1e-6) {
		t.Fatalf("CosineSimilarity row0: got %v want 1", y.Data[0])
	}
	if !(math.Abs(y.Data[1]) < 1e-6) {
		t.Fatalf("CosineSimilarity row1: got %v want 0", y.Data[1])
	}
}

func TestPairwiseDistanceForward(t *testing.T) {
	// L2 distance between [0,0] and [3,4] = 5 (eps negligible).
	x1 := tensor.New([]float64{0, 0}, 1, 2)
	x2 := tensor.New([]float64{3, 4}, 1, 2)
	y := NewPairwiseDistance(2, 0).Forward(x1, x2)
	if math.Abs(y.Data[0]-5) > 1e-3 {
		t.Fatalf("PairwiseDistance: got %v want ~5", y.Data[0])
	}
}

func TestPairwiseDistanceGrad(t *testing.T) {
	x1 := tensor.New([]float64{0, 0}, 1, 2).SetRequiresGrad(true)
	x2 := tensor.New([]float64{3, 4}, 1, 2)
	y := NewPairwiseDistance(2, 1e-6).Forward(x1, x2)
	gradFlows(t, "PairwiseDistance", y.Sum(), x1)
}

// ---- Identity ----

func TestIdentityForward(t *testing.T) {
	x := tensor.New([]float64{1, 2, 3}, 3)
	y := Identity{}.Forward(x)
	for i := range x.Data {
		if y.Data[i] != x.Data[i] {
			t.Fatalf("Identity changed data at %d", i)
		}
	}
}
