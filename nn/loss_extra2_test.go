package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// ---------------------------------------------------------------- helpers --

// logSoftmaxRows computes a row-wise log-softmax of a (N, C) matrix in plain
// Go, for hand-computed reference values.
func logSoftmaxRows(data []float64, n, c int) []float64 {
	out := make([]float64, n*c)
	for i := 0; i < n; i++ {
		row := data[i*c : (i+1)*c]
		mx := math.Inf(-1)
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(v - mx)
		}
		ls := mx + math.Log(sum)
		for j, v := range row {
			out[i*c+j] = v - ls
		}
	}
	return out
}

func approxEq(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.12g want %.12g (diff %.3g)", name, got, want, math.Abs(got-want))
	}
}

// ---------------------------------------------------------- SoftMarginLoss --

func TestSoftMarginLossValue(t *testing.T) {
	x := tensor.New([]float64{1, -2}, 2)
	y := tensor.New([]float64{1, -1}, 2)
	want := (math.Log(1+math.Exp(-1)) + math.Log(1+math.Exp(-2))) / 2
	approxEq(t, "SoftMarginLoss mean", SoftMarginLoss(x, y).Item(), want, 1e-12)

	sum := SoftMarginLoss(x, y, WithReduction(ReduceSum)).Item()
	approxEq(t, "SoftMarginLoss sum", sum, want*2, 1e-12)

	none := SoftMarginLoss(x, y, WithReduction(ReduceNone))
	if none.Numel() != 2 {
		t.Fatalf("SoftMarginLoss none: want 2 elements, got shape %v", none.Shape)
	}
}

// SoftMarginLoss must stay finite for large-magnitude inputs (softplus path).
func TestSoftMarginLossStability(t *testing.T) {
	x := tensor.New([]float64{500, -500}, 2).SetRequiresGrad(true)
	y := tensor.New([]float64{-1, 1}, 2)
	loss := SoftMarginLoss(x, y)
	if v := loss.Item(); math.IsInf(v, 0) || math.IsNaN(v) {
		t.Fatalf("SoftMarginLoss: non-finite loss %v for large inputs", v)
	}
	loss.Backward()
	for i, g := range x.Grad.Data {
		if math.IsInf(g, 0) || math.IsNaN(g) {
			t.Fatalf("SoftMarginLoss: non-finite grad[%d]=%v", i, g)
		}
	}
}

func TestGradCheckSoftMarginLoss(t *testing.T) {
	x := seededRandn(101, 3, 4).SetRequiresGrad(true)
	y := tensor.New([]float64{1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1}, 3, 4)
	gradCheck(t, "SoftMarginLoss",
		func() *tensor.Tensor { return SoftMarginLoss(x, y) },
		[]*tensor.Tensor{x}, gcEps, gcTol, 0)
}

// ------------------------------------------------ MultiLabelSoftMarginLoss --

func TestMultiLabelSoftMarginLossValue(t *testing.T) {
	x := tensor.New([]float64{0.5, -1.0}, 1, 2)
	y := tensor.New([]float64{1, 0}, 1, 2)
	sig := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }
	want := -(math.Log(sig(0.5)) + math.Log(1-sig(-1.0))) / 2
	approxEq(t, "MultiLabelSoftMarginLoss", MultiLabelSoftMarginLoss(x, y).Item(), want, 1e-12)

	// 'none' returns the per-sample (N,) vector.
	x2 := seededRandn(102, 3, 4)
	y2 := tensor.New([]float64{1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0}, 3, 4)
	none := MultiLabelSoftMarginLoss(x2, y2, WithReduction(ReduceNone))
	if len(none.Shape) != 1 || none.Shape[0] != 3 {
		t.Fatalf("MultiLabelSoftMarginLoss none: want shape (3,), got %v", none.Shape)
	}
	mean := MultiLabelSoftMarginLoss(x2, y2).Item()
	approxEq(t, "MultiLabelSoftMarginLoss mean-of-none",
		(none.Data[0]+none.Data[1]+none.Data[2])/3, mean, 1e-12)
}

func TestGradCheckMultiLabelSoftMarginLoss(t *testing.T) {
	x := seededRandn(103, 3, 4).SetRequiresGrad(true)
	y := tensor.New([]float64{1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1}, 3, 4)
	gradCheck(t, "MultiLabelSoftMarginLoss",
		func() *tensor.Tensor { return MultiLabelSoftMarginLoss(x, y) },
		[]*tensor.Tensor{x}, gcEps, gcTol, 0)
}

// ---------------------------------------------------- MultiLabelMarginLoss --

// PyTorch's documented example: x=[[0.1,0.2,0.4,0.8]], y=[[3,0,-1,1]]
// (targets {3,0}; the trailing 1 is ignored past the first -1) -> 0.85.
func TestMultiLabelMarginLossPyTorchExample(t *testing.T) {
	x := tensor.New([]float64{0.1, 0.2, 0.4, 0.8}, 1, 4)
	y := tensor.New([]float64{3, 0, -1, 1}, 1, 4)
	approxEq(t, "MultiLabelMarginLoss", MultiLabelMarginLoss(x, y).Item(), 0.85, 1e-12)

	// 1-D input behaves as a single sample.
	x1 := tensor.New([]float64{0.1, 0.2, 0.4, 0.8}, 4)
	y1 := tensor.New([]float64{3, 0, -1, 1}, 4)
	approxEq(t, "MultiLabelMarginLoss 1D", MultiLabelMarginLoss(x1, y1).Item(), 0.85, 1e-12)
}

func TestMultiLabelMarginLossEmptyTargetRow(t *testing.T) {
	// Second row has no targets (leading -1) and must contribute zero.
	x := tensor.New([]float64{0.1, 0.2, 0.4, 0.8, 1, 2, 3, 4}, 2, 4)
	y := tensor.New([]float64{3, 0, -1, 1, -1, -1, -1, -1}, 2, 4)
	none := MultiLabelMarginLoss(x, y, WithReduction(ReduceNone))
	approxEq(t, "MultiLabelMarginLoss row0", none.Data[0], 0.85, 1e-12)
	approxEq(t, "MultiLabelMarginLoss empty row", none.Data[1], 0, 0)
	approxEq(t, "MultiLabelMarginLoss mean", MultiLabelMarginLoss(x, y).Item(), 0.425, 1e-12)
}

func TestGradCheckMultiLabelMarginLoss(t *testing.T) {
	x := seededRandn(104, 3, 5).SetRequiresGrad(true)
	y := tensor.New([]float64{
		1, 3, -1, -1, -1,
		0, -1, -1, -1, -1,
		2, 4, 0, -1, -1,
	}, 3, 5)
	gradCheck(t, "MultiLabelMarginLoss",
		func() *tensor.Tensor { return MultiLabelMarginLoss(x, y) },
		[]*tensor.Tensor{x}, gcEps, gcTol, 0)
}

// ------------------------------------------ TripletMarginWithDistanceLoss --

func TestTripletMarginWithDistanceLossDefaultMatchesTriplet(t *testing.T) {
	a := seededRandn(105, 4, 3)
	p := seededRandn(106, 4, 3)
	n := seededRandn(107, 4, 3)
	got := TripletMarginWithDistanceLoss(a, p, n, nil, 1.0).Item()
	want := TripletMarginLoss(a, p, n, 1.0).Item()
	approxEq(t, "TripletMarginWithDistanceLoss default", got, want, 1e-12)
}

func TestGradCheckTripletMarginWithDistanceLoss(t *testing.T) {
	a := seededRandn(108, 4, 3).SetRequiresGrad(true)
	p := seededRandn(109, 4, 3).SetRequiresGrad(true)
	n := seededRandn(110, 4, 3).SetRequiresGrad(true)

	// Default (pairwise L2) distance.
	loss := func() *tensor.Tensor { return TripletMarginWithDistanceLoss(a, p, n, nil, 1.0) }
	if loss().Item() <= 0 {
		t.Fatalf("TripletMarginWithDistanceLoss: want an active hinge for a meaningful gradcheck")
	}
	gradCheck(t, "TripletMarginWithDistanceLoss(L2)", loss, []*tensor.Tensor{a, p, n}, gcEps, gcTol, 0)

	// Custom differentiable distance: squared L2.
	sq := func(x1, x2 *tensor.Tensor) *tensor.Tensor {
		return x1.Sub(x2).Square().SumAxis(1, false)
	}
	loss = func() *tensor.Tensor { return TripletMarginWithDistanceLoss(a, p, n, sq, 0.5) }
	gradCheck(t, "TripletMarginWithDistanceLoss(sq)", loss, []*tensor.Tensor{a, p, n}, gcEps, gcTol, 0)
}

// ----------------------------------------------- CrossEntropy/NLL options --

func TestCrossEntropyClassWeights(t *testing.T) {
	logits := tensor.New([]float64{
		1.0, -0.5, 0.2,
		0.3, 0.8, -1.2,
	}, 2, 3)
	targets := tensor.New([]float64{0, 2}, 2)
	w := []float64{0.3, 0.5, 0.2}

	lp := logSoftmaxRows(logits.Data, 2, 3)
	l0 := -w[0] * lp[0*3+0]
	l1 := -w[2] * lp[1*3+2]

	// PyTorch mean: divide by the sum of the targets' weights, not N.
	wantMean := (l0 + l1) / (w[0] + w[2])
	approxEq(t, "CE weighted mean",
		CrossEntropyLoss(logits, targets, WithClassWeights(w)).Item(), wantMean, 1e-12)

	wantSum := l0 + l1
	approxEq(t, "CE weighted sum",
		CrossEntropyLoss(logits, targets, WithClassWeights(w), WithReduction(ReduceSum)).Item(),
		wantSum, 1e-12)

	none := CrossEntropyLoss(logits, targets, WithClassWeights(w), WithReduction(ReduceNone))
	approxEq(t, "CE weighted none[0]", none.Data[0], l0, 1e-12)
	approxEq(t, "CE weighted none[1]", none.Data[1], l1, 1e-12)

	// NLLLoss on explicit log-probs must agree.
	logP := tensor.New(lp, 2, 3)
	approxEq(t, "NLL weighted mean",
		NLLLoss(logP, targets, WithClassWeights(w)).Item(), wantMean, 1e-12)
}

func TestNLLLossIgnoreIndex(t *testing.T) {
	logits := seededRandn(111, 3, 4)
	lp := logSoftmaxRows(logits.Data, 3, 4)
	logP := tensor.New(lp, 3, 4)
	targets := tensor.New([]float64{1, -100, 3}, 3)

	// Ignored sample contributes 0 and is excluded from the denominator.
	want := (-lp[0*4+1] - lp[2*4+3]) / 2
	approxEq(t, "NLL ignore mean",
		NLLLoss(logP, targets, WithIgnoreIndex(-100)).Item(), want, 1e-12)

	none := NLLLoss(logP, targets, WithIgnoreIndex(-100), WithReduction(ReduceNone))
	approxEq(t, "NLL ignore none[1]", none.Data[1], 0, 0)

	// With class weights the denominator is the weight sum of the kept rows.
	w := []float64{1, 2, 3, 4}
	wantW := (2*(-lp[0*4+1]) + 4*(-lp[2*4+3])) / (2 + 4)
	approxEq(t, "NLL ignore+weights mean",
		NLLLoss(logP, targets, WithIgnoreIndex(-100), WithClassWeights(w)).Item(), wantW, 1e-12)

	// CrossEntropyLoss routes through the same path.
	approxEq(t, "CE ignore mean",
		CrossEntropyLoss(logits, targets, WithIgnoreIndex(-100)).Item(), want, 1e-12)
}

func TestCrossEntropyLabelSmoothing(t *testing.T) {
	const eps = 0.1
	logits := seededRandn(112, 3, 4)
	targets := tensor.New([]float64{2, 0, 3}, 3)
	lp := logSoftmaxRows(logits.Data, 3, 4)

	// loss_n = (1-eps)*(-logP[n,y]) + (eps/C)*sum_c(-logP[n,c]); mean over N.
	var want float64
	for n := 0; n < 3; n++ {
		y := int(targets.Data[n])
		var smooth float64
		for c := 0; c < 4; c++ {
			smooth -= lp[n*4+c]
		}
		want += (1-eps)*(-lp[n*4+y]) + (eps/4)*smooth
	}
	want /= 3
	approxEq(t, "CE label smoothing",
		CrossEntropyLoss(logits, targets, WithLabelSmoothing(eps)).Item(), want, 1e-12)

	// Equivalent formulation: smoothed target distribution eps/C.
	var want2 float64
	for n := 0; n < 3; n++ {
		y := int(targets.Data[n])
		for c := 0; c < 4; c++ {
			q := eps / 4
			if c == y {
				q += 1 - eps
			}
			want2 += -q * lp[n*4+c]
		}
	}
	want2 /= 3
	approxEq(t, "CE label smoothing (distribution form)", want, want2, 1e-12)
}

// All three options combined, against the hand-computed ATen decomposition.
func TestCrossEntropyAllOptions(t *testing.T) {
	const eps = 0.2
	logits := seededRandn(113, 4, 3)
	targets := tensor.New([]float64{0, 5, 2, 1}, 4) // 5 = ignored
	w := []float64{0.5, 1.5, 2.0}
	lp := logSoftmaxRows(logits.Data, 4, 3)

	var sum, denom float64
	for n := 0; n < 4; n++ {
		y := int(targets.Data[n])
		if y == 5 {
			continue
		}
		var smooth float64
		for c := 0; c < 3; c++ {
			smooth -= w[c] * lp[n*3+c]
		}
		sum += (1-eps)*(w[y]*-lp[n*3+y]) + (eps/3)*smooth
		denom += w[y]
	}
	want := sum / denom
	got := CrossEntropyLoss(logits, targets,
		WithClassWeights(w), WithIgnoreIndex(5), WithLabelSmoothing(eps)).Item()
	approxEq(t, "CE all options", got, want, 1e-12)
}

func TestGradCheckCrossEntropyOptions(t *testing.T) {
	logits := seededRandn(114, 4, 3).SetRequiresGrad(true)
	targets := tensor.New([]float64{0, 2, -100, 1}, 4)
	w := []float64{0.5, 1.5, 2.0}

	gradCheck(t, "CE weights",
		func() *tensor.Tensor {
			return CrossEntropyLoss(logits, targets, WithIgnoreIndex(-100), WithClassWeights(w))
		},
		[]*tensor.Tensor{logits}, gcEps, gcTol, 0)

	gradCheck(t, "CE smoothing",
		func() *tensor.Tensor {
			return CrossEntropyLoss(logits, targets, WithIgnoreIndex(-100), WithLabelSmoothing(0.1))
		},
		[]*tensor.Tensor{logits}, gcEps, gcTol, 0)

	gradCheck(t, "CE all options",
		func() *tensor.Tensor {
			return CrossEntropyLoss(logits, targets,
				WithClassWeights(w), WithIgnoreIndex(-100), WithLabelSmoothing(0.2))
		},
		[]*tensor.Tensor{logits}, gcEps, gcTol, 0)

	logP := seededRandn(115, 4, 3).SetRequiresGrad(true)
	gradCheck(t, "NLL weights+ignore",
		func() *tensor.Tensor {
			return NLLLoss(logP.LogSoftmax(1), targets, WithClassWeights(w), WithIgnoreIndex(-100))
		},
		[]*tensor.Tensor{logP}, gcEps, gcTol, 0)
}

// Default behavior of CrossEntropyLoss/NLLLoss is unchanged by the new
// option plumbing.
func TestCrossEntropyDefaultUnchanged(t *testing.T) {
	logits := seededRandn(116, 4, 3)
	targets := tensor.New([]float64{0, 2, 1, 2}, 4)
	lp := logSoftmaxRows(logits.Data, 4, 3)
	var want float64
	for n := 0; n < 4; n++ {
		want += -lp[n*3+int(targets.Data[n])]
	}
	want /= 4
	approxEq(t, "CE default", CrossEntropyLoss(logits, targets).Item(), want, 1e-12)
	approxEq(t, "NLL default", NLLLoss(tensor.New(lp, 4, 3), targets).Item(), want, 1e-12)
}
