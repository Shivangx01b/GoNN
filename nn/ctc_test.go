package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// ctcCollapse maps a frame-level path to its label sequence: merge repeats,
// then drop blanks.
func ctcCollapse(path []int, blank int) []int {
	var out []int
	prev := -1
	for _, c := range path {
		if c != prev && c != blank {
			out = append(out, c)
		}
		prev = c
	}
	return out
}

// ctcBruteForce computes -log p(target | logProbs[:,n,:]) by enumerating all
// C^T alignments and summing the probabilities of those that collapse to
// target. Exponential — for tiny test cases only.
func ctcBruteForce(logProbs *tensor.Tensor, n, tLen int, target []int, blank int) float64 {
	N, C := logProbs.Shape[1], logProbs.Shape[2]
	eq := func(a, b []int) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
		return true
	}
	total := 0.0
	path := make([]int, tLen)
	var rec func(t int, acc float64)
	rec = func(t int, acc float64) {
		if t == tLen {
			if eq(ctcCollapse(path, blank), target) {
				total += math.Exp(acc)
			}
			return
		}
		for c := 0; c < C; c++ {
			path[t] = c
			rec(t+1, acc+logProbs.Data[(t*N+n)*C+c])
		}
	}
	rec(0, 0)
	return -math.Log(total)
}

// seededLogProbs returns a (T, N, C) tensor of valid log-probabilities.
func seededLogProbs(seed int64, T, N, C int) *tensor.Tensor {
	return seededRandn(seed, T, N, C).LogSoftmax(2)
}

// The task's hand-verifiable tiny case: T=2, one target label. The valid
// alignments for target [1] with blank 0 are (0,1), (1,0), (1,1); the loss is
// -log of the sum of their probabilities — compare against brute-force
// enumeration exactly.
func TestCTCLossTinyBruteForce(t *testing.T) {
	lp := seededLogProbs(201, 2, 1, 3)
	target := tensor.New([]float64{1}, 1)

	none := CTCLoss(lp, target, []int{2}, []int{1}, 0, WithReduction(ReduceNone))
	want := ctcBruteForce(lp, 0, 2, []int{1}, 0)
	approxEq(t, "CTC tiny brute force", none.Data[0], want, 1e-12)

	// Explicit three-alignment formula as an independent cross-check.
	p := func(c0, c1 int) float64 {
		return math.Exp(lp.Data[0*3+c0] + lp.Data[1*3+c1])
	}
	explicit := -math.Log(p(0, 1) + p(1, 0) + p(1, 1))
	approxEq(t, "CTC tiny explicit", none.Data[0], explicit, 1e-12)

	// Reductions: mean divides by max(targetLen,1)=1 then averages over N=1.
	approxEq(t, "CTC tiny mean", CTCLoss(lp, target, []int{2}, []int{1}, 0).Item(), want, 1e-12)
	approxEq(t, "CTC tiny sum",
		CTCLoss(lp, target, []int{2}, []int{1}, 0, WithReduction(ReduceSum)).Item(), want, 1e-12)
}

// Repeated labels force a blank between them; the alpha recursion must not
// take the s-2 skip. Brute-forced over all 3^5 alignments.
func TestCTCLossRepeatedLabelBruteForce(t *testing.T) {
	lp := seededLogProbs(202, 5, 1, 3)
	target := tensor.New([]float64{1, 1}, 2)
	got := CTCLoss(lp, target, []int{5}, []int{2}, 0, WithReduction(ReduceNone)).Data[0]
	want := ctcBruteForce(lp, 0, 5, []int{1, 1}, 0)
	approxEq(t, "CTC repeated label", got, want, 1e-12)
}

// Batched case with distinct input/target lengths, plus the exact PyTorch
// reduction semantics (mean divides per-sample loss by its target length,
// then averages over the batch).
func TestCTCLossBatchAndReductions(t *testing.T) {
	T, N, C := 4, 2, 4
	lp := seededLogProbs(203, T, N, C)
	targets := tensor.New([]float64{2, 1, 3, 3}, 4) // sample0: [2], sample1: [1,3,3]
	inputLens := []int{3, 4}
	targetLens := []int{1, 3}

	none := CTCLoss(lp, targets, inputLens, targetLens, 0, WithReduction(ReduceNone))
	w0 := ctcBruteForce(lp, 0, 3, []int{2}, 0)
	w1 := ctcBruteForce(lp, 1, 4, []int{1, 3, 3}, 0)
	approxEq(t, "CTC batch none[0]", none.Data[0], w0, 1e-12)
	approxEq(t, "CTC batch none[1]", none.Data[1], w1, 1e-12)

	approxEq(t, "CTC batch sum",
		CTCLoss(lp, targets, inputLens, targetLens, 0, WithReduction(ReduceSum)).Item(),
		w0+w1, 1e-12)
	approxEq(t, "CTC batch mean",
		CTCLoss(lp, targets, inputLens, targetLens, 0).Item(),
		(w0/1+w1/3)/2, 1e-12)
}

// A zero-length target scores the all-blank path; the mean reduction clamps
// the divisor to 1 (ATen's clamp_min).
func TestCTCLossZeroLengthTarget(t *testing.T) {
	T := 3
	lp := seededLogProbs(204, T, 1, 3)
	empty := tensor.New([]float64{}, 0)
	got := CTCLoss(lp, empty, []int{T}, []int{0}, 0, WithReduction(ReduceNone)).Data[0]
	want := 0.0
	for tt := 0; tt < T; tt++ {
		want -= lp.Data[tt*3+0] // blank at every frame
	}
	approxEq(t, "CTC zero-length target", got, want, 1e-12)
	approxEq(t, "CTC zero-length mean",
		CTCLoss(lp, empty, []int{T}, []int{0}, 0).Item(), want, 1e-12)
}

// Inputs too short to fit the target ([1,1] needs >= 3 frames) give +Inf loss
// (PyTorch with zero_infinity=false) and a zero gradient.
func TestCTCLossInfeasible(t *testing.T) {
	lp := seededLogProbs(205, 2, 1, 3).SetRequiresGrad(true)
	target := tensor.New([]float64{1, 1}, 2)
	loss := CTCLoss(lp, target, []int{2}, []int{2}, 0, WithReduction(ReduceSum))
	if !math.IsInf(loss.Item(), 1) {
		t.Fatalf("CTCLoss: want +Inf for infeasible alignment, got %v", loss.Item())
	}
	loss.Backward()
	for i, g := range lp.Grad.Data {
		if g != 0 {
			t.Fatalf("CTCLoss: infeasible sample must have zero grad, grad[%d]=%v", i, g)
		}
	}
}

// The acceptance test: central-difference gradcheck of the analytic
// alpha-beta gradient on (T=6, N=2, C=4) at 1e-5 relative tolerance,
// including a repeated label and unequal lengths. The gradient is checked
// directly w.r.t. the logProbs entries as free variables.
func TestGradCheckCTCLoss(t *testing.T) {
	T, N, C := 6, 2, 4
	lp := seededLogProbs(206, T, N, C).Copy().SetRequiresGrad(true)
	targets := tensor.New([]float64{1, 2, 3, 3, 1}, 5) // sample0: [1,2], sample1: [3,3,1]
	inputLens := []int{6, 5}
	targetLens := []int{2, 3}

	gradCheck(t, "CTCLoss(mean)",
		func() *tensor.Tensor { return CTCLoss(lp, targets, inputLens, targetLens, 0) },
		[]*tensor.Tensor{lp}, 1e-5, 1e-5, 0)

	gradCheck(t, "CTCLoss(sum)",
		func() *tensor.Tensor {
			return CTCLoss(lp, targets, inputLens, targetLens, 0, WithReduction(ReduceSum))
		},
		[]*tensor.Tensor{lp}, 1e-5, 1e-5, 0)
}

// Composing with LogSoftmax reproduces PyTorch's gradient w.r.t. pre-softmax
// logits (PyTorch's raw log_probs gradient differs from the free-variable
// derivative by a term the LogSoftmax backward annihilates, so this chain is
// the PyTorch-parity check).
func TestGradCheckCTCLossThroughLogSoftmax(t *testing.T) {
	T, N, C := 6, 2, 4
	logits := seededRandn(207, T, N, C).SetRequiresGrad(true)
	targets := tensor.New([]float64{1, 2, 3, 3, 1}, 5)
	inputLens := []int{6, 5}
	targetLens := []int{2, 3}

	gradCheck(t, "CTCLoss(logits)",
		func() *tensor.Tensor {
			return CTCLoss(logits.LogSoftmax(2), targets, inputLens, targetLens, 0)
		},
		[]*tensor.Tensor{logits}, 1e-5, 1e-5, 0)
}

// Zero-length targets participate in the batch gradient too.
func TestGradCheckCTCLossZeroLength(t *testing.T) {
	T, N, C := 4, 2, 3
	lp := seededLogProbs(208, T, N, C).Copy().SetRequiresGrad(true)
	targets := tensor.New([]float64{2}, 1) // sample0: [], sample1: [2]
	inputLens := []int{4, 3}
	targetLens := []int{0, 1}

	gradCheck(t, "CTCLoss(zero-length)",
		func() *tensor.Tensor { return CTCLoss(lp, targets, inputLens, targetLens, 0) },
		[]*tensor.Tensor{lp}, 1e-5, 1e-5, 0)
}

// A non-zero blank index must work the same way.
func TestCTCLossNonZeroBlank(t *testing.T) {
	lp := seededLogProbs(209, 3, 1, 3)
	blank := 2
	target := tensor.New([]float64{0, 1}, 2)
	got := CTCLoss(lp, target, []int{3}, []int{2}, blank, WithReduction(ReduceNone)).Data[0]
	want := ctcBruteForce(lp, 0, 3, []int{0, 1}, blank)
	approxEq(t, "CTC non-zero blank", got, want, 1e-12)
}
