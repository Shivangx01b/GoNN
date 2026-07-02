package nn

import (
	"math"
	"math/rand"
	"testing"

	"gonn/tensor"
)

// TestWithGradsToReplacements: gradients accumulate onto the replacement
// tensors' .Grad; the module's own params end the call untouched.
func TestWithGradsToReplacements(t *testing.T) {
	rand.Seed(50)
	l := NewLinear(3, 2, true)
	x := tensor.Randn(4, 3)
	repl := tensor.Randn(2, 3)

	lossFn := func(y *tensor.Tensor) *tensor.Tensor { return y.Pow(2).Mean() }
	_, grads := FunctionalCallGrad(l, x, map[string]*tensor.Tensor{"weight": repl},
		lossFn, WithGradsToReplacements())

	if repl.Grad == nil {
		t.Fatal("replacement received no gradient")
	}
	if d := maxAbsDiff(repl.Grad.Data, grads["weight"]); d != 0 {
		t.Fatalf("replacement grad != returned map: %g", d)
	}
	if l.Weight.Grad != nil {
		t.Fatal("module weight grad should be restored to nil")
	}

	// Accumulation: a second call adds on top.
	first := append([]float64(nil), repl.Grad.Data...)
	FunctionalCallGrad(l, x, map[string]*tensor.Tensor{"weight": repl},
		lossFn, WithGradsToReplacements())
	for i := range first {
		if math.Abs(repl.Grad.Data[i]-2*first[i]) > 1e-12 {
			t.Fatalf("grad did not accumulate at %d: %g vs 2*%g", i, repl.Grad.Data[i], first[i])
		}
	}
}

// TestFunctionalCallBufferReplacement: swapping BatchNorm running stats in
// eval mode — output must equal manual normalization with the swapped stats.
func TestFunctionalCallBufferReplacement(t *testing.T) {
	rand.Seed(51)
	bn := NewBatchNorm1d(3)
	bn.Eval()
	x := tensor.Randn(5, 3)

	mean := tensor.New([]float64{0.3, -0.2, 0.7}, 3)
	varr := tensor.New([]float64{1.5, 0.8, 2.0}, 3)
	y := FunctionalCall(bn, x, map[string]*tensor.Tensor{
		"running_mean": mean, "running_var": varr,
	})

	// Manual eval-mode BN with the swapped stats (affine is identity-initialized).
	for i := 0; i < 5; i++ {
		for c := 0; c < 3; c++ {
			want := (x.Data[i*3+c] - mean.Data[c]) / math.Sqrt(varr.Data[c]+1e-5)
			if math.Abs(y.Data[i*3+c]-want) > 1e-12 {
				t.Fatalf("buffer-swapped BN mismatch at (%d,%d): %g vs %g", i, c, y.Data[i*3+c], want)
			}
		}
	}
	// Originals restored (fresh BN stats: mean 0, var 1).
	for c := 0; c < 3; c++ {
		if bn.RunMean.Data[c] != 0 || bn.RunVar.Data[c] != 1 {
			t.Fatal("running stats not restored")
		}
	}
}

// TestFunctionalCallMulti: multi-input module (MultiHeadAttention) with a
// replaced projection weight equals a twin whose weight was set permanently.
func TestFunctionalCallMulti(t *testing.T) {
	rand.Seed(52)
	mha := NewMultiHeadAttention(4, 2)
	q := tensor.Randn(1, 3, 4)
	repl := tensor.Randn(4, 4)

	got := FunctionalCallMulti(mha, map[string]*tensor.Tensor{"qproj.weight": repl},
		func() *tensor.Tensor { return mha.Forward(q, q, q, false) })

	saved := append([]float64(nil), mha.QProj.Weight.Data...)
	copy(mha.QProj.Weight.Data, repl.Data)
	want := mha.Forward(q, q, q, false)
	copy(mha.QProj.Weight.Data, saved)

	if d := maxAbsDiff(got.Data, want.Data); d != 0 {
		t.Fatalf("FunctionalCallMulti mismatch: %g", d)
	}
	// Restoration check.
	if d := maxAbsDiff(mha.QProj.Weight.Data, saved); d != 0 {
		t.Fatal("module weight not restored")
	}
}

// TestFunctionalCallGradMulti: gradients for a multi-input module computed
// inside the swap window match a manual swap + backward.
func TestFunctionalCallGradMulti(t *testing.T) {
	rand.Seed(53)
	mha := NewMultiHeadAttention(4, 2)
	q := tensor.Randn(1, 3, 4)
	repl := tensor.Randn(4, 4)

	lossVal, grads := FunctionalCallGradMulti(mha,
		map[string]*tensor.Tensor{"qproj.weight": repl},
		func() *tensor.Tensor { return mha.Forward(q, q, q, false).Pow(2).Mean() },
		WithGradsToReplacements())

	// Manual reference: permanent swap, forward+backward, harvest.
	saved := append([]float64(nil), mha.QProj.Weight.Data...)
	copy(mha.QProj.Weight.Data, repl.Data)
	l := mha.Forward(q, q, q, false).Pow(2).Mean()
	if math.Abs(l.Item()-lossVal) > 1e-12 {
		t.Fatalf("loss mismatch: %g vs %g", l.Item(), lossVal)
	}
	l.Backward()
	wantGrad := append([]float64(nil), mha.QProj.Weight.Grad.Data...)
	copy(mha.QProj.Weight.Data, saved)
	mha.QProj.Weight.Grad = nil

	if d := maxAbsDiff(grads["qproj.weight"], wantGrad); d > 1e-12 {
		t.Fatalf("grad mismatch: %g", d)
	}
	if d := maxAbsDiff(repl.Grad.Data, wantGrad); d > 1e-12 {
		t.Fatalf("replacement grad mismatch: %g", d)
	}
}
