package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// With every target inside the shortlist, Forward must agree exactly with a
// plain LogSoftmax over the head outputs + NLL pick (the head IS the full
// computation for shortlist classes).
func TestAdaptiveSoftmaxShortlistAgreement(t *testing.T) {
	m := NewAdaptiveLogSoftmaxWithLoss(8, 5, []int{2})
	x := seededRandn(301, 4, 8)
	targets := tensor.New([]float64{0, 1, 1, 0}, 4)

	out, loss := m.Forward(x, targets)

	headLP := m.Head.Forward(x).LogSoftmax(1) // (4, headSize=3)
	for n := 0; n < 4; n++ {
		want := headLP.Data[n*m.HeadSize+int(targets.Data[n])]
		approxEq(t, "AdaptiveSoftmax shortlist output", out.Data[n], want, 1e-12)
	}
	wantLoss := NLLLoss(headLP, targets).Item()
	approxEq(t, "AdaptiveSoftmax shortlist loss", loss.Item(), wantLoss, 1e-12)
}

// Forward's target log-probs and loss must be consistent with the full
// LogProb distribution: output = LogProb[n, target[n]] and
// loss = NLLLoss(LogProb, target); rows must exp-sum to 1; Predict must be
// the row argmax.
func TestAdaptiveSoftmaxLogProbConsistency(t *testing.T) {
	m := NewAdaptiveLogSoftmaxWithLoss(8, 7, []int{2, 4}, WithHeadBias(true), WithDivValue(2.0))
	N := 6
	x := seededRandn(302, N, 8)
	targets := tensor.New([]float64{0, 3, 6, 2, 5, 1}, N) // shortlist + both clusters

	out, loss := m.Forward(x, targets)
	full := m.LogProb(x) // (N, 7)

	for n := 0; n < N; n++ {
		// Rows are proper log-distributions.
		var sum float64
		for c := 0; c < 7; c++ {
			sum += math.Exp(full.Data[n*7+c])
		}
		approxEq(t, "AdaptiveSoftmax row prob sum", sum, 1.0, 1e-9)

		want := full.Data[n*7+int(targets.Data[n])]
		approxEq(t, "AdaptiveSoftmax output vs LogProb", out.Data[n], want, 1e-9)
	}

	approxEq(t, "AdaptiveSoftmax loss vs NLL(LogProb)",
		loss.Item(), NLLLoss(full, targets).Item(), 1e-9)

	// Predict = argmax of LogProb.
	pred := m.Predict(x)
	for n := 0; n < N; n++ {
		best, bestC := math.Inf(-1), 0
		for c := 0; c < 7; c++ {
			if v := full.Data[n*7+c]; v > best {
				best, bestC = v, c
			}
		}
		if int(pred.Data[n]) != bestC {
			t.Errorf("Predict[%d]=%v, want argmax %d", n, pred.Data[n], bestC)
		}
	}
}

// The acceptance gradcheck: loss w.r.t. input and every parameter on the
// small config (cutoffs [2] with 5 classes).
func TestGradCheckAdaptiveSoftmax(t *testing.T) {
	m := NewAdaptiveLogSoftmaxWithLoss(8, 5, []int{2})
	x := seededRandn(303, 6, 8).SetRequiresGrad(true)
	targets := tensor.New([]float64{0, 1, 2, 4, 3, 1}, 6) // shortlist + tail

	loss := func() *tensor.Tensor {
		_, l := m.Forward(x, targets)
		return l
	}
	gradCheck(t, "AdaptiveLogSoftmaxWithLoss", loss, append(m.Parameters(), x), gcEps, gcTol, 0)
}

// Two tail clusters, head bias on, and a divValue large enough to hit the
// hsz >= 1 clamp on the second cluster.
func TestGradCheckAdaptiveSoftmaxTwoClusters(t *testing.T) {
	m := NewAdaptiveLogSoftmaxWithLoss(8, 6, []int{2, 4}, WithHeadBias(true))
	x := seededRandn(304, 5, 8).SetRequiresGrad(true)
	targets := tensor.New([]float64{0, 2, 4, 5, 1}, 5)

	loss := func() *tensor.Tensor {
		_, l := m.Forward(x, targets)
		return l
	}
	gradCheck(t, "AdaptiveLogSoftmaxWithLoss(2 clusters)", loss, append(m.Parameters(), x), gcEps, gcTol, 0)
}

// Structure mirrors PyTorch: head Linear(in, cutoffs[0]+nClusters), tail i =
// Sequential(Linear(in, in/div^(i+1), no bias), Linear(hsz, clusterSize, no
// bias)), with PyTorch-style dotted parameter names.
func TestAdaptiveSoftmaxStructure(t *testing.T) {
	m := NewAdaptiveLogSoftmaxWithLoss(16, 10, []int{4, 8})
	if m.ShortlistSize != 4 || m.NClusters != 2 || m.HeadSize != 6 {
		t.Fatalf("structure: shortlist=%d clusters=%d head=%d", m.ShortlistSize, m.NClusters, m.HeadSize)
	}
	// hsz_0 = 16/4 = 4, cluster size 4; hsz_1 = 16/16 = 1, cluster size 2.
	l00 := m.Tail[0].Layers[0].(*Linear)
	l01 := m.Tail[0].Layers[1].(*Linear)
	l10 := m.Tail[1].Layers[0].(*Linear)
	l11 := m.Tail[1].Layers[1].(*Linear)
	if l00.OutFeatures != 4 || l01.InFeatures != 4 || l01.OutFeatures != 4 {
		t.Errorf("tail0 sizes: %d %d %d", l00.OutFeatures, l01.InFeatures, l01.OutFeatures)
	}
	if l10.OutFeatures != 1 || l11.InFeatures != 1 || l11.OutFeatures != 2 {
		t.Errorf("tail1 sizes: %d %d %d", l10.OutFeatures, l11.InFeatures, l11.OutFeatures)
	}
	if l00.Bias != nil || l01.Bias != nil || m.Head.Bias != nil {
		t.Errorf("bias flags: tail must be bias-free and head_bias defaults to false")
	}

	names := map[string]bool{}
	for _, p := range m.NamedParameters() {
		names[p.Name] = true
	}
	for _, want := range []string{"head.weight", "tail.0.0.weight", "tail.0.1.weight", "tail.1.0.weight", "tail.1.1.weight"} {
		if !names[want] {
			t.Errorf("missing parameter %q (have %v)", want, names)
		}
	}
	if len(m.NamedParameters()) != 5 {
		t.Errorf("want 5 parameters, got %d", len(m.NamedParameters()))
	}

	// WithHeadBias(true) adds head.bias.
	mb := NewAdaptiveLogSoftmaxWithLoss(16, 10, []int{4}, WithHeadBias(true))
	if mb.Head.Bias == nil {
		t.Errorf("WithHeadBias(true): head.bias missing")
	}
}

func TestAdaptiveSoftmaxValidation(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: expected panic", name)
			}
		}()
		f()
	}
	mustPanic("empty cutoffs", func() { NewAdaptiveLogSoftmaxWithLoss(4, 5, nil) })
	mustPanic("non-increasing cutoffs", func() { NewAdaptiveLogSoftmaxWithLoss(4, 5, []int{3, 2}) })
	mustPanic("cutoff at nClasses", func() { NewAdaptiveLogSoftmaxWithLoss(4, 5, []int{5}) })
	mustPanic("cutoff zero", func() { NewAdaptiveLogSoftmaxWithLoss(4, 5, []int{0}) })
	m := NewAdaptiveLogSoftmaxWithLoss(4, 5, []int{2})
	mustPanic("target out of range", func() {
		m.Forward(seededRandn(305, 2, 4), tensor.New([]float64{0, 5}, 2))
	})
}
