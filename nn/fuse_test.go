package nn

import (
	"testing"

	"gonn/tensor"
)

// randomizeBN gives the batch norm non-trivial running statistics and affine
// parameters so fusion is exercised beyond the identity transform.
func randomizeBN(bn *batchNormNd, seed int64) {
	r := seededRandn(seed, 4*bn.NumFeatures)
	for c := 0; c < bn.NumFeatures; c++ {
		bn.RunMean.Data[c] = r.Data[4*c]
		bn.RunVar.Data[c] = 0.5 + r.Data[4*c+1]*r.Data[4*c+1] // strictly positive
		bn.Weight.Data[c] = 1 + 0.3*r.Data[4*c+2]
		bn.Bias.Data[c] = 0.5 * r.Data[4*c+3]
	}
}

func TestFuseConvBNEval(t *testing.T) {
	conv := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
	bn := NewBatchNorm2d(3)
	randomizeBN(&bn.batchNormNd, 301)
	bn.Eval()

	x := seededRandn(302, 2, 2, 6, 7)
	want := bn.Forward(conv.Forward(x))
	fused := FuseConvBNEval(conv, bn)
	got := fused.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("fused conv+bn differs from conv->bn eval: max diff %g", d)
	}
}

func TestFuseConvBNEvalNoBias(t *testing.T) {
	conv := NewConv2d(2, 4, 3, WithPad(1), WithNoBias())
	bn := NewBatchNorm2d(4)
	randomizeBN(&bn.batchNormNd, 303)
	bn.Eval()

	x := seededRandn(304, 2, 2, 5, 5)
	want := bn.Forward(conv.Forward(x))
	fused := FuseConvBNEval(conv, bn)
	got := fused.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("fused bias-free conv+bn differs from conv->bn eval: max diff %g", d)
	}
}

func TestFuseLinearBNEval(t *testing.T) {
	l := NewLinear(6, 4, true)
	bn := NewBatchNorm1d(4)
	randomizeBN(&bn.batchNormNd, 305)
	bn.Eval()

	x := seededRandn(306, 5, 6)
	want := bn.Forward(l.Forward(x))
	fused := FuseLinearBNEval(l, bn)
	got := fused.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("fused linear+bn differs from linear->bn eval: max diff %g", d)
	}
}

func TestGradCheckFusedLayers(t *testing.T) {
	// The fused result is a plain layer, so it is fully trainable/gradcheckable.
	l := NewLinear(4, 3, true)
	bn := NewBatchNorm1d(3)
	randomizeBN(&bn.batchNormNd, 307)
	bn.Eval()
	fused := FuseLinearBNEval(l, bn)
	x := seededRandn(308, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return fused.Forward(x).Square().Mean() }
	gradCheck(t, "FusedLinearBN", loss, append(fused.Parameters(), x), gcEps, gcTol, 0)
}
