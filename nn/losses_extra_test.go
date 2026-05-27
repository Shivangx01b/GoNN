package nn

import (
	"testing"

	"gonn/tensor"
)

// assertScalar checks that t is a scalar tensor and returns its value.
func assertScalar(t *testing.T, x *tensor.Tensor) float64 {
	t.Helper()
	if len(x.Data) != 1 {
		t.Fatalf("expected scalar tensor, got shape %v (len=%d)", x.Shape, len(x.Data))
	}
	return x.Data[0]
}

// assertGradFlows runs loss.Backward() and verifies param's Grad is non-nil
// and has at least one non-zero entry.
func assertGradFlows(t *testing.T, name string, loss *tensor.Tensor, param *tensor.Tensor) {
	t.Helper()
	loss.Backward()
	if param.Grad == nil {
		t.Fatalf("%s: grad is nil on param", name)
	}
	nonZero := false
	for _, v := range param.Grad.Data {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatalf("%s: all gradients are zero", name)
	}
}

func TestSmoothL1Loss(t *testing.T) {
	pred := tensor.New([]float64{1, 2, 3}, 3).SetRequiresGrad(true)
	tgt := tensor.New([]float64{0, 0, 0}, 3)
	loss := SmoothL1Loss(pred, tgt, 1.0)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("SmoothL1Loss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "SmoothL1Loss", loss, pred)
}

func TestL1LossAlias(t *testing.T) {
	pred := tensor.New([]float64{1, -2, 3}, 3).SetRequiresGrad(true)
	tgt := tensor.New([]float64{0, 0, 0}, 3)
	loss := L1Loss(pred, tgt)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("L1Loss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "L1Loss", loss, pred)
}

func TestPoissonNLLLossLogInput(t *testing.T) {
	input := tensor.New([]float64{0.5, 1.0, -0.5}, 3).SetRequiresGrad(true)
	tgt := tensor.New([]float64{2, 1, 0}, 3)
	loss := PoissonNLLLoss(input, tgt, true)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("PoissonNLLLoss(log): expected > 0, got %v", v)
	}
	assertGradFlows(t, "PoissonNLLLoss(log)", loss, input)
}

func TestPoissonNLLLossRaw(t *testing.T) {
	input := tensor.New([]float64{1.5, 2.0, 0.5}, 3).SetRequiresGrad(true)
	tgt := tensor.New([]float64{2, 1, 0}, 3)
	loss := PoissonNLLLoss(input, tgt, false)
	if v := assertScalar(t, loss); v == 0 {
		t.Fatalf("PoissonNLLLoss(raw): expected non-zero, got %v", v)
	}
	assertGradFlows(t, "PoissonNLLLoss(raw)", loss, input)
}

func TestGaussianNLLLoss(t *testing.T) {
	input := tensor.New([]float64{0.5, 1.0, -0.5}, 3).SetRequiresGrad(true)
	tgt := tensor.New([]float64{2, 1, 0}, 3)
	varT := tensor.New([]float64{1, 1, 1}, 3)
	loss := GaussianNLLLoss(input, tgt, varT)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("GaussianNLLLoss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "GaussianNLLLoss", loss, input)
}

func TestMarginRankingLoss(t *testing.T) {
	// x1 < x2 but y = +1 (we expect x1 > x2) -> positive loss.
	x1 := tensor.New([]float64{0.5, 1.0, 1.5}, 3).SetRequiresGrad(true)
	x2 := tensor.New([]float64{1.0, 1.0, 1.0}, 3)
	y := tensor.New([]float64{1, 1, 1}, 3)
	loss := MarginRankingLoss(x1, x2, y, 1.0)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("MarginRankingLoss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "MarginRankingLoss", loss, x1)
}

func TestHingeEmbeddingLoss(t *testing.T) {
	x := tensor.New([]float64{0.5, 0.2, 1.5}, 3).SetRequiresGrad(true)
	y := tensor.New([]float64{1, -1, -1}, 3)
	loss := HingeEmbeddingLoss(x, y, 1.0)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("HingeEmbeddingLoss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "HingeEmbeddingLoss", loss, x)
}

func TestCosineEmbeddingLoss(t *testing.T) {
	x1 := tensor.New([]float64{1, 0, 0, 1}, 2, 2).SetRequiresGrad(true)
	x2 := tensor.New([]float64{0, 1, 1, 0}, 2, 2)
	y := tensor.New([]float64{1, -1}, 2) // expect similar then dissimilar
	loss := CosineEmbeddingLoss(x1, x2, y, 0.0)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("CosineEmbeddingLoss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "CosineEmbeddingLoss", loss, x1)
}

func TestTripletMarginLoss(t *testing.T) {
	a := tensor.New([]float64{0, 0, 0, 0}, 2, 2).SetRequiresGrad(true)
	p := tensor.New([]float64{1, 1, 1, 1}, 2, 2) // far from anchor
	n := tensor.New([]float64{0.1, 0.1, 0.1, 0.1}, 2, 2) // close to anchor
	loss := TripletMarginLoss(a, p, n, 1.0)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("TripletMarginLoss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "TripletMarginLoss", loss, a)
}

func TestMultiMarginLoss(t *testing.T) {
	// Logits where the target class is NOT the highest, so loss > 0.
	input := tensor.New([]float64{
		0.2, 0.5, 0.1, // target = 0
		0.1, 0.3, 0.9, // target = 1
	}, 2, 3).SetRequiresGrad(true)
	target := tensor.New([]float64{0, 1}, 2)
	loss := MultiMarginLoss(input, target, 1.0)
	if v := assertScalar(t, loss); v <= 0 {
		t.Fatalf("MultiMarginLoss: expected > 0, got %v", v)
	}
	assertGradFlows(t, "MultiMarginLoss", loss, input)
}

func TestPReLUForwardAndGrad(t *testing.T) {
	p := NewPReLU(1)
	x := tensor.New([]float64{-2, -1, 0, 1, 2}, 5).SetRequiresGrad(true)
	y := p.Forward(x)
	// weight = 0.25 -> negatives scaled by 0.25, positives unchanged.
	want := []float64{-0.5, -0.25, 0, 1, 2}
	for i := range want {
		if got := y.Data[i]; got != want[i] {
			t.Fatalf("PReLU forward[%d]: got %v want %v", i, got, want[i])
		}
	}
	y.Sum().Backward()
	if p.Weight.Grad == nil {
		t.Fatalf("PReLU: weight grad is nil")
	}
	// dL/dw = sum over negative elements = -2 + -1 = -3
	if got := p.Weight.Grad.Data[0]; got != -3 {
		t.Fatalf("PReLU weight grad: got %v want -3", got)
	}
}

func TestGLUForwardAndGrad(t *testing.T) {
	// Input shape (1, 4): a = [1, 2], b = [0, 0].
	// sigmoid(0) = 0.5, so output = [0.5, 1.0].
	x := tensor.New([]float64{1, 2, 0, 0}, 1, 4).SetRequiresGrad(true)
	g := GLU{Dim: 1}
	y := g.Forward(x)
	if y.Shape[0] != 1 || y.Shape[1] != 2 {
		t.Fatalf("GLU shape: got %v want [1 2]", y.Shape)
	}
	if y.Data[0] != 0.5 || y.Data[1] != 1.0 {
		t.Fatalf("GLU forward: got %v want [0.5 1.0]", y.Data)
	}
	y.Sum().Backward()
	if x.Grad == nil {
		t.Fatalf("GLU: input grad is nil")
	}
	nonZero := false
	for _, v := range x.Grad.Data {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatalf("GLU: all input grads are zero")
	}
}

func TestNewActivationModules(t *testing.T) {
	x := tensor.New([]float64{-1, 0, 1}, 3)
	// Just confirm forward runs and produces correct shape for each new wrapper.
	mods := []Module{
		LogSigmoid{},
		Hardshrink{Lambda: 0.5},
		Softshrink{Lambda: 0.5},
		Tanhshrink{},
		Threshold{Thresh: 0, Value: -1},
		CELU{Alpha: 1.0},
		Mish{},
		HardSwish{},
		Softplus{},
		Softsign{},
		HardSigmoid{},
		SELU{},
		ReLU6{},
	}
	for _, m := range mods {
		y := m.Forward(x)
		if len(y.Data) != 3 {
			t.Fatalf("module %T: expected len 3, got %d", m, len(y.Data))
		}
	}
}
