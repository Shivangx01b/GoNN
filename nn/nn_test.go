package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

func TestLinearForwardShape(t *testing.T) {
	l := NewLinear(4, 3, true)
	x := tensor.Randn(2, 4)
	y := l.Forward(x)
	if y.Shape[0] != 2 || y.Shape[1] != 3 {
		t.Fatalf("Linear shape: got %v want [2 3]", y.Shape)
	}
}

func TestLinearBackpropTrains(t *testing.T) {
	// Tiny regression: y = 2x + 3, 1D -> 1D.
	l := NewLinear(1, 1, true)
	xData := make([]float64, 50)
	yData := make([]float64, 50)
	for i := range xData {
		xData[i] = float64(i)/10 - 2.5
		yData[i] = 2*xData[i] + 3
	}
	X := tensor.New(xData, 50, 1)
	Y := tensor.New(yData, 50, 1)

	lr := 0.05
	for iter := 0; iter < 500; iter++ {
		l.Weight.Grad = nil
		if l.Bias != nil {
			l.Bias.Grad = nil
		}
		pred := l.Forward(X)
		loss := MSELoss(pred, Y)
		loss.Backward()
		for _, p := range l.Parameters() {
			for i := range p.Data {
				p.Data[i] -= lr * p.Grad.Data[i]
			}
		}
	}
	w := l.Weight.Data[0]
	b := l.Bias.Data[0]
	if math.Abs(w-2) > 0.05 || math.Abs(b-3) > 0.05 {
		t.Fatalf("Linear did not learn y=2x+3, got w=%v b=%v", w, b)
	}
}

func TestSequentialMLPForwardShape(t *testing.T) {
	m := NewSequential(
		NewLinear(8, 16, true),
		ReLU{},
		NewLinear(16, 4, true),
	)
	x := tensor.Randn(5, 8)
	y := m.Forward(x)
	if y.Shape[0] != 5 || y.Shape[1] != 4 {
		t.Fatalf("Sequential shape: got %v want [5 4]", y.Shape)
	}
	if len(m.Parameters()) != 4 { // 2 linears * (W + b)
		t.Fatalf("expected 4 params, got %d", len(m.Parameters()))
	}
}

func TestCrossEntropyLossDecreases(t *testing.T) {
	logits := tensor.New([]float64{0.1, 0.2, 0.7, 0.1, 0.8, 0.1}, 2, 3).SetRequiresGrad(true)
	targets := tensor.New([]float64{2, 1}, 2)
	loss1 := CrossEntropyLoss(logits, targets)
	v1 := loss1.Data[0]
	// Bump logits in the direction of the target class.
	logits.Data[0*3+2] += 1
	logits.Data[1*3+1] += 1
	loss2 := CrossEntropyLoss(logits, targets)
	v2 := loss2.Data[0]
	if v2 >= v1 {
		t.Fatalf("CE should decrease after pushing target logit up, got %v -> %v", v1, v2)
	}
}

func TestMSELossGradient(t *testing.T) {
	pred := tensor.New([]float64{1, 2, 3}, 3).SetRequiresGrad(true)
	tgt := tensor.New([]float64{0, 0, 0}, 3)
	loss := MSELoss(pred, tgt)
	loss.Backward()
	// d/dpred (1/N * sum(pred^2)) = 2/N * pred = 2/3 * [1,2,3]
	want := []float64{2.0 / 3, 4.0 / 3, 6.0 / 3}
	for i := range want {
		if math.Abs(pred.Grad.Data[i]-want[i]) > 1e-9 {
			t.Fatalf("MSE grad[%d]: got %v want %v", i, pred.Grad.Data[i], want[i])
		}
	}
}

func TestActivationModulesPassThrough(t *testing.T) {
	x := tensor.New([]float64{-1, 0, 1}, 3)
	if g := (ReLU{}).Forward(x).Data; g[0] != 0 || g[1] != 0 || g[2] != 1 {
		t.Fatalf("ReLU: got %v", g)
	}
	if g := (Sigmoid{}).Forward(x).Data; math.Abs(g[1]-0.5) > 1e-9 {
		t.Fatalf("Sigmoid(0) != 0.5, got %v", g[1])
	}
}
