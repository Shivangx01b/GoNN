package tensor

import (
	"math"
	"testing"
)

func TestLogSigmoidForward(t *testing.T) {
	x := New([]float64{-1, 0, 1, 2}, 4)
	y := x.LogSigmoid()
	want := []float64{
		-math.Log(1 + math.Exp(1)),  // log(sigmoid(-1)) = -log(1+e)
		-math.Log(2),                // log(0.5) = -log(2)
		-math.Log(1 + math.Exp(-1)),
		-math.Log(1 + math.Exp(-2)),
	}
	approxEq(t, y.Data, want, 1e-9)
}

func TestLogSigmoidBackward(t *testing.T) {
	x := New([]float64{-1, 0, 1}, 3).SetRequiresGrad(true)
	x.LogSigmoid().Sum().Backward()
	// d/dx log(sigmoid(x)) = 1 / (1 + exp(x))
	want := []float64{
		1 / (1 + math.Exp(-1)),
		1 / (1 + math.Exp(0)),
		1 / (1 + math.Exp(1)),
	}
	approxEq(t, x.Grad.Data, want, 1e-9)
}

func TestHardshrinkForward(t *testing.T) {
	x := New([]float64{-2, -0.5, 0, 0.5, 2}, 5)
	y := x.Hardshrink(1.0)
	approxEq(t, y.Data, []float64{-2, 0, 0, 0, 2}, 1e-9)
}

func TestHardshrinkBackward(t *testing.T) {
	x := New([]float64{-2, -0.5, 0, 0.5, 2}, 5).SetRequiresGrad(true)
	x.Hardshrink(1.0).Sum().Backward()
	approxEq(t, x.Grad.Data, []float64{1, 0, 0, 0, 1}, 1e-9)
}

func TestSoftshrinkForward(t *testing.T) {
	x := New([]float64{-2, -0.5, 0, 0.5, 2}, 5)
	y := x.Softshrink(1.0)
	approxEq(t, y.Data, []float64{-1, 0, 0, 0, 1}, 1e-9)
}

func TestSoftshrinkBackward(t *testing.T) {
	x := New([]float64{-2, -0.5, 0.5, 2}, 4).SetRequiresGrad(true)
	x.Softshrink(1.0).Sum().Backward()
	approxEq(t, x.Grad.Data, []float64{1, 0, 0, 1}, 1e-9)
}

func TestTanhshrinkForward(t *testing.T) {
	x := New([]float64{-1, 0, 1}, 3)
	y := x.Tanhshrink()
	want := []float64{-1 - math.Tanh(-1), 0, 1 - math.Tanh(1)}
	approxEq(t, y.Data, want, 1e-9)
}

func TestTanhshrinkBackward(t *testing.T) {
	x := New([]float64{-1, 0.5, 1}, 3).SetRequiresGrad(true)
	x.Tanhshrink().Sum().Backward()
	want := []float64{
		math.Tanh(-1) * math.Tanh(-1),
		math.Tanh(0.5) * math.Tanh(0.5),
		math.Tanh(1) * math.Tanh(1),
	}
	approxEq(t, x.Grad.Data, want, 1e-9)
}

func TestThresholdForward(t *testing.T) {
	x := New([]float64{-2, -1, 0, 1, 2}, 5)
	y := x.Threshold(0, -5)
	approxEq(t, y.Data, []float64{-5, -5, -5, 1, 2}, 1e-9)
}

func TestThresholdBackward(t *testing.T) {
	x := New([]float64{-2, 0, 2}, 3).SetRequiresGrad(true)
	x.Threshold(0, -5).Sum().Backward()
	approxEq(t, x.Grad.Data, []float64{0, 0, 1}, 1e-9)
}

func TestCELUForward(t *testing.T) {
	x := New([]float64{-1, 0, 1}, 3)
	y := x.CELU(1.0)
	// alpha=1: x>=0 -> x; x<0 -> exp(x)-1
	want := []float64{math.Exp(-1) - 1, 0, 1}
	approxEq(t, y.Data, want, 1e-9)
}

func TestCELUBackward(t *testing.T) {
	x := New([]float64{-2, -0.5, 0.5, 2}, 4).SetRequiresGrad(true)
	x.CELU(1.0).Sum().Backward()
	want := []float64{math.Exp(-2), math.Exp(-0.5), 1, 1}
	approxEq(t, x.Grad.Data, want, 1e-9)
}

func TestRReLUForward(t *testing.T) {
	// Deterministic midpoint slope = 0.2 from U(0.1, 0.3).
	x := New([]float64{-2, -1, 0, 1, 2}, 5)
	y := x.RReLU(0.1, 0.3, nil)
	approxEq(t, y.Data, []float64{-0.4, -0.2, 0, 1, 2}, 1e-9)
}

func TestRReLUBackward(t *testing.T) {
	x := New([]float64{-1, 1}, 2).SetRequiresGrad(true)
	x.RReLU(0.1, 0.3, nil).Sum().Backward()
	approxEq(t, x.Grad.Data, []float64{0.2, 1}, 1e-9)
}
