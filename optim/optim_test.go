package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

// optimize uses the optimizer to minimize 0.5*(x-target)^2 starting from x0.
// Returns the value of x after iters steps.
func optimize(opt Optimizer, x *tensor.Tensor, target float64, iters int) float64 {
	for i := 0; i < iters; i++ {
		opt.ZeroGrad()
		diff := x.SubScalar(target)
		loss := diff.Square().Sum().MulScalar(0.5)
		loss.Backward()
		opt.Step()
	}
	return x.Data[0]
}

func TestSGDConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 1, 200)
	if math.Abs(got-1) > 1e-3 {
		t.Fatalf("SGD did not converge to 1, got %v", got)
	}
}

func TestSGDMomentumConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 0.05, WithMomentum(0.9))
	got := optimize(opt, x, 1, 200)
	if math.Abs(got-1) > 1e-2 {
		t.Fatalf("SGD+momentum did not converge to 1, got %v", got)
	}
}

func TestAdamConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewAdam([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 2, 300)
	if math.Abs(got-2) > 1e-2 {
		t.Fatalf("Adam did not converge to 2, got %v", got)
	}
}

func TestAdamWConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewAdamW([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 2, 300)
	if math.Abs(got-2) > 0.1 { // AdamW pulls slightly toward 0 via weight decay
		t.Fatalf("AdamW did not converge near 2, got %v", got)
	}
}

func TestRMSpropConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewRMSprop([]*tensor.Tensor{x}, 0.05)
	got := optimize(opt, x, 1, 300)
	if math.Abs(got-1) > 1e-2 {
		t.Fatalf("RMSprop did not converge to 1, got %v", got)
	}
}

func TestStepLRSchedule(t *testing.T) {
	x := tensor.New([]float64{1}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 1.0)
	sched := NewStepLR(opt, 5, 0.5)
	for i := 0; i < 5; i++ {
		sched.Step()
	}
	if math.Abs(opt.LR()-0.5) > 1e-9 {
		t.Fatalf("StepLR: lr after 5 steps = %v, want 0.5", opt.LR())
	}
}
