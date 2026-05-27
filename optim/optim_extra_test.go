package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

func TestAdamaxConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewAdamax([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 2, 300)
	if math.Abs(got-2) > 1e-2 {
		t.Fatalf("Adamax did not converge to 2, got %v", got)
	}
}

func TestRAdamConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewRAdam([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 2, 300)
	if math.Abs(got-2) > 1e-2 {
		t.Fatalf("RAdam did not converge to 2, got %v", got)
	}
}

func TestRpropConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewRprop([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 1, 300)
	if math.Abs(got-1) > 1e-2 {
		t.Fatalf("Rprop did not converge to 1, got %v", got)
	}
}

func TestLBFGSConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewLBFGS([]*tensor.Tensor{x}, 1)
	target := 1.0
	closure := func() float64 {
		opt.ZeroGrad()
		diff := x.SubScalar(target)
		loss := diff.Square().Sum().MulScalar(0.5)
		loss.Backward()
		return loss.Item()
	}
	// A handful of outer Steps is plenty for a 1-D quadratic.
	for i := 0; i < 10; i++ {
		opt.Step(closure)
	}
	if math.Abs(x.Data[0]-target) > 1e-3 {
		t.Fatalf("LBFGS did not converge to %v, got %v", target, x.Data[0])
	}
}

func TestPolynomialLRSchedule(t *testing.T) {
	x := tensor.New([]float64{1}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 1.0)
	sched := NewPolynomialLR(opt, 10, 1.0)
	for i := 0; i < 5; i++ {
		sched.Step()
	}
	// lr = 1 * (1 - 5/10)^1 = 0.5
	if math.Abs(opt.LR()-0.5) > 1e-9 {
		t.Fatalf("PolynomialLR: lr after 5 steps = %v, want 0.5", opt.LR())
	}
}

func TestChainedScheduler(t *testing.T) {
	x := tensor.New([]float64{1}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 1.0)
	// Two ExponentialLRs with gamma=0.5 chained -> effective gamma 0.25 per step.
	s1 := NewExponentialLR(opt, 0.5)
	s2 := NewExponentialLR(opt, 0.5)
	sched := NewChainedScheduler(s1, s2)
	sched.Step()
	if math.Abs(opt.LR()-0.25) > 1e-9 {
		t.Fatalf("ChainedScheduler: lr after 1 step = %v, want 0.25", opt.LR())
	}
}

func TestSequentialLRSchedule(t *testing.T) {
	x := tensor.New([]float64{1}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 1.0)
	a := NewExponentialLR(opt, 0.5)
	b := NewExponentialLR(opt, 0.1)
	sched := NewSequentialLR([]Scheduler{a, b}, []int{3})
	// Three steps with `a`: lr = 1 * 0.5^3 = 0.125
	for i := 0; i < 3; i++ {
		sched.Step()
	}
	if math.Abs(opt.LR()-0.125) > 1e-9 {
		t.Fatalf("SequentialLR: lr after 3 steps = %v, want 0.125", opt.LR())
	}
	// Fourth step crosses milestone -> uses `b`.
	sched.Step()
	if math.Abs(opt.LR()-0.0125) > 1e-9 {
		t.Fatalf("SequentialLR: lr after 4 steps = %v, want 0.0125", opt.LR())
	}
}

func TestCyclicLRSchedule(t *testing.T) {
	x := tensor.New([]float64{1}, 1).SetRequiresGrad(true)
	opt := NewSGD([]*tensor.Tensor{x}, 0.0)
	sched := NewCyclicLR(opt, 0.1, 1.0, 5)
	// Constructor sets lr to baseLR.
	if math.Abs(opt.LR()-0.1) > 1e-9 {
		t.Fatalf("CyclicLR init: lr = %v, want 0.1", opt.LR())
	}
	// After 5 steps we are at the peak (maxLR).
	for i := 0; i < 5; i++ {
		sched.Step()
	}
	if math.Abs(opt.LR()-1.0) > 1e-9 {
		t.Fatalf("CyclicLR peak: lr = %v, want 1.0", opt.LR())
	}
	// After another 5 steps we are back at baseLR.
	for i := 0; i < 5; i++ {
		sched.Step()
	}
	if math.Abs(opt.LR()-0.1) > 1e-9 {
		t.Fatalf("CyclicLR trough: lr = %v, want 0.1", opt.LR())
	}
}
