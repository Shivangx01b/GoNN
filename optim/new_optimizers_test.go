package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

func TestASGDConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewASGD([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 1, 500)
	if math.Abs(got-1) > 1e-2 {
		t.Fatalf("ASGD did not converge to 1, got %v", got)
	}
}

func TestSparseAdamConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewSparseAdam([]*tensor.Tensor{x}, 0.1)
	got := optimize(opt, x, 2, 300)
	if math.Abs(got-2) > 1e-2 {
		t.Fatalf("SparseAdam did not converge to 2, got %v", got)
	}
}

// TestSparseAdamSkipsZeroGrad verifies that entries with zero gradient are not
// updated (the dense-equivalent sparse behaviour).
func TestSparseAdamSkipsZeroGrad(t *testing.T) {
	x := tensor.New([]float64{5, 9}, 2).SetRequiresGrad(true)
	opt := NewSparseAdam([]*tensor.Tensor{x}, 0.1)
	// Manually set a gradient that is nonzero only on entry 0.
	x.Grad = tensor.New([]float64{1, 0}, 2)
	opt.Step()
	if x.Data[0] == 5 {
		t.Fatalf("SparseAdam should have updated entry 0, got %v", x.Data[0])
	}
	if x.Data[1] != 9 {
		t.Fatalf("SparseAdam should NOT have updated entry 1 (zero grad), got %v", x.Data[1])
	}
}

func TestLionConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewLion([]*tensor.Tensor{x}, 0.05)
	got := optimize(opt, x, 1, 500)
	if math.Abs(got-1) > 1e-2 {
		t.Fatalf("Lion did not converge to 1, got %v", got)
	}
}

func TestLAMBConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewLAMB([]*tensor.Tensor{x}, 0.02)
	got := optimize(opt, x, 2, 2000)
	if math.Abs(got-2) > 1e-2 {
		t.Fatalf("LAMB did not converge to 2, got %v", got)
	}
}

func TestAdafactorConverges(t *testing.T) {
	x := tensor.New([]float64{5}, 1).SetRequiresGrad(true)
	opt := NewAdafactor([]*tensor.Tensor{x}, 1.0)
	got := optimize(opt, x, 2, 800)
	if math.Abs(got-2) > 1e-2 {
		t.Fatalf("Adafactor did not converge to 2, got %v", got)
	}
}
