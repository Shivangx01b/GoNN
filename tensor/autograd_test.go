package tensor

import "testing"

// TestBackwardDeepChain builds a very deep single-op chain and verifies that
// Backward traverses it without recursion (the old recursive topo sort risked
// exhausting the goroutine stack on graphs this deep) and produces the right
// gradient: y = x + 1 + 1 + ... has dy/dx = 1.
func TestBackwardDeepChain(t *testing.T) {
	const depth = 100_000
	x := Scalar(1.5).SetRequiresGrad(true)
	y := x
	for i := 0; i < depth; i++ {
		y = y.AddScalar(1)
	}
	y.Backward()
	if x.Grad == nil {
		t.Fatal("no gradient reached the leaf")
	}
	if got := x.Grad.Data[0]; got != 1 {
		t.Fatalf("dy/dx = %g, want 1", got)
	}
	if got := y.Item(); got != 1.5+depth {
		t.Fatalf("y = %g, want %g", got, 1.5+float64(depth))
	}
}

// TestBackwardDiamond checks gradient accumulation through a diamond-shaped
// graph: y = (x*2) + (x*3) so dy/dx = 5.
func TestBackwardDiamond(t *testing.T) {
	x := Scalar(2).SetRequiresGrad(true)
	a := x.MulScalar(2)
	b := x.MulScalar(3)
	y := a.Add(b)
	y.Backward()
	if got := x.Grad.Data[0]; got != 5 {
		t.Fatalf("dy/dx = %g, want 5", got)
	}
}

// TestNormalizeAxisValidation locks in the new out-of-range axis rejection.
func TestNormalizeAxisValidation(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for out-of-range axis")
		}
	}()
	Zeros(2, 3).SumAxis(2, false) // rank 2, axis 2 -> invalid
}
