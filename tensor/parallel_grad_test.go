package tensor

// Concurrency tests for leaf gradient accumulation: G goroutines each build
// their own forward graph over the SAME leaf tensors and call Backward
// concurrently. Run with -race in CI (needs cgo); without -race these still
// verify logical correctness, because all per-graph gradient contributions
// are integer-valued — float64 addition of small integers is exact in any
// order, so the concurrent result must equal the sequential result bit for
// bit.

import (
	"sync"
	"testing"
)

// TestConcurrentBackwardSharedLeaf: each goroutine computes
// sum(leaf * scale_i) so its contribution to leaf.Grad is the constant
// scale_i per element. Sum over i of integer scales is exact regardless of
// accumulation order.
func TestConcurrentBackwardSharedLeaf(t *testing.T) {
	const G = 16
	build := func(leaf *Tensor, i int) *Tensor {
		return leaf.MulScalar(float64(i + 1)).Sum()
	}

	// Sequential reference.
	ref := New([]float64{1, 2, 3, 4}, 4).SetRequiresGrad(true)
	for i := 0; i < G; i++ {
		build(ref, i).Backward()
	}

	// Concurrent run over a fresh leaf with identical data.
	leaf := New([]float64{1, 2, 3, 4}, 4).SetRequiresGrad(true)
	var wg sync.WaitGroup
	for i := 0; i < G; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			build(leaf, i).Backward()
		}(i)
	}
	wg.Wait()

	if leaf.Grad == nil {
		t.Fatal("leaf.Grad is nil after concurrent Backward")
	}
	for k := range leaf.Grad.Data {
		if leaf.Grad.Data[k] != ref.Grad.Data[k] {
			t.Errorf("Grad[%d]: concurrent=%v sequential=%v", k, leaf.Grad.Data[k], ref.Grad.Data[k])
		}
	}
}

// TestConcurrentBackwardTwoLeavesDeepGraph exercises interior nodes (MatMul,
// Add broadcasting a bias, Sum) with TWO shared leaves, checking that both
// receive exact sums. Inputs and gradients are integer-valued throughout.
func TestConcurrentBackwardTwoLeavesDeepGraph(t *testing.T) {
	const G = 12
	newLeaves := func() (w, b *Tensor) {
		w = New([]float64{1, 2, 3, 4, 5, 6}, 2, 3).SetRequiresGrad(true)
		b = New([]float64{1, -1, 2}, 3).SetRequiresGrad(true)
		return
	}
	// Per-goroutine integer input matrix (graph-local; only w and b are shared).
	build := func(w, b *Tensor, i int) *Tensor {
		x := New([]float64{
			float64(i + 1), float64(i - 3),
			float64(2 * i), float64(-i),
			float64(i % 5), float64(i + 2),
		}, 3, 2)
		return x.MatMul(w).Add(b).Sum()
	}

	wRef, bRef := newLeaves()
	for i := 0; i < G; i++ {
		build(wRef, bRef, i).Backward()
	}

	w, b := newLeaves()
	var wg sync.WaitGroup
	for i := 0; i < G; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			build(w, b, i).Backward()
		}(i)
	}
	wg.Wait()

	for name, pair := range map[string][2]*Tensor{"w": {w, wRef}, "b": {b, bRef}} {
		got, want := pair[0].Grad, pair[1].Grad
		if got == nil {
			t.Fatalf("%s.Grad is nil after concurrent Backward", name)
		}
		for k := range got.Data {
			if got.Data[k] != want.Data[k] {
				t.Errorf("%s.Grad[%d]: concurrent=%v sequential=%v", name, k, got.Data[k], want.Data[k])
			}
		}
	}
}

// TestConcurrentBackwardFirstWriteRace targets the Grad==nil allocation race:
// all goroutines start from a leaf whose Grad is nil, contribute the identical
// constant 1 per element, and the final Grad must be exactly G — a dropped
// first-write would yield less.
func TestConcurrentBackwardFirstWriteRace(t *testing.T) {
	const G = 32
	for trial := 0; trial < 20; trial++ {
		leaf := New([]float64{5, 7}, 2).SetRequiresGrad(true)
		var wg sync.WaitGroup
		for i := 0; i < G; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				leaf.AddScalar(3).Sum().Backward()
			}()
		}
		wg.Wait()
		for k := range leaf.Grad.Data {
			if leaf.Grad.Data[k] != G {
				t.Fatalf("trial %d: Grad[%d]=%v, want %d (lost a concurrent contribution)",
					trial, k, leaf.Grad.Data[k], G)
			}
		}
	}
}
