package backend

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

// refGemm is a naive reference for row-major batched C = op(A) @ op(B).
func refGemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	out := make([]float64, batch*m*n)
	at := func(bi, i, j int) float64 { // op(A)[i,j]
		if transA {
			return a[bi*m*k+j*m+i] // stored (k,m)
		}
		return a[bi*m*k+i*k+j] // stored (m,k)
	}
	bt := func(bi, i, j int) float64 { // op(B)[i,j]
		if transB {
			return b[bi*k*n+j*k+i] // stored (n,k)
		}
		return b[bi*k*n+i*n+j] // stored (k,n)
	}
	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float64
				for p := 0; p < k; p++ {
					s += at(bi, i, p) * bt(bi, p, j)
				}
				out[bi*m*n+i*n+j] = s
			}
		}
	}
	return out
}

func randSlice(rng *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rng.NormFloat64()
	}
	return s
}

func TestCPUGemm(t *testing.T) {
	be := cpuBackend{}
	if be.Name() != CPU {
		t.Fatalf("Name() = %q, want %q", be.Name(), CPU)
	}
	rng := rand.New(rand.NewSource(7))

	cases := []struct {
		batch, m, k, n   int
		transA, transB   bool
	}{
		{1, 3, 4, 5, false, false}, // plain 2D
		{1, 3, 4, 5, true, false},
		{1, 3, 4, 5, false, true},
		{1, 3, 4, 5, true, true},
		{4, 2, 3, 2, false, false}, // batched
		{4, 2, 3, 2, true, true},
		{2, 1, 1, 1, false, false}, // degenerate
		{3, 5, 2, 4, false, true},
	}
	for _, c := range cases {
		a := randSlice(rng, c.batch*c.m*c.k)
		b := randSlice(rng, c.batch*c.k*c.n)
		got := be.Gemm(a, b, c.batch, c.m, c.k, c.n, c.transA, c.transB)
		want := refGemm(a, b, c.batch, c.m, c.k, c.n, c.transA, c.transB)
		if len(got) != len(want) {
			t.Fatalf("Gemm%+v: len %d, want %d", c, len(got), len(want))
		}
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-12 {
				t.Fatalf("Gemm%+v: [%d] = %v, want %v", c, i, got[i], want[i])
			}
		}
	}

	// Empty dims produce zeros without panicking.
	if got := be.Gemm(nil, nil, 0, 3, 4, 5, false, false); len(got) != 0 {
		t.Fatalf("Gemm batch=0: len %d, want 0", len(got))
	}
}

// TestUseCurrentConcurrent hammers Use/Current from many goroutines under the
// race detector to verify the atomic backend selection.
func TestUseCurrentConcurrent(t *testing.T) {
	defer Use(Current()) // restore whatever was active

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				prev := Use(cpuBackend{})
				if Current() == nil {
					t.Error("Current() returned nil")
					return
				}
				Use(prev)
			}
		}()
	}
	wg.Wait()
	if Current().Name() != CPU {
		t.Fatalf("backend after restore = %q, want %q", Current().Name(), CPU)
	}
}

// TestBackend_InterfaceSatisfied is a compile-time check that cpuBackend
// implements Backend and is the default. It must NOT implement Elementwiser —
// the tensor package's Go closures are the CPU elementwise path.
func TestBackend_InterfaceSatisfied(t *testing.T) {
	var b Backend = cpuBackend{}
	if _, ok := b.(Elementwiser); ok {
		t.Fatal("cpuBackend must not implement Elementwiser")
	}
	if Current().Name() != CPU {
		t.Fatalf("default backend = %q, want %q", Current().Name(), CPU)
	}
}
