package tensor

import (
	"math"
	"math/rand"
	"testing"
)

func randTensor(seed int64, shape ...int) *Tensor {
	rng := rand.New(rand.NewSource(seed))
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = rng.NormFloat64()
	}
	return t
}

// unrolledBMM computes (B,M,K)@(B,K,N) as a loop of 2D MatMuls — the
// reference the batched path must match exactly (same gonum GEMM per slice).
func unrolledBMM(a, b *Tensor) *Tensor {
	B, m, k := a.Shape[0], a.Shape[1], a.Shape[2]
	n := b.Shape[2]
	out := Zeros(B, m, n)
	for bi := 0; bi < B; bi++ {
		as := New(append([]float64(nil), a.Data[bi*m*k:(bi+1)*m*k]...), m, k)
		bs := New(append([]float64(nil), b.Data[bi*k*n:(bi+1)*k*n]...), k, n)
		cs := as.MatMul(bs)
		copy(out.Data[bi*m*n:(bi+1)*m*n], cs.Data)
	}
	return out
}

func TestBMMMatchesUnrolled(t *testing.T) {
	a := randTensor(1, 4, 3, 5)
	b := randTensor(2, 4, 5, 2)
	got := a.BMM(b)
	want := unrolledBMM(a, b)
	if !shapesEqual(got.Shape, want.Shape) {
		t.Fatalf("shape %v, want %v", got.Shape, want.Shape)
	}
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("[%d] = %v, want %v", i, got.Data[i], want.Data[i])
		}
	}
}

func TestMatMul4D(t *testing.T) {
	// (B,H,M,K) @ (B,H,K,N)
	a := randTensor(3, 2, 3, 4, 5)
	b := randTensor(4, 2, 3, 5, 2)
	got := a.MatMul(b)
	if !shapesEqual(got.Shape, []int{2, 3, 4, 2}) {
		t.Fatalf("shape %v, want [2 3 4 2]", got.Shape)
	}
	// Check one slice against 2D matmul.
	a2 := New(append([]float64(nil), a.Data[1*3*4*5+2*4*5:1*3*4*5+3*4*5]...), 4, 5)
	b2 := New(append([]float64(nil), b.Data[1*3*5*2+2*5*2:1*3*5*2+3*5*2]...), 5, 2)
	want := a2.MatMul(b2)
	off := (1*3 + 2) * 4 * 2
	for i := range want.Data {
		if got.Data[off+i] != want.Data[i] {
			t.Fatalf("slice[%d] = %v, want %v", i, got.Data[off+i], want.Data[i])
		}
	}
}

func TestMatMulBroadcastBatch(t *testing.T) {
	// (B,M,K) @ (K,N): the 2D right operand broadcasts across the batch.
	a := randTensor(5, 3, 2, 4)
	w := randTensor(6, 4, 6)
	got := a.MatMul(w)
	if !shapesEqual(got.Shape, []int{3, 2, 6}) {
		t.Fatalf("shape %v, want [3 2 6]", got.Shape)
	}
	for bi := 0; bi < 3; bi++ {
		as := New(append([]float64(nil), a.Data[bi*2*4:(bi+1)*2*4]...), 2, 4)
		want := as.MatMul(w)
		for i := range want.Data {
			if got.Data[bi*2*6+i] != want.Data[i] {
				t.Fatalf("batch %d [%d] = %v, want %v", bi, i, got.Data[bi*2*6+i], want.Data[i])
			}
		}
	}

	// (1,H,M,K) @ (B,1,K,N) -> (B,H,M,N) full two-sided broadcast.
	x := randTensor(7, 1, 2, 3, 4)
	y := randTensor(8, 3, 1, 4, 5)
	z := x.MatMul(y)
	if !shapesEqual(z.Shape, []int{3, 2, 3, 5}) {
		t.Fatalf("shape %v, want [3 2 3 5]", z.Shape)
	}
}

// matMulGradCheck verifies batched-matmul grads against central differences.
func matMulGradCheck(t *testing.T, name string, a, b *Tensor) {
	t.Helper()
	a.SetRequiresGrad(true)
	b.SetRequiresGrad(true)
	loss := func() *Tensor { return a.MatMul(b).Square().Mean() }

	a.ZeroGrad()
	b.ZeroGrad()
	loss().Backward()
	analytic := [][]float64{
		append([]float64(nil), a.Grad.Data...),
		append([]float64(nil), b.Grad.Data...),
	}
	const eps, tol = 1e-6, 1e-6
	for wi, w := range []*Tensor{a, b} {
		for j := range w.Data {
			orig := w.Data[j]
			w.Data[j] = orig + eps
			fp := loss().Item()
			w.Data[j] = orig - eps
			fm := loss().Item()
			w.Data[j] = orig
			num := (fp - fm) / (2 * eps)
			got := analytic[wi][j]
			if math.Abs(num-got) > tol*math.Max(1, math.Abs(num)) {
				t.Fatalf("%s: wrt[%d][%d]: analytic %v, numeric %v", name, wi, j, got, num)
			}
		}
	}
}

func TestBatchedMatMulGrad(t *testing.T) {
	matMulGradCheck(t, "bmm", randTensor(10, 2, 3, 4), randTensor(11, 2, 4, 2))
	matMulGradCheck(t, "4d", randTensor(12, 2, 2, 3, 2), randTensor(13, 2, 2, 2, 3))
	matMulGradCheck(t, "broadcast-right-2d", randTensor(14, 3, 2, 4), randTensor(15, 4, 3))
	matMulGradCheck(t, "broadcast-batch", randTensor(16, 1, 2, 3, 4), randTensor(17, 3, 1, 4, 2))
	matMulGradCheck(t, "2d-regression", randTensor(18, 3, 4), randTensor(19, 4, 2))
}

func TestBMMShapeErrors(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func()
	}{
		{"rank", func() { Zeros(2, 3).BMM(Zeros(2, 3, 4)) }},
		{"batch", func() { Zeros(2, 3, 4).BMM(Zeros(3, 4, 5)) }},
		{"inner", func() { Zeros(2, 3, 4).BMM(Zeros(2, 5, 6)) }},
		{"matmul-1d", func() { Zeros(3).MatMul(Zeros(3, 4)) }},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic", c.name)
				}
			}()
			c.f()
		}()
	}
}
