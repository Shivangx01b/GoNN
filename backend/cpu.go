package backend

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// cpuBackend implements only the required Backend core. It does not implement
// Elementwiser: the tensor package's pure-Go closures are the CPU elementwise
// path, so those ops are defined exactly once.
type cpuBackend struct{}

func (cpuBackend) Name() Device { return CPU }

// Gemm computes the row-major batched C = op(A) @ op(B) via gonum's BLAS
// Dgemm — a blocked, cache-aware, multi-threaded pure-Go kernel (~15x faster
// than a naive triple loop; a hand-tuned SIMD BLAS like MKL is still faster).
// Batches run as a loop of 2D GEMMs over contiguous slices, so batch == 1 is
// numerically identical to the historical 2D path.
func (cpuBackend) Gemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	out := make([]float64, batch*m*n)
	if batch == 0 || m == 0 || n == 0 || k == 0 {
		return out
	}
	// gonum takes the STORED matrix dims plus a trans flag; op(A) is (m,k) so
	// the stored A is (k,m) when transA, and likewise for B.
	ta, tb := blas.NoTrans, blas.NoTrans
	arows, acols := m, k
	if transA {
		ta = blas.Trans
		arows, acols = k, m
	}
	brows, bcols := k, n
	if transB {
		tb = blas.Trans
		brows, bcols = n, k
	}
	sa, sb, sc := m*k, k*n, m*n // per-batch element strides
	for bi := 0; bi < batch; bi++ {
		blas64.Gemm(ta, tb, 1,
			blas64.General{Rows: arows, Cols: acols, Stride: acols, Data: a[bi*sa : (bi+1)*sa]},
			blas64.General{Rows: brows, Cols: bcols, Stride: bcols, Data: b[bi*sb : (bi+1)*sb]},
			0,
			blas64.General{Rows: m, Cols: n, Stride: n, Data: out[bi*sc : (bi+1)*sc]},
		)
	}
	return out
}

func (cpuBackend) Synchronize() {}
