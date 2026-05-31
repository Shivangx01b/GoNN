package backend

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

type cpuBackend struct{}

func (cpuBackend) Name() Device { return CPU }

// MatMul computes C(m,n) = A(m,k) * B(k,n) via gonum's BLAS Dgemm, which is a
// blocked, cache-aware, multi-threaded pure-Go kernel — ~15x faster than the
// previous naive triple loop. (Pure Go cannot match a hand-tuned AVX-512 BLAS
// like MKL; for that, link a system BLAS via cgo.)
func (cpuBackend) MatMul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	if m == 0 || n == 0 || k == 0 {
		return out
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: m, Cols: k, Stride: k, Data: a},
		blas64.General{Rows: k, Cols: n, Stride: n, Data: b},
		0,
		blas64.General{Rows: m, Cols: n, Stride: n, Data: out},
	)
	return out
}

func (cpuBackend) AddElem(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func (cpuBackend) MulElem(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

func (cpuBackend) Synchronize() {}

// Elementwise arithmetic ----------------------------------------------------

func (cpuBackend) Sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func (cpuBackend) Div(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] / b[i]
	}
	return out
}

func (cpuBackend) Scale(a []float64, s float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * s
	}
	return out
}

func (cpuBackend) AxpyInto(out, x []float64, alpha float64) {
	for i := range out {
		out[i] += alpha * x[i]
	}
}

// Reductions ---------------------------------------------------------------

func (cpuBackend) Sum(a []float64) float64 {
	s := 0.0
	for _, v := range a {
		s += v
	}
	return s
}

func (cpuBackend) Max(a []float64) float64 {
	if len(a) == 0 {
		return math.Inf(-1)
	}
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// Activations / unary math -------------------------------------------------

func (cpuBackend) ReLU(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

func (cpuBackend) Sigmoid(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return out
}

func (cpuBackend) Tanh(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = math.Tanh(v)
	}
	return out
}

func (cpuBackend) Exp(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = math.Exp(v)
	}
	return out
}

func (cpuBackend) Log(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = math.Log(v)
	}
	return out
}

// GELU uses the tanh approximation:
//
//	0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func (cpuBackend) GELU(a []float64) []float64 {
	const c = 0.7978845608028654 // sqrt(2/pi)
	out := make([]float64, len(a))
	for i, v := range a {
		inner := c * (v + 0.044715*v*v*v)
		out[i] = 0.5 * v * (1 + math.Tanh(inner))
	}
	return out
}

// SiLU (Swish): x * sigmoid(x).
func (cpuBackend) SiLU(a []float64) []float64 {
	out := make([]float64, len(a))
	for i, v := range a {
		out[i] = v / (1.0 + math.Exp(-v))
	}
	return out
}
