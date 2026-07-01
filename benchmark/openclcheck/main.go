//go:build opencl
// +build opencl

// Correctness check for the OpenCL backend. Runs ops through
// backend.Current() (the OpenCL backend) and compares against a pure-Go
// reference, including the batched/transposed GEMM paths. Exits non-zero on
// mismatch.
//
//	go run -tags opencl ./benchmark/openclcheck
package main

import (
	"fmt"
	"math"
	"os"

	"gonn/backend"
	"gonn/backend/opencl"
)

// refGemm is a naive reference for row-major batched C = op(A) @ op(B).
func refGemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	out := make([]float64, batch*m*n)
	at := func(bi, i, j int) float64 {
		if transA {
			return a[bi*m*k+j*m+i]
		}
		return a[bi*m*k+i*k+j]
	}
	bt := func(bi, i, j int) float64 {
		if transB {
			return b[bi*k*n+j*k+i]
		}
		return b[bi*k*n+i*n+j]
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

func maxAbsDiff(x, y []float64) float64 {
	d := 0.0
	for i := range x {
		if v := math.Abs(x[i] - y[i]); v > d {
			d = v
		}
	}
	return d
}

func check(name string, got, want []float64, tol float64) bool {
	d := maxAbsDiff(got, want)
	ok := d <= tol
	fmt.Printf("  %-24s maxAbsDiff=%.3e  tol=%.0e  [%s]\n", name, d, tol,
		map[bool]string{true: "OK", false: "FAIL"}[ok])
	return ok
}

func waveData(n int, freq float64) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = math.Sin(float64(i) * freq)
	}
	return d
}

func main() {
	b, err := opencl.Backend()
	if err != nil {
		fmt.Println("OpenCL backend unavailable:", err)
		os.Exit(1)
	}
	backend.Use(b)
	be := backend.Current()
	fmt.Println("Backend:", be.Name())

	ok := true

	// ---- GEMM: plain, trans, batched ----------------------------------------
	m, k, n := 64, 48, 32
	A := waveData(m*k, 0.1)
	B := waveData(k*n, 0.07)
	ok = check("gemm",
		be.Gemm(A, B, 1, m, k, n, false, false),
		refGemm(A, B, 1, m, k, n, false, false), 1e-9) && ok
	At := waveData(k*m, 0.11)
	Bt := waveData(n*k, 0.05)
	ok = check("gemm transA",
		be.Gemm(At, B, 1, m, k, n, true, false),
		refGemm(At, B, 1, m, k, n, true, false), 1e-9) && ok
	ok = check("gemm transB",
		be.Gemm(A, Bt, 1, m, k, n, false, true),
		refGemm(A, Bt, 1, m, k, n, false, true), 1e-9) && ok
	const batch = 3
	Ab := waveData(batch*m*k, 0.09)
	Bb := waveData(batch*k*n, 0.06)
	ok = check("gemm batched",
		be.Gemm(Ab, Bb, batch, m, k, n, false, false),
		refGemm(Ab, Bb, batch, m, k, n, false, false), 1e-9) && ok

	// ---- Elementwise capability ---------------------------------------------
	ew, has := be.(backend.Elementwiser)
	if !has {
		fmt.Println("backend does not implement Elementwiser [FAIL]")
		os.Exit(1)
	}
	x := []float64{-2, -0.5, 0, 0.5, 1, 2, 3, -3}
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	run1 := func(name string, kind backend.UnaryKind, in []float64, f func(float64) float64, tol float64) {
		got := make([]float64, len(in))
		if !ew.Unary(kind, in, got) {
			fmt.Printf("  %-24s declined [FAIL]\n", name)
			ok = false
			return
		}
		want := make([]float64, len(in))
		for i, v := range in {
			want[i] = f(v)
		}
		ok = check(name, got, want, tol) && ok
	}
	run1("relu", backend.UnaryReLU, x, func(v float64) float64 { return math.Max(v, 0) }, 1e-12)
	run1("sigmoid", backend.UnarySigmoid, x, func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }, 1e-12)
	run1("tanh", backend.UnaryTanh, x, math.Tanh, 1e-12)
	run1("exp", backend.UnaryExp, x, math.Exp, 1e-12)
	run1("log", backend.UnaryLog, []float64{0.1, 0.5, 1, 2, 5, 10, 0.01, 3}, math.Log, 1e-12)
	run1("gelu", backend.UnaryGELU, x, func(v float64) float64 {
		return 0.5 * v * (1 + math.Tanh(0.7978845608028654*(v+0.044715*v*v*v)))
	}, 1e-12)
	run1("silu", backend.UnarySiLU, x, func(v float64) float64 { return v / (1 + math.Exp(-v)) }, 1e-12)
	run1("gelu_exact", backend.UnaryGELUExact, x, func(v float64) float64 {
		return 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
	}, 1e-12)

	run2 := func(name string, kind backend.BinaryKind, f func(a, b float64) float64) {
		got := make([]float64, len(x))
		if !ew.Binary(kind, x, y, got) {
			fmt.Printf("  %-24s declined [FAIL]\n", name)
			ok = false
			return
		}
		want := make([]float64, len(x))
		for i := range x {
			want[i] = f(x[i], y[i])
		}
		ok = check(name, got, want, 1e-12) && ok
	}
	run2("add", backend.BinaryAdd, func(a, b float64) float64 { return a + b })
	run2("sub", backend.BinarySub, func(a, b float64) float64 { return a - b })
	run2("mul", backend.BinaryMul, func(a, b float64) float64 { return a * b })
	run2("div", backend.BinaryDiv, func(a, b float64) float64 { return a / b })

	if !ok {
		fmt.Println("OPENCL VERIFY: FAILED")
		os.Exit(1)
	}
	fmt.Println("OPENCL VERIFY: ALL OK on", be.Name())
	// Note: a throughput benchmark only makes sense on a GPU OpenCL device. This
	// check is meant to run on any OpenCL runtime (incl. the oclgrind simulator),
	// where timing reflects the interpreter, not hardware — so we don't print it.
}
