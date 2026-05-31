//go:build opencl
// +build opencl

// Correctness check + quick benchmark for the OpenCL backend. Runs ops through
// backend.Current() (the OpenCL backend) and compares against a pure-Go
// reference. Exits non-zero on mismatch.
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

func refMatMul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for kk := 0; kk < k; kk++ {
			x := a[i*k+kk]
			for j := 0; j < n; j++ {
				out[i*n+j] += x * b[kk*n+j]
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
	fmt.Printf("  %-16s maxAbsDiff=%.3e  tol=%.0e  [%s]\n", name, d, tol,
		map[bool]string{true: "OK", false: "FAIL"}[ok])
	return ok
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

	m, k, n := 64, 48, 32
	A := make([]float64, m*k)
	B := make([]float64, k*n)
	for i := range A {
		A[i] = math.Sin(float64(i) * 0.1)
	}
	for i := range B {
		B[i] = math.Cos(float64(i) * 0.07)
	}
	ok = check("matmul", be.MatMul(A, B, m, k, n), refMatMul(A, B, m, k, n), 1e-9) && ok

	x := []float64{-2, -0.5, 0, 0.5, 1, 2, 3, -3}
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	add := make([]float64, len(x))
	mul := make([]float64, len(x))
	sub := make([]float64, len(x))
	relu := make([]float64, len(x))
	sig := make([]float64, len(x))
	for i := range x {
		add[i] = x[i] + y[i]
		mul[i] = x[i] * y[i]
		sub[i] = x[i] - y[i]
		if x[i] > 0 {
			relu[i] = x[i]
		}
		sig[i] = 1.0 / (1.0 + math.Exp(-x[i]))
	}
	ok = check("add", be.AddElem(x, y), add, 1e-12) && ok
	ok = check("mul", be.MulElem(x, y), mul, 1e-12) && ok
	ok = check("sub", be.Sub(x, y), sub, 1e-12) && ok
	ok = check("relu", be.ReLU(x), relu, 1e-12) && ok
	ok = check("sigmoid", be.Sigmoid(x), sig, 1e-12) && ok

	sum, mx := 0.0, math.Inf(-1)
	for _, v := range x {
		sum += v
		if v > mx {
			mx = v
		}
	}
	if d := math.Abs(be.Sum(x) - sum); d > 1e-9 {
		fmt.Printf("  %-16s diff=%.3e [FAIL]\n", "sum", d)
		ok = false
	} else {
		fmt.Printf("  %-16s diff=%.3e [OK]\n", "sum", d)
	}
	if d := math.Abs(be.Max(x) - mx); d > 1e-12 {
		fmt.Printf("  %-16s diff=%.3e [FAIL]\n", "max", d)
		ok = false
	} else {
		fmt.Printf("  %-16s diff=%.3e [OK]\n", "max", d)
	}

	if !ok {
		fmt.Println("OPENCL VERIFY: FAILED")
		os.Exit(1)
	}
	fmt.Println("OPENCL VERIFY: ALL OK on", be.Name())
	// Note: a throughput benchmark only makes sense on a GPU OpenCL device. This
	// check is meant to run on any OpenCL runtime (incl. the oclgrind simulator),
	// where timing reflects the interpreter, not hardware — so we don't print it.
}
