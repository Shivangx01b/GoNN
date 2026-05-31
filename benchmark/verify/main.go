// Correctness check for the active compute backend (intended for CUDA).
// Builds small inputs, runs ops through backend.Current(), and compares against
// an independent pure-Go reference. Exits non-zero on mismatch.
//
//	go run -tags cuda ./benchmark/verify
package main

import (
	"fmt"
	"math"
	"os"

	"gonn/backend"
	"gonn/backend/cuda"
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
	status := "OK"
	ok := d <= tol
	if !ok {
		status = "FAIL"
	}
	fmt.Printf("  %-16s maxAbsDiff=%.3e  tol=%.0e  [%s]\n", name, d, tol, status)
	return ok
}

func main() {
	b, err := cuda.Backend()
	if err != nil {
		fmt.Println("CUDA backend unavailable:", err)
		os.Exit(1)
	}
	backend.Use(b)
	be := backend.Current()
	fmt.Println("Backend:", be.Name())

	allOK := true

	// matmul 64x48 @ 48x32
	m, k, n := 64, 48, 32
	A := make([]float64, m*k)
	B := make([]float64, k*n)
	for i := range A {
		A[i] = math.Sin(float64(i) * 0.1)
	}
	for i := range B {
		B[i] = math.Cos(float64(i) * 0.07)
	}
	allOK = check("matmul", be.MatMul(A, B, m, k, n), refMatMul(A, B, m, k, n), 1e-9) && allOK

	// elementwise
	x := []float64{-2, -0.5, 0, 0.5, 1, 2, 3, -3}
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	add := make([]float64, len(x))
	mul := make([]float64, len(x))
	relu := make([]float64, len(x))
	sig := make([]float64, len(x))
	for i := range x {
		add[i] = x[i] + y[i]
		mul[i] = x[i] * y[i]
		if x[i] > 0 {
			relu[i] = x[i]
		}
		sig[i] = 1.0 / (1.0 + math.Exp(-x[i]))
	}
	allOK = check("add", be.AddElem(x, y), add, 1e-12) && allOK
	allOK = check("mul", be.MulElem(x, y), mul, 1e-12) && allOK
	allOK = check("relu", be.ReLU(x), relu, 1e-12) && allOK
	allOK = check("sigmoid", be.Sigmoid(x), sig, 1e-12) && allOK

	// reductions
	sum := 0.0
	mx := math.Inf(-1)
	for _, v := range x {
		sum += v
		if v > mx {
			mx = v
		}
	}
	if d := math.Abs(be.Sum(x) - sum); d > 1e-12 {
		fmt.Printf("  %-16s diff=%.3e  [FAIL]\n", "sum", d)
		allOK = false
	} else {
		fmt.Printf("  %-16s diff=%.3e  [OK]\n", "sum", d)
	}
	if d := math.Abs(be.Max(x) - mx); d > 1e-12 {
		fmt.Printf("  %-16s diff=%.3e  [FAIL]\n", "max", d)
		allOK = false
	} else {
		fmt.Printf("  %-16s diff=%.3e  [OK]\n", "max", d)
	}

	if !allOK {
		fmt.Println("VERIFY: FAILED")
		os.Exit(1)
	}
	fmt.Println("VERIFY: ALL OK on", be.Name())
}
