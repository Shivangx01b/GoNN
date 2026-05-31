//go:build cuda
// +build cuda

// Device-resident buffers: run an MLP forward entirely on the GPU (upload once,
// chain matmul/relu on device, download once) vs the eager per-call backend
// (every op copies host<->device). Plus an fp16 tensor-core GEMM benchmark.
//
//	go run -tags cuda ./benchmark/resident
package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"gonn/backend"
	"gonn/backend/cuda"
)

func randv(n int, seed uint64) []float64 {
	d := make([]float64, n)
	s := seed
	for i := range d {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		d[i] = float64(int64(s%2000))/1000.0 - 1.0
	}
	return d
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if v := math.Abs(a[i] - b[i]); v > m {
			m = v
		}
	}
	return m
}

func median(ds []time.Duration) float64 {
	for i := 1; i < len(ds); i++ {
		for j := i; j > 0 && ds[j] < ds[j-1]; j-- {
			ds[j], ds[j-1] = ds[j-1], ds[j]
		}
	}
	return float64(ds[len(ds)/2].Microseconds()) / 1000.0
}

func main() {
	b, err := cuda.Backend()
	if err != nil {
		fmt.Println("CUDA unavailable:", err)
		os.Exit(1)
	}
	backend.Use(b)
	be := backend.Current()

	// MLP: X(B,D0) -> W1(D0,D1) -> relu -> W2(D1,D2) -> relu -> W3(D2,D3)
	const B, D0, D1, D2, D3 = 256, 1024, 1024, 1024, 1024
	X := randv(B*D0, 1)
	W1 := randv(D0*D1, 2)
	W2 := randv(D1*D2, 3)
	W3 := randv(D2*D3, 4)

	// --- eager per-call forward (each op copies H2D/D2H) ---
	eager := func() []float64 {
		h1 := be.ReLU(be.MatMul(X, W1, B, D0, D1))
		h2 := be.ReLU(be.MatMul(h1, W2, B, D1, D2))
		return be.MatMul(h2, W3, B, D2, D3)
	}

	// --- device-resident forward (weights uploaded once, kept on device) ---
	dW1 := cuda.DevUpload(W1)
	dW2 := cuda.DevUpload(W2)
	dW3 := cuda.DevUpload(W3)
	dH1 := cuda.DevAlloc(B * D1)
	dH2 := cuda.DevAlloc(B * D2)
	dOut := cuda.DevAlloc(B * D3)
	defer func() {
		dW1.Free(); dW2.Free(); dW3.Free(); dH1.Free(); dH2.Free(); dOut.Free()
	}()
	resident := func() []float64 {
		dX := cuda.DevUpload(X) // only the input crosses the bus each call
		defer dX.Free()
		cuda.DevMatMul(dX, dW1, dH1, B, D0, D1)
		cuda.DevReLU(dH1, B*D1)
		cuda.DevMatMul(dH1, dW2, dH2, B, D1, D2)
		cuda.DevReLU(dH2, B*D2)
		cuda.DevMatMul(dH2, dW3, dOut, B, D2, D3)
		cuda.DevSync()
		return dOut.Download()
	}

	// correctness
	oe := eager()
	or := resident()
	diff := maxAbsDiff(oe, or)
	fmt.Printf("=== MLP forward: resident vs eager  maxAbsDiff=%.3e  [%s] ===\n",
		diff, map[bool]string{true: "OK", false: "FAIL"}[diff < 1e-9])
	if diff >= 1e-9 {
		os.Exit(1)
	}

	// speed
	const iters = 30
	durs := make([]time.Duration, iters)
	for i := 0; i < 3; i++ {
		eager()
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		eager()
		durs[i] = time.Since(t0)
	}
	ems := median(durs)
	for i := 0; i < 3; i++ {
		resident()
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		resident()
		durs[i] = time.Since(t0)
	}
	rms := median(durs)
	fmt.Printf("  eager (per-call H2D/D2H): %7.3f ms/forward\n", ems)
	fmt.Printf("  device-resident:          %7.3f ms/forward   (%.1fx faster)\n", rms, ems/rms)

	// --- fp16 tensor-core GEMM vs fp32/fp64 (resident, event-timed) ---
	fmt.Println("=== resident GEMM dtype comparison (N=2048, GFLOP/s) ===")
	N, it := 2048, 50
	f64ms := cuda.BenchMatMulDev(N, N, N, it, false)
	f32ms := cuda.BenchMatMulDev(N, N, N, it, true)
	f16ms := cuda.BenchMatMulF16Dev(N, N, N, it)
	gf := func(ms float64) float64 { return 2.0 * float64(N) * float64(N) * float64(N) / 1e9 / (ms / 1000.0) }
	fmt.Printf("  f64:               %.1f GFLOP/s\n", gf(f64ms))
	fmt.Printf("  f32:               %.1f GFLOP/s\n", gf(f32ms))
	fmt.Printf("  f16 (tensor core): %.1f GFLOP/s\n", gf(f16ms))
	fmt.Println("RESIDENT: ALL OK")
}
