//go:build cuda
// +build cuda

// Integration check + benchmark for MultiHeadAttention.ForwardFused (GPU fused
// flash-attention kernel) vs the regular autograd Forward path.
//
//	go run -tags cuda ./benchmark/mha
package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"gonn/nn"
	"gonn/tensor"
)

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if v := math.Abs(a[i] - b[i]); v > m {
			m = v
		}
	}
	return m
}

func randT(shape ...int) *tensor.Tensor {
	t := tensor.Zeros(shape...)
	var s uint64 = 0x2545f4914f6cdd1d
	for i := range t.Data {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		t.Data[i] = float64(int64(s%2000))/1000.0 - 1.0
	}
	return t
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
	const E, H = 512, 8 // headDim = 64
	mha := nn.NewMultiHeadAttention(E, H)

	// ---- correctness: ForwardFused vs Forward ----
	fmt.Println("=== MHA ForwardFused vs Forward (correctness) ===")
	ok := true
	for _, causal := range []bool{false, true} {
		x := randT(2, 64, E)
		ref := mha.Forward(x, x, x, causal)
		fused := mha.ForwardFused(x, x, x, causal)
		diff := maxAbsDiff(ref.Data, fused.Data)
		status := "OK"
		if diff > 1e-9 {
			status = "FAIL"
			ok = false
		}
		fmt.Printf("  causal=%-5v  maxAbsDiff=%.3e  [%s]\n", causal, diff, status)
	}
	if !ok {
		fmt.Println("MHA FUSED CORRECTNESS FAILED")
		os.Exit(1)
	}

	// ---- speed: Forward vs ForwardFused (modest size so the regular path,
	// whose selector-based batched matmul materializes ~B*H full (S,S) tensors
	// in the autograd graph, does not exhaust memory) ----
	fmt.Println("=== MHA speed: Forward vs ForwardFused (B=2, S=256, E=512, H=8, causal) ===")
	x := randT(2, 256, E)
	const iters = 10

	durs := make([]time.Duration, iters)
	for i := 0; i < 2; i++ {
		mha.Forward(x, x, x, true)
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		mha.Forward(x, x, x, true)
		durs[i] = time.Since(t0)
	}
	regMs := median(durs)

	for i := 0; i < 3; i++ {
		mha.ForwardFused(x, x, x, true)
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		mha.ForwardFused(x, x, x, true)
		durs[i] = time.Since(t0)
	}
	fusedMs := median(durs)

	fmt.Printf("  Forward       (autograd, batched-matmul core): %8.3f ms\n", regMs)
	fmt.Printf("  ForwardFused  (fused GPU flash-attn core):      %8.3f ms\n", fusedMs)
	fmt.Printf("  speedup: %.1fx\n", regMs/fusedMs)

	// fused-only at a larger size the regular path cannot handle (OOM).
	fmt.Println("=== ForwardFused at B=8, S=512 (regular Forward OOMs here) ===")
	xl := randT(8, 512, E)
	for i := 0; i < 3; i++ {
		mha.ForwardFused(xl, xl, xl, true)
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		mha.ForwardFused(xl, xl, xl, true)
		durs[i] = time.Since(t0)
	}
	fmt.Printf("  ForwardFused B=8,S=512 causal: %.3f ms/call\n", median(durs))
}
