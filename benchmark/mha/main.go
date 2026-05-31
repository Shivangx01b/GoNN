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

	// ---- speed: on a CUDA build, BOTH Forward (differentiable, training-capable)
	// and ForwardFused (inference-only) now run the GPU fused-attention core, so
	// they are comparable. Both handle B=8,S=512 — a size the previous
	// selector-based batched-matmul core could not (it materialized ~B*H full
	// (S,S) tensors in the autograd graph and OOM'd). ----
	const iters = 10
	durs := make([]time.Duration, iters)

	fmt.Println("=== MHA speed, B=8, S=512, E=512, H=8, causal (both fused on GPU) ===")
	xl := randT(8, 512, E)
	for i := 0; i < 2; i++ {
		mha.Forward(xl, xl, xl, true)
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		mha.Forward(xl, xl, xl, true)
		durs[i] = time.Since(t0)
	}
	fmt.Printf("  Forward      (differentiable fused core, trains): %8.3f ms\n", median(durs))

	for i := 0; i < 3; i++ {
		mha.ForwardFused(xl, xl, xl, true)
	}
	for i := 0; i < iters; i++ {
		t0 := time.Now()
		mha.ForwardFused(xl, xl, xl, true)
		durs[i] = time.Since(t0)
	}
	fmt.Printf("  ForwardFused (inference-only fused core):         %8.3f ms\n", median(durs))
}
