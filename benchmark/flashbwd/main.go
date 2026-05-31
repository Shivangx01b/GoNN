//go:build cuda
// +build cuda

// Finite-difference gradient check for the fused flash-attention backward
// kernel, plus a tiny MHA training-loop sanity check (loss must decrease).
//
//	go run -tags cuda ./benchmark/flashbwd
package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"gonn/backend/cuda"
	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

func randv(n int) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = rand.NormFloat64()
	}
	return d
}

// loss = sum(O * dO) for fixed dO, so dLoss/dO = dO and the backward's dQ/dK/dV
// are exactly the gradients we finite-difference against.
func loss(O, dO []float64) float64 {
	s := 0.0
	for i := range O {
		s += O[i] * dO[i]
	}
	return s
}

func main() {
	rand.Seed(1)
	const BH, S, D = 2, 6, 64
	scale := 1.0 / math.Sqrt(float64(D))
	n := BH * S * D

	fail := false
	for _, causal := range []bool{false, true} {
		Q, K, V := randv(n), randv(n), randv(n)
		dO := randv(n)
		_, _ = cuda.FlashAttnF64Fwd(Q, K, V, BH, S, D, scale, causal) // warm
		dQ, dK, dV := func() ([]float64, []float64, []float64) {
			O, L := cuda.FlashAttnF64Fwd(Q, K, V, BH, S, D, scale, causal)
			return cuda.FlashAttnF64Bwd(Q, K, V, O, L, dO, BH, S, D, scale, causal)
		}()

		const eps = 1e-6
		numGrad := func(x []float64, idx int) float64 {
			save := x[idx]
			x[idx] = save + eps
			Op, _ := cuda.FlashAttnF64Fwd(Q, K, V, BH, S, D, scale, causal)
			lp := loss(Op, dO)
			x[idx] = save - eps
			Om, _ := cuda.FlashAttnF64Fwd(Q, K, V, BH, S, D, scale, causal)
			lm := loss(Om, dO)
			x[idx] = save
			return (lp - lm) / (2 * eps)
		}

		maxRel := 0.0
		check := func(name string, analytic, x []float64) {
			for _, idx := range []int{0, 1, 64, 130, 200, n - 1} {
				ng := numGrad(x, idx)
				ag := analytic[idx]
				den := math.Abs(ng) + math.Abs(ag) + 1e-9
				rel := math.Abs(ng-ag) / den
				if rel > maxRel {
					maxRel = rel
				}
				if rel > 1e-4 {
					fmt.Printf("  [%s causal=%v] idx=%d analytic=%.6e numeric=%.6e relErr=%.2e MISMATCH\n",
						name, causal, idx, ag, ng, rel)
					fail = true
				}
			}
		}
		check("dQ", dQ, Q)
		check("dK", dK, K)
		check("dV", dV, V)
		fmt.Printf("=== gradcheck causal=%-5v  maxRelErr=%.2e  [%s]\n", causal, maxRel,
			map[bool]string{true: "FAIL", false: "OK"}[maxRel > 1e-4])
	}
	if fail {
		fmt.Println("GRADCHECK FAILED")
		os.Exit(1)
	}

	// ---- training sanity: MHA self-attention should reduce a simple loss ----
	fmt.Println("=== MHA training with fused kernel (loss should drop) ===")
	const E, H = 64, 1 // head dim 64
	mha := nn.NewMultiHeadAttention(E, H)
	opt := optim.NewAdam(mha.Parameters(), 1e-2)
	x := tensor.New(randv(2*8*E), 2, 8, E)
	target := tensor.New(randv(2*8*E), 2, 8, E)
	var first, last float64
	for step := 0; step < 30; step++ {
		opt.ZeroGrad()
		out := mha.Forward(x, x, x, true) // uses fused diff path on GPU
		diff := out.Sub(target)
		l := diff.Mul(diff).Mean()
		l.Backward()
		opt.Step()
		if step == 0 {
			first = l.Data[0]
		}
		last = l.Data[0]
	}
	fmt.Printf("  loss: %.5f -> %.5f  (%s)\n", first, last,
		map[bool]string{true: "decreased", false: "DID NOT DECREASE"}[last < first])
	if last >= first {
		os.Exit(1)
	}
	fmt.Println("FLASH BACKWARD: ALL OK")
}
