//go:build cuda
// +build cuda

package nn

import "gonn/backend/cuda"

// fusedAttnAvailable reports whether the fused CUDA flash-attention kernel is
// compiled in (requires -tags cuda).
func fusedAttnAvailable() bool { return true }

// fusedAttnF64 runs the fused flash-attention forward on the GPU. Q,K,V are
// flat (BH, S, D) row-major float64; returns O with the same layout.
func fusedAttnF64(Q, K, V []float64, BH, S, D int, scale float64, causal bool) []float64 {
	O := make([]float64, len(Q))
	cuda.FlashAttnF64(Q, K, V, O, BH, S, D, scale, causal)
	return O
}
