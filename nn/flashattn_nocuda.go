//go:build !cuda
// +build !cuda

package nn

import "gonn/tensor"

// fusedAttnAvailable reports whether the fused CUDA flash-attention kernel is
// compiled in. False on the default (CPU) build.
func fusedAttnAvailable() bool { return false }

// fusedAttnF64 is unavailable without the cuda build tag.
func fusedAttnF64(Q, K, V []float64, BH, S, D int, scale float64, causal bool) []float64 {
	panic("nn: fused attention requires building with -tags cuda")
}

// fusedAttnDiff is unavailable without the cuda build tag.
func fusedAttnDiff(qp, kp, vp *tensor.Tensor, BH, S, D int, scale float64, causal bool) *tensor.Tensor {
	panic("nn: fused attention requires building with -tags cuda")
}
