//go:build cuda
// +build cuda

package nn

import (
	"gonn/backend/cuda"
	"gonn/tensor"
)

// fusedAttnAvailable reports whether the fused CUDA flash-attention kernel is
// compiled in (requires -tags cuda).
func fusedAttnAvailable() bool { return true }

// fusedAttnF64 runs the fused flash-attention forward on the GPU (inference,
// no autograd). Q,K,V are flat (BH, S, D) row-major float64; returns O.
func fusedAttnF64(Q, K, V []float64, BH, S, D int, scale float64, causal bool) []float64 {
	O := make([]float64, len(Q))
	cuda.FlashAttnF64(Q, K, V, O, BH, S, D, scale, causal)
	return O
}

// fusedAttnDiff is the differentiable fused attention core. qp,kp,vp are
// (B,H,S,D) tensors whose data is contiguous (B*H, S, D). The returned tensor
// carries a custom autograd node that runs the fused GPU backward kernel.
func fusedAttnDiff(qp, kp, vp *tensor.Tensor, BH, S, D int, scale float64, causal bool) *tensor.Tensor {
	O, L := cuda.FlashAttnF64Fwd(qp.Data, kp.Data, vp.Data, BH, S, D, scale, causal)
	out := tensor.New(O, qp.Shape...)
	tensor.MakeNode(out, "FlashAttn", []*tensor.Tensor{qp, kp, vp}, func(grad *tensor.Tensor) []*tensor.Tensor {
		dQ, dK, dV := cuda.FlashAttnF64Bwd(qp.Data, kp.Data, vp.Data, O, L, grad.Data, BH, S, D, scale, causal)
		return []*tensor.Tensor{
			tensor.New(dQ, qp.Shape...),
			tensor.New(dK, kp.Shape...),
			tensor.New(dV, vp.Shape...),
		}
	})
	return out
}
