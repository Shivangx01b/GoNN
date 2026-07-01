package optim

import (
	"math"

	"gonn/tensor"
)

// Gradient clipping utilities. These are free functions rather than optimizer
// options because clipping is a training-loop concern orthogonal to the
// optimizer choice (PyTorch draws the same line), and callers usually want
// the returned pre-clip norm for logging. Call between Backward() and Step():
//
//	loss.Backward()
//	optim.ClipGradNorm(opt.Parameters(), 1.0)
//	opt.Step()

// ClipGradNorm rescales all gradients in place so that their global L2 norm
// is at most maxNorm, and returns the total norm measured BEFORE clipping.
// Mirrors torch.nn.utils.clip_grad_norm_: scale = maxNorm/(totalNorm+1e-6),
// applied only when it is < 1. nil params and nil grads are skipped.
// Equivalent to TotalGradNorm followed by ClipGradsWithNorm.
func ClipGradNorm(params []*tensor.Tensor, maxNorm float64) float64 {
	totalNorm := TotalGradNorm(params)
	ClipGradsWithNorm(params, maxNorm, totalNorm)
	return totalNorm
}

// TotalGradNorm returns the global L2 norm over all gradients — the same
// norm ClipGradNorm computes and returns, standalone. Mirrors
// torch.nn.utils.get_total_norm over parameter gradients (L2 only; GoNN
// does not take a norm-type argument). nil params and nil grads are skipped.
func TotalGradNorm(params []*tensor.Tensor) float64 {
	var sq float64
	for _, p := range params {
		if p == nil || p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data {
			sq += g * g
		}
	}
	return math.Sqrt(sq)
}

// ClipGradsWithNorm rescales all gradients in place given a PRECOMPUTED
// total norm, mirroring torch.nn.utils.clip_grads_with_norm_:
// scale = maxNorm/(totalNorm+1e-6), clamped to <= 1 so gradients are never
// amplified. Pair with TotalGradNorm to decouple norm measurement from
// clipping (e.g. measuring once across gradient-accumulation steps).
// nil params and nil grads are skipped.
func ClipGradsWithNorm(params []*tensor.Tensor, maxNorm, totalNorm float64) {
	scale := maxNorm / (totalNorm + 1e-6)
	if scale >= 1 {
		return
	}
	for _, p := range params {
		if p == nil || p.Grad == nil {
			continue
		}
		for i := range p.Grad.Data {
			p.Grad.Data[i] *= scale
		}
	}
}

// ClipGradValue clamps every gradient element in place to the range
// [-clipValue, clipValue]. nil params and nil grads are skipped.
func ClipGradValue(params []*tensor.Tensor, clipValue float64) {
	for _, p := range params {
		if p == nil || p.Grad == nil {
			continue
		}
		for i, g := range p.Grad.Data {
			if g > clipValue {
				p.Grad.Data[i] = clipValue
			} else if g < -clipValue {
				p.Grad.Data[i] = -clipValue
			}
		}
	}
}
