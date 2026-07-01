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
func ClipGradNorm(params []*tensor.Tensor, maxNorm float64) float64 {
	var sq float64
	for _, p := range params {
		if p == nil || p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data {
			sq += g * g
		}
	}
	totalNorm := math.Sqrt(sq)
	scale := maxNorm / (totalNorm + 1e-6)
	if scale < 1 {
		for _, p := range params {
			if p == nil || p.Grad == nil {
				continue
			}
			for i := range p.Grad.Data {
				p.Grad.Data[i] *= scale
			}
		}
	}
	return totalNorm
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
