package nn

import (
	"math"

	"gonn/tensor"
)

// MSELoss returns mean((pred-target)^2) as a scalar tensor.
func MSELoss(pred, target *tensor.Tensor) *tensor.Tensor {
	diff := pred.Sub(target)
	return diff.Square().Mean()
}

// MAELoss returns mean(|pred-target|) as a scalar tensor.
func MAELoss(pred, target *tensor.Tensor) *tensor.Tensor {
	return pred.Sub(target).Abs().Mean()
}

// HuberLoss returns the smooth-L1-like Huber loss with threshold delta.
func HuberLoss(pred, target *tensor.Tensor, delta float64) *tensor.Tensor {
	diff := pred.Sub(target)
	absD := diff.Abs()
	// quadratic: 0.5 * diff^2; linear: delta*(|diff|-0.5*delta)
	// Use a smooth blend: 0.5*min(|d|,delta)^2 + delta*max(|d|-delta, 0)
	clipped := absD.Clip(0, delta)
	quad := clipped.Square().MulScalar(0.5)
	excess := absD.Sub(tensor.Scalar(delta)).Clip(0, math.Inf(1)).MulScalar(delta)
	return quad.Add(excess).Mean()
}

// CrossEntropyLoss = NLLLoss(logSoftmax(logits), targets).
// logits: (N, C); targets: (N,) integer class indices stored as float64.
func CrossEntropyLoss(logits, targets *tensor.Tensor) *tensor.Tensor {
	logProbs := logits.LogSoftmax(1)
	return NLLLoss(logProbs, targets)
}

// NLLLoss negative log likelihood over class indices.
// logProbs: (N, C); targets: (N,).
func NLLLoss(logProbs, targets *tensor.Tensor) *tensor.Tensor {
	N, C := logProbs.Shape[0], logProbs.Shape[1]
	if len(targets.Data) != N {
		panic("NLLLoss: targets must have N entries")
	}
	// Build (N, C) one-hot mask, then -mean(sum(mask * logProbs, dim=1)).
	mask := tensor.Zeros(N, C)
	for i, v := range targets.Data {
		idx := int(v)
		if idx < 0 || idx >= C {
			panic("NLLLoss: target out of range")
		}
		mask.Data[i*C+idx] = 1
	}
	picked := logProbs.Mul(mask).SumAxis(1, false) // (N,)
	return picked.Mean().Neg()
}

// BCELoss expects pred in [0,1]. Returns -mean(t*log(p) + (1-t)*log(1-p)).
func BCELoss(pred, target *tensor.Tensor) *tensor.Tensor {
	one := tensor.Scalar(1)
	a := target.Mul(pred.Log())
	b := one.Sub(target).Mul(one.Sub(pred).Log())
	return a.Add(b).Mean().Neg()
}

// BCEWithLogitsLoss is the numerically stable version using softplus.
// loss = mean( max(z,0) - z*t + log(1+exp(-|z|)) )
func BCEWithLogitsLoss(logits, target *tensor.Tensor) *tensor.Tensor {
	// max(z,0) = ReLU(z); log(1+exp(-|z|)) = softplus(-|z|).
	maxPart := logits.ReLU()
	zt := logits.Mul(target)
	sp := logits.Abs().Neg().Softplus()
	return maxPart.Sub(zt).Add(sp).Mean()
}

// KLDivLoss: input is log-probabilities, target is probabilities.
// KL(target || input) = sum(target * (log(target) - input)). We omit the
// target*log(target) entropy term (constant w.r.t. params) and return
// mean( target * (log(target) - input) ).
func KLDivLoss(input, target *tensor.Tensor) *tensor.Tensor {
	// To avoid log(0), clip target above 1e-12 before log.
	tClipped := target.Clip(1e-12, math.Inf(1))
	return target.Mul(tClipped.Log().Sub(input)).Mean()
}
