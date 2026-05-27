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

// SmoothL1Loss is the Smooth-L1 / Huber-style loss with transition point beta.
// It is mathematically identical to HuberLoss with delta=beta.
func SmoothL1Loss(pred, target *tensor.Tensor, beta float64) *tensor.Tensor {
	return HuberLoss(pred, target, beta)
}

// L1Loss is an alias for MAELoss (mean absolute error).
func L1Loss(pred, target *tensor.Tensor) *tensor.Tensor { return MAELoss(pred, target) }

// PoissonNLLLoss: negative log likelihood of a Poisson distribution.
// If logInput is true: loss = mean(exp(input) - target*input).
// Else                : loss = mean(input - target*log(input + eps)).
func PoissonNLLLoss(input, target *tensor.Tensor, logInput bool) *tensor.Tensor {
	if logInput {
		return input.Exp().Sub(target.Mul(input)).Mean()
	}
	const eps = 1e-8
	logTerm := input.AddScalar(eps).Log()
	return input.Sub(target.Mul(logTerm)).Mean()
}

// GaussianNLLLoss: negative log likelihood for a Gaussian with diagonal
// variance. loss = mean(0.5 * (log(varT) + (input - target)^2 / varT)).
// varT must be strictly positive.
func GaussianNLLLoss(input, target, varT *tensor.Tensor) *tensor.Tensor {
	diff := input.Sub(target)
	sq := diff.Square()
	return varT.Log().Add(sq.Div(varT)).MulScalar(0.5).Mean()
}

// MarginRankingLoss: mean(max(0, -y*(x1-x2) + margin)).
// y should contain +/-1 entries (broadcast-compatible with x1, x2).
func MarginRankingLoss(x1, x2, y *tensor.Tensor, margin float64) *tensor.Tensor {
	diff := x1.Sub(x2)
	score := y.Mul(diff).Neg().AddScalar(margin) // -y*(x1-x2) + margin
	return score.Clip(0, math.Inf(1)).Mean()
}

// HingeEmbeddingLoss: y=1 -> loss=x; y=-1 -> loss=max(0, margin-x). Mean.
// Implemented as: posMask*x + negMask*max(0, margin-x), where the masks come
// from y itself: posMask = (y+1)/2, negMask = (1-y)/2.
func HingeEmbeddingLoss(x, y *tensor.Tensor, margin float64) *tensor.Tensor {
	posMask := y.AddScalar(1).MulScalar(0.5)
	negMask := y.Neg().AddScalar(1).MulScalar(0.5)
	negTerm := x.Neg().AddScalar(margin).Clip(0, math.Inf(1))
	return posMask.Mul(x).Add(negMask.Mul(negTerm)).Mean()
}

// CosineEmbeddingLoss: with cosine similarity cs = (x1.x2)/(||x1||*||x2||),
// loss = mean( y=1  ? 1 - cs
//              y=-1 ? max(0, cs - margin) ).
// x1, x2 are (N, D); y is (N,).
func CosineEmbeddingLoss(x1, x2, y *tensor.Tensor, margin float64) *tensor.Tensor {
	const eps = 1e-8
	dot := x1.Mul(x2).SumAxis(1, false)                            // (N,)
	n1 := x1.Square().SumAxis(1, false).AddScalar(eps).Sqrt()      // (N,)
	n2 := x2.Square().SumAxis(1, false).AddScalar(eps).Sqrt()      // (N,)
	cs := dot.Div(n1.Mul(n2))                                      // (N,)
	posMask := y.AddScalar(1).MulScalar(0.5)                       // 1 when y=1
	negMask := y.Neg().AddScalar(1).MulScalar(0.5)                 // 1 when y=-1
	posTerm := cs.Neg().AddScalar(1)                               // 1 - cs
	negTerm := cs.SubScalar(margin).Clip(0, math.Inf(1))           // max(0, cs - margin)
	return posMask.Mul(posTerm).Add(negMask.Mul(negTerm)).Mean()
}

// TripletMarginLoss: mean(max(0, ||a-p||_2 - ||a-n||_2 + margin)).
// anchor, positive, negative share shape (N, D).
func TripletMarginLoss(anchor, positive, negative *tensor.Tensor, margin float64) *tensor.Tensor {
	const eps = 1e-8
	dp := anchor.Sub(positive).Square().SumAxis(1, false).AddScalar(eps).Sqrt()
	dn := anchor.Sub(negative).Square().SumAxis(1, false).AddScalar(eps).Sqrt()
	return dp.Sub(dn).AddScalar(margin).Clip(0, math.Inf(1)).Mean()
}

// MultiMarginLoss: multi-class hinge loss.
// input: (N, C); target: (N,) integer class indices stored as float64.
// loss = mean over batch of mean over non-target classes j of
//        max(0, margin - input[target] + input[j]).
func MultiMarginLoss(input, target *tensor.Tensor, margin float64) *tensor.Tensor {
	if len(input.Shape) != 2 {
		panic("MultiMarginLoss: input must be 2D (N, C)")
	}
	N, C := input.Shape[0], input.Shape[1]
	if len(target.Data) != N {
		panic("MultiMarginLoss: target must have N entries")
	}
	// Pick the target score per row: x_t with shape (N, 1).
	mask := tensor.Zeros(N, C)
	for i, v := range target.Data {
		idx := int(v)
		if idx < 0 || idx >= C {
			panic("MultiMarginLoss: target out of range")
		}
		mask.Data[i*C+idx] = 1
	}
	xt := input.Mul(mask).SumAxis(1, true) // (N, 1)
	// margins[i, j] = margin - x_t[i] + input[i, j].
	margins := input.Sub(xt).AddScalar(margin).Clip(0, math.Inf(1)) // (N, C)
	// Zero out the target column so it doesn't contribute, then average over
	// the (C-1) non-target classes per row, then over the batch.
	nonTargetMask := tensor.Ones(N, C).Sub(mask) // 0 at target col, 1 elsewhere
	masked := margins.Mul(nonTargetMask)
	perRow := masked.SumAxis(1, false).DivScalar(float64(C - 1)) // (N,)
	return perRow.Mean()
}

// CTCLoss (Connectionist Temporal Classification) is intentionally not
// implemented in v1. The dynamic-programming forward-backward algorithm over
// the expanded label sequence with blank tokens and log-space accumulation
// requires significant additional infrastructure (a real log-sum-exp, custom
// autograd, masked alignment) that is out of scope here. Track this as a
// future addition.
