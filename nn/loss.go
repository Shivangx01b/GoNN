package nn

import (
	"math"

	"gonn/tensor"
)

// Losses are plain functions (PyTorch's functional forms): they compose in
// any training loop and don't need the Module machinery. Every loss accepts
// trailing LossOpt options; the default reduction is the mean, matching the
// historical behavior, so existing call sites are unchanged.

// Reduction selects how a loss aggregates its per-element (or per-sample)
// values.
type Reduction int

const (
	// ReduceMean averages the loss values (default).
	ReduceMean Reduction = iota
	// ReduceSum sums the loss values.
	ReduceSum
	// ReduceNone returns the unreduced loss tensor.
	ReduceNone
)

// LossOpt configures a loss function.
type LossOpt func(*lossOpts)

type lossOpts struct {
	reduction Reduction
}

// WithReduction selects mean (default), sum, or no reduction.
func WithReduction(r Reduction) LossOpt { return func(o *lossOpts) { o.reduction = r } }

// applyReduction reduces the raw loss tensor per the options.
func applyReduction(t *tensor.Tensor, opts []LossOpt) *tensor.Tensor {
	o := lossOpts{reduction: ReduceMean}
	for _, fn := range opts {
		fn(&o)
	}
	switch o.reduction {
	case ReduceSum:
		return t.Sum()
	case ReduceNone:
		return t
	default:
		return t.Mean()
	}
}

// MSELoss returns mean((pred-target)^2).
func MSELoss(pred, target *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	return applyReduction(pred.Sub(target).Square(), opts)
}

// MAELoss returns mean(|pred-target|).
func MAELoss(pred, target *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	return applyReduction(pred.Sub(target).Abs(), opts)
}

// L1Loss is an alias for MAELoss (mean absolute error).
func L1Loss(pred, target *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	return MAELoss(pred, target, opts...)
}

// HuberLoss returns the smooth-L1-like Huber loss with threshold delta.
func HuberLoss(pred, target *tensor.Tensor, delta float64, opts ...LossOpt) *tensor.Tensor {
	diff := pred.Sub(target)
	absD := diff.Abs()
	// quadratic: 0.5 * diff^2; linear: delta*(|diff|-0.5*delta)
	// Use a smooth blend: 0.5*min(|d|,delta)^2 + delta*max(|d|-delta, 0)
	clipped := absD.Clip(0, delta)
	quad := clipped.Square().MulScalar(0.5)
	excess := absD.Sub(tensor.Scalar(delta)).Clip(0, math.Inf(1)).MulScalar(delta)
	return applyReduction(quad.Add(excess), opts)
}

// SmoothL1Loss is the Smooth-L1 / Huber-style loss with transition point beta.
// It is mathematically identical to HuberLoss with delta=beta.
func SmoothL1Loss(pred, target *tensor.Tensor, beta float64, opts ...LossOpt) *tensor.Tensor {
	return HuberLoss(pred, target, beta, opts...)
}

// CrossEntropyLoss = NLLLoss(logSoftmax(logits), targets).
// logits: (N, C); targets: (N,) integer class indices stored as float64.
func CrossEntropyLoss(logits, targets *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	return NLLLoss(logits.LogSoftmax(1), targets, opts...)
}

// pickClass gathers input[i, targets[i]] as an (N, 1) tensor with autograd
// (Gather has a scatter-add backward) — O(N) instead of the historical
// O(N*C) one-hot mask.
func pickClass(op string, input, targets *tensor.Tensor) *tensor.Tensor {
	N, C := input.Shape[0], input.Shape[1]
	if len(targets.Data) != N {
		panic(op + ": targets must have N entries")
	}
	for _, v := range targets.Data {
		if idx := int(v); idx < 0 || idx >= C {
			panic(op + ": target out of range")
		}
	}
	return input.Gather(1, targets.Reshape(N, 1))
}

// NLLLoss negative log likelihood over class indices.
// logProbs: (N, C); targets: (N,).
func NLLLoss(logProbs, targets *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	N := logProbs.Shape[0]
	picked := pickClass("NLLLoss", logProbs, targets).Reshape(N) // (N,)
	return applyReduction(picked.Neg(), opts)
}

// BCELoss expects pred in [0,1]. Returns -mean(t*log(p) + (1-t)*log(1-p)).
// pred is clamped to [eps, 1-eps] to avoid log(0) -> -Inf / NaN gradients.
func BCELoss(pred, target *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	const eps = 1e-12
	pred = pred.Clip(eps, 1-eps)
	one := tensor.Scalar(1)
	a := target.Mul(pred.Log())
	b := one.Sub(target).Mul(one.Sub(pred).Log())
	return applyReduction(a.Add(b).Neg(), opts)
}

// BCEWithLogitsLoss is the numerically stable version using softplus.
// loss = mean( max(z,0) - z*t + log(1+exp(-|z|)) )
func BCEWithLogitsLoss(logits, target *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	// max(z,0) = ReLU(z); log(1+exp(-|z|)) = softplus(-|z|).
	maxPart := logits.ReLU()
	zt := logits.Mul(target)
	sp := logits.Abs().Neg().Softplus()
	return applyReduction(maxPart.Sub(zt).Add(sp), opts)
}

// KLDivLoss: input is log-probabilities, target is probabilities.
// KL(target || input) = sum(target * (log(target) - input)). We omit the
// target*log(target) entropy term (constant w.r.t. params) and return
// mean( target * (log(target) - input) ).
func KLDivLoss(input, target *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	// To avoid log(0), clip target above 1e-12 before log.
	tClipped := target.Clip(1e-12, math.Inf(1))
	return applyReduction(target.Mul(tClipped.Log().Sub(input)), opts)
}

// PoissonNLLLoss: negative log likelihood of a Poisson distribution.
// If logInput is true: loss = mean(exp(input) - target*input).
// Else                : loss = mean(input - target*log(input + eps)).
func PoissonNLLLoss(input, target *tensor.Tensor, logInput bool, opts ...LossOpt) *tensor.Tensor {
	if logInput {
		return applyReduction(input.Exp().Sub(target.Mul(input)), opts)
	}
	const eps = 1e-8
	logTerm := input.AddScalar(eps).Log()
	return applyReduction(input.Sub(target.Mul(logTerm)), opts)
}

// GaussianNLLLoss: negative log likelihood for a Gaussian with diagonal
// variance. loss = mean(0.5 * (log(varT) + (input - target)^2 / varT)).
// varT must be strictly positive.
func GaussianNLLLoss(input, target, varT *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	diff := input.Sub(target)
	sq := diff.Square()
	return applyReduction(varT.Log().Add(sq.Div(varT)).MulScalar(0.5), opts)
}

// MarginRankingLoss: mean(max(0, -y*(x1-x2) + margin)).
// y should contain +/-1 entries (broadcast-compatible with x1, x2).
func MarginRankingLoss(x1, x2, y *tensor.Tensor, margin float64, opts ...LossOpt) *tensor.Tensor {
	diff := x1.Sub(x2)
	score := y.Mul(diff).Neg().AddScalar(margin) // -y*(x1-x2) + margin
	return applyReduction(score.Clip(0, math.Inf(1)), opts)
}

// HingeEmbeddingLoss: y=1 -> loss=x; y=-1 -> loss=max(0, margin-x). Mean.
// Implemented as: posMask*x + negMask*max(0, margin-x), where the masks come
// from y itself: posMask = (y+1)/2, negMask = (1-y)/2.
func HingeEmbeddingLoss(x, y *tensor.Tensor, margin float64, opts ...LossOpt) *tensor.Tensor {
	posMask := y.AddScalar(1).MulScalar(0.5)
	negMask := y.Neg().AddScalar(1).MulScalar(0.5)
	negTerm := x.Neg().AddScalar(margin).Clip(0, math.Inf(1))
	return applyReduction(posMask.Mul(x).Add(negMask.Mul(negTerm)), opts)
}

// CosineEmbeddingLoss: with cosine similarity cs = (x1.x2)/(||x1||*||x2||),
// loss = mean( y=1  ? 1 - cs
//
//	y=-1 ? max(0, cs - margin) ).
//
// x1, x2 are (N, D); y is (N,).
func CosineEmbeddingLoss(x1, x2, y *tensor.Tensor, margin float64, opts ...LossOpt) *tensor.Tensor {
	const eps = 1e-8
	dot := x1.Mul(x2).SumAxis(1, false)                       // (N,)
	n1 := x1.Square().SumAxis(1, false).AddScalar(eps).Sqrt() // (N,)
	n2 := x2.Square().SumAxis(1, false).AddScalar(eps).Sqrt() // (N,)
	cs := dot.Div(n1.Mul(n2))                                 // (N,)
	posMask := y.AddScalar(1).MulScalar(0.5)                  // 1 when y=1
	negMask := y.Neg().AddScalar(1).MulScalar(0.5)            // 1 when y=-1
	posTerm := cs.Neg().AddScalar(1)                          // 1 - cs
	negTerm := cs.SubScalar(margin).Clip(0, math.Inf(1))      // max(0, cs - margin)
	return applyReduction(posMask.Mul(posTerm).Add(negMask.Mul(negTerm)), opts)
}

// TripletMarginLoss: mean(max(0, ||a-p||_2 - ||a-n||_2 + margin)).
// anchor, positive, negative share shape (N, D).
func TripletMarginLoss(anchor, positive, negative *tensor.Tensor, margin float64, opts ...LossOpt) *tensor.Tensor {
	const eps = 1e-8
	dp := anchor.Sub(positive).Square().SumAxis(1, false).AddScalar(eps).Sqrt()
	dn := anchor.Sub(negative).Square().SumAxis(1, false).AddScalar(eps).Sqrt()
	return applyReduction(dp.Sub(dn).AddScalar(margin).Clip(0, math.Inf(1)), opts)
}

// MultiMarginLoss: multi-class hinge loss.
// input: (N, C); target: (N,) integer class indices stored as float64.
// loss = mean over batch of mean over non-target classes j of
//
//	max(0, margin - input[target] + input[j]).
func MultiMarginLoss(input, target *tensor.Tensor, margin float64, opts ...LossOpt) *tensor.Tensor {
	if len(input.Shape) != 2 {
		panic("MultiMarginLoss: input must be 2D (N, C)")
	}
	N, C := input.Shape[0], input.Shape[1]
	xt := pickClass("MultiMarginLoss", input, target) // (N, 1)
	// margins[i, j] = margin - x_t[i] + input[i, j].
	margins := input.Sub(xt).AddScalar(margin).Clip(0, math.Inf(1)) // (N, C)
	// Zero out the target column so it doesn't contribute, then average over
	// the (C-1) non-target classes per row, then over the batch.
	mask := tensor.Zeros(N, C)
	for i, v := range target.Data {
		mask.Data[i*C+int(v)] = 1
	}
	nonTargetMask := tensor.Ones(N, C).Sub(mask) // 0 at target col, 1 elsewhere
	masked := margins.Mul(nonTargetMask)
	perRow := masked.SumAxis(1, false).DivScalar(float64(C - 1)) // (N,)
	return applyReduction(perRow, opts)
}

// CTCLoss (Connectionist Temporal Classification) is intentionally not
// implemented in v1. The dynamic-programming forward-backward algorithm over
// the expanded label sequence with blank tokens and log-space accumulation
// requires significant additional infrastructure (a real log-sum-exp, custom
// autograd, masked alignment) that is out of scope here. Track this as a
// future addition.
