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

	// Class-index loss options (CrossEntropyLoss / NLLLoss only; every other
	// loss silently ignores them).
	classWeights   []float64
	ignoreIndex    int
	hasIgnore      bool
	labelSmoothing float64
}

// WithReduction selects mean (default), sum, or no reduction.
func WithReduction(r Reduction) LossOpt { return func(o *lossOpts) { o.reduction = r } }

// WithClassWeights supplies a per-class rescaling weight w (length C) for
// CrossEntropyLoss / NLLLoss: sample n contributes w[y_n] * loss_n, and —
// matching PyTorch exactly — the mean reduction divides by the SUM OF THE
// WEIGHTS of the (non-ignored) targets, not by N. Other losses ignore this
// option.
func WithClassWeights(w []float64) LossOpt {
	cp := append([]float64(nil), w...)
	return func(o *lossOpts) { o.classWeights = cp }
}

// WithIgnoreIndex makes CrossEntropyLoss / NLLLoss targets equal to idx
// contribute zero loss and be excluded from the mean denominator (count of
// non-ignored samples, or sum of their class weights). Other losses ignore
// this option.
func WithIgnoreIndex(idx int) LossOpt {
	return func(o *lossOpts) { o.ignoreIndex = idx; o.hasIgnore = true }
}

// WithLabelSmoothing sets the label-smoothing amount eps in [0,1] for
// CrossEntropyLoss (PyTorch's label_smoothing): the target distribution
// becomes (1-eps)*one_hot + eps/C, so per sample
//
//	loss_n = (1-eps) * nll_n + (eps/C) * sum_c w_c * (-logProbs[n,c]),
//
// with ignored samples contributing zero to both terms. NLLLoss and all
// other losses ignore this option.
func WithLabelSmoothing(eps float64) LossOpt {
	return func(o *lossOpts) { o.labelSmoothing = eps }
}

// parseLossOpts folds the option list over the defaults.
func parseLossOpts(opts []LossOpt) lossOpts {
	o := lossOpts{reduction: ReduceMean}
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// applyReduction reduces the raw loss tensor per the options.
func applyReduction(t *tensor.Tensor, opts []LossOpt) *tensor.Tensor {
	switch parseLossOpts(opts).reduction {
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
// Honors WithClassWeights, WithIgnoreIndex and WithLabelSmoothing.
func CrossEntropyLoss(logits, targets *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	o := parseLossOpts(opts)
	logP := logits.LogSoftmax(1)
	vec, denom := classNLLVec("CrossEntropyLoss", logP, targets, o)
	if eps := o.labelSmoothing; eps != 0 {
		N, C := logP.Shape[0], logP.Shape[1]
		// smooth_n = -sum_c w_c * logP[n, c]  (w_c = 1 without class weights),
		// zeroed for ignored samples — matches PyTorch's
		// (1-eps)*nll + (eps/C)*smooth decomposition of the eps/C-smoothed
		// target distribution.
		var smooth *tensor.Tensor
		if o.classWeights != nil {
			wRow := tensor.New(append([]float64(nil), o.classWeights...), 1, C)
			smooth = logP.Mul(wRow).SumAxis(1, false).Neg() // (N,)
		} else {
			smooth = logP.SumAxis(1, false).Neg() // (N,)
		}
		if o.hasIgnore {
			keep := tensor.Ones(N)
			for i, v := range targets.Data {
				if int(v) == o.ignoreIndex {
					keep.Data[i] = 0
				}
			}
			smooth = smooth.Mul(keep)
		}
		vec = vec.MulScalar(1 - eps).Add(smooth.MulScalar(eps / float64(C)))
	}
	return reduceClassLoss(vec, denom, o)
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
// Honors WithClassWeights and WithIgnoreIndex (WithLabelSmoothing applies to
// CrossEntropyLoss only and is ignored here, as in PyTorch).
func NLLLoss(logProbs, targets *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	o := parseLossOpts(opts)
	vec, denom := classNLLVec("NLLLoss", logProbs, targets, o)
	return reduceClassLoss(vec, denom, o)
}

// classNLLVec computes the per-sample weighted NLL vector
//
//	l_n = -w[y_n] * logProbs[n, y_n]   (0 for ignored samples)
//
// and the mean-reduction denominator: sum over non-ignored samples of w[y_n]
// (which is the non-ignored count when no class weights are given, and N in
// the plain case — PyTorch's exact semantics).
func classNLLVec(op string, logProbs, targets *tensor.Tensor, o lossOpts) (*tensor.Tensor, float64) {
	if len(logProbs.Shape) != 2 {
		panic(op + ": input must be 2D (N, C)")
	}
	N, C := logProbs.Shape[0], logProbs.Shape[1]
	if len(targets.Data) != N {
		panic(op + ": targets must have N entries")
	}
	if o.classWeights != nil && len(o.classWeights) != C {
		panic(op + ": class weights must have C entries")
	}
	safeIdx := tensor.Zeros(N, 1) // gather index; 0 for ignored rows
	w := tensor.Zeros(N)          // effective per-sample weight; 0 for ignored
	denom := 0.0
	for i, v := range targets.Data {
		idx := int(v)
		if o.hasIgnore && idx == o.ignoreIndex {
			continue // safeIdx stays 0, weight stays 0
		}
		if idx < 0 || idx >= C {
			panic(op + ": target out of range")
		}
		safeIdx.Data[i] = float64(idx)
		wi := 1.0
		if o.classWeights != nil {
			wi = o.classWeights[idx]
		}
		w.Data[i] = wi
		denom += wi
	}
	picked := logProbs.Gather(1, safeIdx).Reshape(N) // (N,)
	// Multiplying by the constant 0 weight both zeroes an ignored sample's
	// loss and kills its gradient, so the placeholder index 0 is harmless.
	return picked.Mul(w).Neg(), denom
}

// reduceClassLoss applies the reduction for class-index losses: mean divides
// by the weighted denominator from classNLLVec (PyTorch semantics), not N.
func reduceClassLoss(vec *tensor.Tensor, denom float64, o lossOpts) *tensor.Tensor {
	switch o.reduction {
	case ReduceSum:
		return vec.Sum()
	case ReduceNone:
		return vec
	default:
		return vec.Sum().DivScalar(denom)
	}
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

// SoftMarginLoss: two-class logistic loss over +/-1 targets,
//
//	loss = mean_i log(1 + exp(-y[i]*x[i])),
//
// computed via the numerically stable softplus identity
// log(1+exp(z)) = softplus(z). x and y share any shape; the default mean
// reduction averages over every element (PyTorch's x.nelement()).
func SoftMarginLoss(x, y *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	return applyReduction(y.Mul(x).Neg().Softplus(), opts)
}

// MultiLabelSoftMarginLoss: one-vs-all logistic loss over C classes.
// x: (N, C) logits; y: (N, C) with entries in {0, 1}.
//
//	loss_n = -1/C * sum_c [ y*log σ(x) + (1-y)*log(1-σ(x)) ],
//
// computed per element with the stable BCE-with-logits form
// max(z,0) - z*y + softplus(-|z|), then averaged over classes per sample and
// reduced over the batch (mean by default; 'none' returns the (N,) vector).
func MultiLabelSoftMarginLoss(x, y *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	if len(x.Shape) != 2 {
		panic("MultiLabelSoftMarginLoss: input must be 2D (N, C)")
	}
	perElem := x.ReLU().Sub(x.Mul(y)).Add(x.Abs().Neg().Softplus()) // (N, C)
	return applyReduction(perElem.MeanAxis(1, false), opts)
}

// MultiLabelMarginLoss: multi-class multi-label hinge loss.
// x: (N, C) scores; y: (N, C) integer class indices stored as float64,
// padded with -1 — only the contiguous non-negative block at the front of
// each row is a target (the first -1 terminates it). Per sample,
//
//	loss_n = sum_{j in targets} sum_{i not in target set}
//	           max(0, 1 - (x[y[j]] - x[i])) / C,
//
// exactly PyTorch's formula (denominator x.size(-1) = C; the inner sum
// skips every class that appears among the targets). 1-D inputs (C,) are
// treated as a single sample. Default reduction: mean over the batch.
func MultiLabelMarginLoss(x, y *tensor.Tensor, opts ...LossOpt) *tensor.Tensor {
	if len(x.Shape) == 1 {
		x = x.Reshape(1, x.Shape[0])
		y = y.Reshape(1, y.Shape[0])
	}
	if len(x.Shape) != 2 {
		panic("MultiLabelMarginLoss: input must be 1D (C,) or 2D (N, C)")
	}
	N, C := x.Shape[0], x.Shape[1]
	if len(y.Shape) != 2 || y.Shape[0] != N || y.Shape[1] != C {
		panic("MultiLabelMarginLoss: target must have the same shape as input")
	}
	perSample := make([]*tensor.Tensor, N)
	for n := 0; n < N; n++ {
		// Valid targets: the contiguous block before the first -1.
		var tgt []float64
		inTarget := make([]bool, C)
		for c := 0; c < C; c++ {
			v := int(y.Data[n*C+c])
			if v < 0 {
				break
			}
			if v >= C {
				panic("MultiLabelMarginLoss: target class out of range")
			}
			tgt = append(tgt, float64(v))
			inTarget[v] = true
		}
		if len(tgt) == 0 {
			perSample[n] = tensor.Zeros(1)
			continue
		}
		S := len(tgt)
		row := x.IndexSelect(0, tensor.New([]float64{float64(n)}, 1)) // (1, C)
		xt := row.Gather(1, tensor.New(tgt, 1, S)).Reshape(S, 1)      // (S, 1)
		// hinge[j, i] = max(0, 1 + x[i] - x[y[j]]) via (1,C)-(S,1) broadcast.
		hinge := row.Sub(xt).AddScalar(1).Clip(0, math.Inf(1)) // (S, C)
		// Mask out every target class from the i-sum (constant 0/1 mask, so
		// masked entries also contribute no gradient).
		mask := tensor.Ones(1, C)
		for c := 0; c < C; c++ {
			if inTarget[c] {
				mask.Data[c] = 0
			}
		}
		perSample[n] = hinge.Mul(mask).Sum().DivScalar(float64(C)) // (1,)
	}
	return applyReduction(tensor.Concat(0, perSample...), opts)
}

// DistanceFunc maps two (N, D) tensors to a per-sample (N,) distance tensor.
// It must be built from differentiable tensor ops so gradients flow.
type DistanceFunc func(x1, x2 *tensor.Tensor) *tensor.Tensor

// PairwiseL2Distance is the default DistanceFunc: the per-sample Euclidean
// distance sqrt(sum_d (x1-x2)^2 + eps), matching TripletMarginLoss.
func PairwiseL2Distance(x1, x2 *tensor.Tensor) *tensor.Tensor {
	const eps = 1e-8
	return x1.Sub(x2).Square().SumAxis(1, false).AddScalar(eps).Sqrt()
}

// TripletMarginWithDistanceLoss: like TripletMarginLoss but with a caller-
// supplied distance:
//
//	loss = mean(max(0, d(a,p) - d(a,n) + margin)).
//
// distance == nil selects PairwiseL2Distance (PyTorch's default
// nn.PairwiseDistance). anchor, positive, negative share shape (N, D).
func TripletMarginWithDistanceLoss(anchor, positive, negative *tensor.Tensor, distance DistanceFunc, margin float64, opts ...LossOpt) *tensor.Tensor {
	if distance == nil {
		distance = PairwiseL2Distance
	}
	dp := distance(anchor, positive) // (N,)
	dn := distance(anchor, negative) // (N,)
	return applyReduction(dp.Sub(dn).AddScalar(margin).Clip(0, math.Inf(1)), opts)
}
