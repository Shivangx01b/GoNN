package nn

import (
	"math"

	"gonn/tensor"
)

// CTCLoss — Connectionist Temporal Classification loss (Graves et al. 2006),
// mirroring torch.nn.CTCLoss.
//
//	logProbs:      (T, N, C) log-probabilities (apply LogSoftmax over C first).
//	targets:       flat (sum(targetLengths),) class indices stored as float64,
//	               concatenated per sample; must not contain blank.
//	inputLengths:  per-sample valid time steps (each in [1, T]).
//	targetLengths: per-sample target lengths (>= 0; zero-length is allowed and
//	               scores the all-blank path).
//	blank:         index of the blank class in [0, C).
//
// The forward pass runs the standard log-space alpha recursion over the
// extended label sequence (blanks interleaved, length 2S+1); the per-sample
// loss is -logsumexp(alpha_T[last two states]). If the input is too short to
// align to the target the loss is +Inf (PyTorch with zero_infinity=false).
//
// Gradient (attached via tensor.MakeNode): the alpha-beta posterior form
//
//	dLoss_n/dlogProbs[t,n,c] = -exp(loss_n) *
//	    sum_{s: ext[s]=c} exp(alpha[t,s] + beta[t,s] - logProbs[t,n,ext[s]])
//
// for t < inputLengths[n] and 0 elsewhere. This is the exact derivative with
// respect to the logProbs entries treated as free variables (it passes finite
// differences directly). Note PyTorch reports exp(logProbs) minus that
// posterior instead — the two differ only by a term that the LogSoftmax
// backward annihilates, so gradients w.r.t. pre-softmax logits are identical.
//
// Reduction (PyTorch semantics): the default mean divides each sample's loss
// by max(targetLengths[n], 1) and then averages over the batch; sum adds the
// raw per-sample losses; none returns the (N,) vector. Class-index options
// (WithClassWeights, WithIgnoreIndex, WithLabelSmoothing) are ignored.
func CTCLoss(logProbs, targets *tensor.Tensor, inputLengths, targetLengths []int, blank int, opts ...LossOpt) *tensor.Tensor {
	if len(logProbs.Shape) != 3 {
		panic("CTCLoss: logProbs must be 3D (T, N, C)")
	}
	T, N, C := logProbs.Shape[0], logProbs.Shape[1], logProbs.Shape[2]
	if blank < 0 || blank >= C {
		panic("CTCLoss: blank index out of range")
	}
	if len(inputLengths) != N || len(targetLengths) != N {
		panic("CTCLoss: inputLengths and targetLengths must each have N entries")
	}
	total := 0
	for n := 0; n < N; n++ {
		if inputLengths[n] < 1 || inputLengths[n] > T {
			panic("CTCLoss: input length out of range [1, T]")
		}
		if targetLengths[n] < 0 {
			panic("CTCLoss: negative target length")
		}
		total += targetLengths[n]
	}
	if targets.Numel() != total {
		panic("CTCLoss: targets must have sum(targetLengths) entries")
	}

	lp := func(t, n, c int) float64 { return logProbs.Data[(t*N+n)*C+c] }

	losses := make([]float64, N)
	gradTables := make([][]float64, N) // gradTables[n][t*C+c] = dloss_n/dlogProbs[t,n,c]

	off := 0
	for n := 0; n < N; n++ {
		S, Tn := targetLengths[n], inputLengths[n]
		lab := make([]int, S)
		for j := 0; j < S; j++ {
			v := int(targets.Data[off+j])
			if v < 0 || v >= C {
				panic("CTCLoss: target class out of range")
			}
			if v == blank {
				panic("CTCLoss: targets must not contain the blank index")
			}
			lab[j] = v
		}
		off += S

		// Extended sequence: blank, l1, blank, l2, ..., blank (length 2S+1).
		L := 2*S + 1
		ext := make([]int, L)
		for s := 1; s < L; s += 2 {
			ext[s] = lab[(s-1)/2]
		}
		for s := 0; s < L; s += 2 {
			ext[s] = blank
		}
		// The diagonal skip s-2 -> s is allowed only onto a label state whose
		// label differs from the one two states back (repeats need the blank).
		canSkip := func(s int) bool { return s >= 2 && ext[s] != blank && ext[s] != ext[s-2] }

		neg := math.Inf(-1)
		alpha := make([][]float64, Tn)
		beta := make([][]float64, Tn)
		for t := 0; t < Tn; t++ {
			alpha[t] = make([]float64, L)
			beta[t] = make([]float64, L)
			for s := 0; s < L; s++ {
				alpha[t][s], beta[t][s] = neg, neg
			}
		}

		alpha[0][0] = lp(0, n, ext[0])
		if L > 1 {
			alpha[0][1] = lp(0, n, ext[1])
		}
		for t := 1; t < Tn; t++ {
			for s := 0; s < L; s++ {
				a := alpha[t-1][s]
				if s >= 1 {
					a = ctcLogAdd(a, alpha[t-1][s-1])
				}
				if canSkip(s) {
					a = ctcLogAdd(a, alpha[t-1][s-2])
				}
				alpha[t][s] = a + lp(t, n, ext[s])
			}
		}
		ll := alpha[Tn-1][L-1]
		if L > 1 {
			ll = ctcLogAdd(ll, alpha[Tn-1][L-2])
		}
		losses[n] = -ll

		if math.IsInf(losses[n], 1) {
			// No feasible alignment: infinite loss, zero gradient (PyTorch
			// leaves the gradient undefined; zero_infinity would zero both).
			continue
		}

		// Beta recursion (symmetric: includes the emission at t).
		beta[Tn-1][L-1] = lp(Tn-1, n, ext[L-1])
		if L > 1 {
			beta[Tn-1][L-2] = lp(Tn-1, n, ext[L-2])
		}
		for t := Tn - 2; t >= 0; t-- {
			for s := 0; s < L; s++ {
				b := beta[t+1][s]
				if s+1 < L {
					b = ctcLogAdd(b, beta[t+1][s+1])
				}
				if s+2 < L && canSkip(s+2) {
					b = ctcLogAdd(b, beta[t+1][s+2])
				}
				beta[t][s] = b + lp(t, n, ext[s])
			}
		}

		// Posterior gradient. alpha and beta both include the emission at t,
		// so alpha+beta-lp is the joint log-probability of all paths passing
		// through state s at time t.
		w := make([]float64, Tn*C)
		for i := range w {
			w[i] = neg
		}
		for t := 0; t < Tn; t++ {
			for s := 0; s < L; s++ {
				c := ext[s]
				ab := alpha[t][s] + beta[t][s] - lp(t, n, c)
				w[t*C+c] = ctcLogAdd(w[t*C+c], ab)
			}
		}
		g := make([]float64, Tn*C)
		for i, wi := range w {
			if !math.IsInf(wi, -1) {
				g[i] = -math.Exp(wi + losses[n]) // 1/p = exp(loss_n)
			}
		}
		gradTables[n] = g
	}

	lossVec := tensor.New(losses, N)
	tensor.MakeNode(lossVec, "CTCLoss", []*tensor.Tensor{logProbs}, func(grad *tensor.Tensor) []*tensor.Tensor {
		dlp := tensor.Zeros(T, N, C)
		for n := 0; n < N; n++ {
			if gradTables[n] == nil {
				continue
			}
			gn := grad.Data[n]
			for t := 0; t < inputLengths[n]; t++ {
				for c := 0; c < C; c++ {
					dlp.Data[(t*N+n)*C+c] = gn * gradTables[n][t*C+c]
				}
			}
		}
		return []*tensor.Tensor{dlp}
	})

	switch parseLossOpts(opts).reduction {
	case ReduceNone:
		return lossVec
	case ReduceSum:
		return lossVec.Sum()
	default:
		// PyTorch mean: divide each sample by max(targetLength, 1), then
		// average over the batch (ATen clamps zero-length targets to 1).
		inv := tensor.Zeros(N)
		for n := 0; n < N; n++ {
			tl := targetLengths[n]
			if tl < 1 {
				tl = 1
			}
			inv.Data[n] = 1 / float64(tl)
		}
		return lossVec.Mul(inv).Mean()
	}
}

// ctcLogAdd returns log(exp(a) + exp(b)) without overflow; -Inf is the
// identity.
func ctcLogAdd(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}
