package nn

import (
	"math"
	"strconv"

	"gonn/tensor"
)

// AdaptiveSoftmaxOpt configures NewAdaptiveLogSoftmaxWithLoss.
type AdaptiveSoftmaxOpt func(*adaptiveSoftmaxOpts)

type adaptiveSoftmaxOpts struct {
	divValue float64
	headBias bool
}

// WithDivValue sets the exponential base used to shrink each tail cluster's
// projection size (PyTorch's div_value, default 4.0).
func WithDivValue(v float64) AdaptiveSoftmaxOpt {
	return func(o *adaptiveSoftmaxOpts) { o.divValue = v }
}

// WithHeadBias adds a bias to the head projection (PyTorch's head_bias,
// default false).
func WithHeadBias(b bool) AdaptiveSoftmaxOpt {
	return func(o *adaptiveSoftmaxOpts) { o.headBias = b }
}

// AdaptiveLogSoftmaxWithLoss is an efficient softmax approximation for large
// output spaces (Grave et al., "Efficient softmax approximation for GPUs"),
// mirroring torch.nn.AdaptiveLogSoftmaxWithLoss. Classes must be sorted by
// descending frequency: [0, cutoffs[0]) is the cheap shortlist handled by the
// head; each remaining slice [cutoffs[i], cutoffs[i+1]) is a tail cluster
// reached through a low-rank two-layer projection and a per-cluster token in
// the head. It takes two inputs (features and targets), so like Bilinear it
// satisfies Child but not Module.
type AdaptiveLogSoftmaxWithLoss struct {
	Base
	InFeatures    int
	NClasses      int
	Cutoffs       []int // user cutoffs with NClasses appended
	DivValue      float64
	HeadBias      bool
	ShortlistSize int
	NClusters     int
	HeadSize      int
	Head          *Linear
	Tail          []*Sequential
}

// NewAdaptiveLogSoftmaxWithLoss builds the module. cutoffs must be unique,
// strictly increasing integers in [1, nClasses-1]. The head is
// Linear(inFeatures, cutoffs[0]+nClusters, headBias); tail cluster i is
// Sequential(Linear(inFeatures, hsz_i, false), Linear(hsz_i, clusterSize_i,
// false)) with hsz_i = floor(inFeatures / divValue^(i+1)) (clamped to >= 1).
func NewAdaptiveLogSoftmaxWithLoss(inFeatures, nClasses int, cutoffs []int, opts ...AdaptiveSoftmaxOpt) *AdaptiveLogSoftmaxWithLoss {
	o := adaptiveSoftmaxOpts{divValue: 4.0}
	for _, fn := range opts {
		fn(&o)
	}
	if inFeatures < 1 || nClasses < 2 {
		panic("AdaptiveLogSoftmaxWithLoss: need inFeatures >= 1 and nClasses >= 2")
	}
	if o.divValue <= 0 {
		panic("AdaptiveLogSoftmaxWithLoss: divValue must be positive")
	}
	if len(cutoffs) == 0 {
		panic("AdaptiveLogSoftmaxWithLoss: cutoffs must be non-empty")
	}
	prev := 0
	for _, c := range cutoffs {
		if c <= prev || c > nClasses-1 {
			panic("AdaptiveLogSoftmaxWithLoss: cutoffs must be unique, increasing integers in [1, nClasses-1]")
		}
		prev = c
	}

	a := &AdaptiveLogSoftmaxWithLoss{
		InFeatures: inFeatures,
		NClasses:   nClasses,
		Cutoffs:    append(append([]int(nil), cutoffs...), nClasses),
		DivValue:   o.divValue,
		HeadBias:   o.headBias,
	}
	a.ShortlistSize = a.Cutoffs[0]
	a.NClusters = len(a.Cutoffs) - 1
	a.HeadSize = a.ShortlistSize + a.NClusters

	a.Head = NewLinear(inFeatures, a.HeadSize, o.headBias)
	a.regChild("head", a.Head)
	for i := 0; i < a.NClusters; i++ {
		hsz := int(math.Floor(float64(inFeatures) / math.Pow(o.divValue, float64(i+1))))
		if hsz < 1 {
			hsz = 1
		}
		osz := a.Cutoffs[i+1] - a.Cutoffs[i]
		seq := NewSequential(NewLinear(inFeatures, hsz, false), NewLinear(hsz, osz, false))
		a.Tail = append(a.Tail, seq)
		a.regChild("tail."+strconv.Itoa(i), seq)
	}
	return a
}

// clusterIndex returns the tail cluster index for class t >= ShortlistSize.
func (a *AdaptiveLogSoftmaxWithLoss) clusterIndex(t int) int {
	for i := 0; i < a.NClusters; i++ {
		if t < a.Cutoffs[i+1] {
			return i
		}
	}
	panic("AdaptiveLogSoftmaxWithLoss: target out of range")
}

// Forward computes, for input (N, InFeatures) and target (N,) class indices
// stored as float64, the (N,) log-probabilities of each sample's target class
// and the scalar loss = mean(-output) — exactly PyTorch's NamedTuple(output,
// loss). Shortlist targets read the head log-softmax directly; a tail target
// adds its cluster token's head log-probability and its within-cluster
// log-probability.
func (a *AdaptiveLogSoftmaxWithLoss) Forward(input, target *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	if len(input.Shape) != 2 || input.Shape[1] != a.InFeatures {
		panic("AdaptiveLogSoftmaxWithLoss: input must be (N, InFeatures)")
	}
	N := input.Shape[0]
	if target.Numel() != N {
		panic("AdaptiveLogSoftmaxWithLoss: target must have N entries")
	}

	// Partition rows by target cluster and pick each row's head column: the
	// target itself for shortlist rows, the cluster token otherwise.
	headIdx := tensor.Zeros(N, 1)
	var shortRows []float64
	buckets := make([][]float64, a.NClusters)
	for i := 0; i < N; i++ {
		t := int(target.Data[i])
		if t < 0 || t >= a.NClasses {
			panic("AdaptiveLogSoftmaxWithLoss: target out of range")
		}
		if t < a.ShortlistSize {
			headIdx.Data[i] = float64(t)
			shortRows = append(shortRows, float64(i))
		} else {
			c := a.clusterIndex(t)
			headIdx.Data[i] = float64(a.ShortlistSize + c)
			buckets[c] = append(buckets[c], float64(i))
		}
	}

	headLogProb := a.Head.Forward(input).LogSoftmax(1)    // (N, HeadSize)
	headPart := headLogProb.Gather(1, headIdx).Reshape(N) // (N,)

	// Tail part: per-cluster local log-probs, scattered back to the original
	// row order with a permutation IndexSelect (differentiable scatter).
	var parts []*tensor.Tensor
	var order []float64
	if len(shortRows) > 0 {
		parts = append(parts, tensor.Zeros(len(shortRows))) // shortlist adds 0
		order = append(order, shortRows...)
	}
	for c := 0; c < a.NClusters; c++ {
		rows := buckets[c]
		if len(rows) == 0 {
			continue
		}
		M := len(rows)
		sub := input.IndexSelect(0, tensor.New(rows, M))       // (M, F)
		clusterLogProb := a.Tail[c].Forward(sub).LogSoftmax(1) // (M, osz)
		rel := tensor.Zeros(M, 1)
		for k, r := range rows {
			rel.Data[k] = target.Data[int(r)] - float64(a.Cutoffs[c])
		}
		parts = append(parts, clusterLogProb.Gather(1, rel).Reshape(M))
		order = append(order, rows...)
	}
	inv := make([]float64, N)
	for k, r := range order {
		inv[int(r)] = float64(k)
	}
	tailPart := tensor.Concat(0, parts...).IndexSelect(0, tensor.New(inv, N))

	output := headPart.Add(tailPart) // (N,)
	loss := output.Neg().Mean()
	return output, loss
}

// LogProb returns the full (N, NClasses) log-probability distribution:
// shortlist columns come straight from the head log-softmax; cluster columns
// are the cluster's local log-softmax shifted by its head token's
// log-probability. Rows are proper log-distributions (they exp-sum to 1).
func (a *AdaptiveLogSoftmaxWithLoss) LogProb(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape) != 2 || input.Shape[1] != a.InFeatures {
		panic("AdaptiveLogSoftmaxWithLoss: input must be (N, InFeatures)")
	}
	headLogProb := a.Head.Forward(input).LogSoftmax(1) // (N, HeadSize)

	shortCols := make([]float64, a.ShortlistSize)
	for i := range shortCols {
		shortCols[i] = float64(i)
	}
	pieces := []*tensor.Tensor{headLogProb.IndexSelect(1, tensor.New(shortCols, a.ShortlistSize))}
	for c := 0; c < a.NClusters; c++ {
		clusterLogProb := a.Tail[c].Forward(input).LogSoftmax(1) // (N, osz)
		tokenCol := headLogProb.IndexSelect(1,
			tensor.New([]float64{float64(a.ShortlistSize + c)}, 1)) // (N, 1)
		pieces = append(pieces, clusterLogProb.Add(tokenCol))
	}
	return tensor.Concat(1, pieces...)
}

// Predict returns the (N,) most-likely class index per sample, equivalent to
// LogProb(input).ArgMax(1).
func (a *AdaptiveLogSoftmaxWithLoss) Predict(input *tensor.Tensor) *tensor.Tensor {
	return a.LogProb(input).ArgMax(1)
}
