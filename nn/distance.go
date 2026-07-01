package nn

import "gonn/tensor"

// Distance/similarity modules. Both take two inputs, so they satisfy Child
// but not Module.

// CosineSimilarity computes the cosine similarity between x1 and x2 along Dim,
// mirroring torch.nn.CosineSimilarity. Eps avoids division by zero.
type CosineSimilarity struct {
	Base
	Dim int
	Eps float64
}

// NewCosineSimilarity builds a CosineSimilarity over dim with the given eps
// (0 selects the default 1e-8).
func NewCosineSimilarity(dim int, eps float64) *CosineSimilarity {
	if eps == 0 {
		eps = 1e-8
	}
	return &CosineSimilarity{Dim: dim, Eps: eps}
}

// Forward computes sum(x1*x2) / (||x1|| * ||x2||) along Dim, reducing that dim.
func (c *CosineSimilarity) Forward(x1, x2 *tensor.Tensor) *tensor.Tensor {
	dot := x1.Mul(x2).SumAxis(c.Dim, false)
	n1 := x1.Mul(x1).SumAxis(c.Dim, false).Sqrt()
	n2 := x2.Mul(x2).SumAxis(c.Dim, false).Sqrt()
	denom := n1.Mul(n2).AddScalar(c.Eps)
	return dot.Div(denom)
}

// PairwiseDistance computes the batched p-norm distance between x1 and x2
// along the last dimension, mirroring torch.nn.PairwiseDistance.
type PairwiseDistance struct {
	Base
	P   float64
	Eps float64
}

// NewPairwiseDistance builds a PairwiseDistance with norm P (0 selects 2)
// and stabilizer Eps (0 selects 1e-6).
func NewPairwiseDistance(p, eps float64) *PairwiseDistance {
	if p == 0 {
		p = 2
	}
	if eps == 0 {
		eps = 1e-6
	}
	return &PairwiseDistance{P: p, Eps: eps}
}

// Forward computes (sum |x1 - x2 + eps|^p)^(1/p) along the last dim.
func (d *PairwiseDistance) Forward(x1, x2 *tensor.Tensor) *tensor.Tensor {
	diff := x1.Sub(x2).AddScalar(d.Eps)
	last := len(x1.Shape) - 1
	powed := diff.Abs().Pow(d.P).SumAxis(last, false)
	return powed.Pow(1.0 / d.P)
}
