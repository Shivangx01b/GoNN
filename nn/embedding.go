package nn

import (
	"fmt"

	"gonn/tensor"
)

// Embedding maps integer indices to dense vectors.
type Embedding struct {
	Base
	NumEmbeddings int
	EmbeddingDim  int
	Weight        *tensor.Tensor // (NumEmbeddings, EmbeddingDim)
}

// NewEmbedding constructs an Embedding with N(0, 1) initialized weights.
func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	e := &Embedding{NumEmbeddings: numEmbeddings, EmbeddingDim: embeddingDim}
	e.Weight = e.reg("weight", tensor.Randn(numEmbeddings, embeddingDim).SetRequiresGrad(true))
	return e
}

// Forward looks up rows of Weight by integer indices (cast from float64)
// via IndexSelect — O(n·dim) with a scatter-add backward, instead of the
// historical O(n·vocab·dim) one-hot matmul. Indices are not differentiable.
func (e *Embedding) Forward(indices *tensor.Tensor) *tensor.Tensor {
	// Strict bounds check: IndexSelect wraps negative indices Python-style,
	// but Embedding has always rejected them.
	for _, v := range indices.Data {
		if idx := int(v); idx < 0 || idx >= e.NumEmbeddings {
			panic(fmt.Sprintf("Embedding: index %d out of range [0,%d)", idx, e.NumEmbeddings))
		}
	}
	flat := indices.Reshape(indices.Numel())
	out := e.Weight.IndexSelect(0, flat) // (n, EmbeddingDim)
	outShape := append(append([]int(nil), indices.Shape...), e.EmbeddingDim)
	return out.Reshape(outShape...)
}
