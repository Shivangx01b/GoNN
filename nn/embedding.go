package nn

import (
	"gonn/tensor"
)

// Embedding maps integer indices to dense vectors.
type Embedding struct {
	NumEmbeddings int
	EmbeddingDim  int
	Weight        *tensor.Tensor // (NumEmbeddings, EmbeddingDim)
}

// NewEmbedding constructs an Embedding with N(0, 1) initialized weights.
func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	w := tensor.Randn(numEmbeddings, embeddingDim).SetRequiresGrad(true)
	return &Embedding{NumEmbeddings: numEmbeddings, EmbeddingDim: embeddingDim, Weight: w}
}

// Forward looks up rows of Weight by integer indices (cast from float64).
// Implemented as a one-hot @ Weight matmul so gradient scatters to Weight via
// the existing matmul backward. Indices are not differentiable.
func (e *Embedding) Forward(indices *tensor.Tensor) *tensor.Tensor {
	n := len(indices.Data)
	one := tensor.Zeros(n, e.NumEmbeddings)
	for i, v := range indices.Data {
		idx := int(v)
		if idx < 0 || idx >= e.NumEmbeddings {
			panic("Embedding: index out of range")
		}
		one.Data[i*e.NumEmbeddings+idx] = 1
	}
	out := one.MatMul(e.Weight) // (n, EmbeddingDim)
	outShape := append([]int(nil), indices.Shape...)
	outShape = append(outShape, e.EmbeddingDim)
	return out.Reshape(outShape...)
}

// Parameters returns the weight matrix.
func (e *Embedding) Parameters() []*tensor.Tensor { return []*tensor.Tensor{e.Weight} }
