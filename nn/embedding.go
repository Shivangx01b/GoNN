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

// EmbeddingBagOpt configures EmbeddingBag.
type EmbeddingBagOpt func(*embeddingBagOpts)

type embeddingBagOpts struct {
	mode string
}

// WithBagMode sets the per-bag reduction: "sum", "mean" (default), or "max"
// (PyTorch EmbeddingBag modes).
func WithBagMode(mode string) EmbeddingBagOpt {
	return func(o *embeddingBagOpts) { o.mode = mode }
}

// EmbeddingBag looks up embeddings and reduces them per bag ("sum", "mean",
// or "max") without materializing the intermediate (N, dim) result in the
// caller — the PyTorch torch.nn.EmbeddingBag equivalent.
type EmbeddingBag struct {
	Base
	NumEmbeddings int
	EmbeddingDim  int
	Mode          string
	Weight        *tensor.Tensor // (NumEmbeddings, EmbeddingDim)
}

// NewEmbeddingBag constructs an EmbeddingBag with N(0, 1) initialized
// weights. Default mode is "mean"; use WithBagMode("sum"|"mean"|"max").
func NewEmbeddingBag(numEmbeddings, dim int, opts ...EmbeddingBagOpt) *EmbeddingBag {
	o := embeddingBagOpts{mode: "mean"}
	for _, fn := range opts {
		fn(&o)
	}
	switch o.mode {
	case "sum", "mean", "max":
	default:
		panic(fmt.Sprintf("EmbeddingBag: unknown mode %q (want sum|mean|max)", o.mode))
	}
	e := &EmbeddingBag{NumEmbeddings: numEmbeddings, EmbeddingDim: dim, Mode: o.mode}
	e.Weight = e.reg("weight", tensor.Randn(numEmbeddings, dim).SetRequiresGrad(true))
	return e
}

// checkBagIndices bounds-checks embedding indices (negatives rejected, like
// Embedding).
func (e *EmbeddingBag) checkBagIndices(input *tensor.Tensor) {
	for _, v := range input.Data {
		if idx := int(v); idx < 0 || idx >= e.NumEmbeddings {
			panic(fmt.Sprintf("EmbeddingBag: index %d out of range [0,%d)", idx, e.NumEmbeddings))
		}
	}
}

// Forward reduces bags of embeddings. input is a flat 1-D tensor of indices;
// offsets is a 1-D tensor of bag start positions (PyTorch semantics:
// offsets[0] must be 0, offsets must be nondecreasing, and the last bag runs
// to the end of input). Returns (numBags, dim). Empty bags produce zero rows
// in every mode.
//
// "sum" and "mean" are computed as one GEMM against a constant (numBags, N)
// bag matrix, so gradients flow to Weight through the IndexSelect backward;
// "max" uses a differentiable per-bag MaxAxis.
func (e *EmbeddingBag) Forward(input, offsets *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape) != 1 {
		panic("EmbeddingBag.Forward: input must be 1-D (flat indices)")
	}
	if len(offsets.Shape) != 1 || offsets.Numel() == 0 {
		panic("EmbeddingBag.Forward: offsets must be a non-empty 1-D tensor")
	}
	N := input.Numel()
	numBags := offsets.Numel()

	starts := make([]int, numBags)
	for i, v := range offsets.Data {
		starts[i] = int(v)
	}
	if starts[0] != 0 {
		panic("EmbeddingBag.Forward: offsets[0] must be 0")
	}
	for i := 1; i < numBags; i++ {
		if starts[i] < starts[i-1] {
			panic("EmbeddingBag.Forward: offsets must be nondecreasing")
		}
	}
	if starts[numBags-1] > N {
		panic(fmt.Sprintf("EmbeddingBag.Forward: offset %d out of range [0,%d]", starts[numBags-1], N))
	}
	ends := make([]int, numBags)
	copy(ends, starts[1:])
	ends[numBags-1] = N // last bag runs to the end

	if N == 0 {
		return tensor.Zeros(numBags, e.EmbeddingDim)
	}
	e.checkBagIndices(input)
	gathered := e.Weight.IndexSelect(0, input) // (N, dim)

	switch e.Mode {
	case "sum", "mean":
		// One GEMM: out = bagMat @ gathered, bagMat (numBags, N) constant.
		bagMat := tensor.Zeros(numBags, N)
		for b := 0; b < numBags; b++ {
			count := ends[b] - starts[b]
			if count == 0 {
				continue // empty bag -> zero row
			}
			w := 1.0
			if e.Mode == "mean" {
				w = 1.0 / float64(count)
			}
			for j := starts[b]; j < ends[b]; j++ {
				bagMat.Data[b*N+j] = w
			}
		}
		return bagMat.MatMul(gathered) // (numBags, dim)
	default: // "max"
		rows := make([]*tensor.Tensor, numBags)
		for b := 0; b < numBags; b++ {
			count := ends[b] - starts[b]
			if count == 0 {
				rows[b] = tensor.Zeros(e.EmbeddingDim) // empty bag -> zeros
				continue
			}
			idx := make([]float64, count)
			for j := range idx {
				idx[j] = float64(starts[b] + j)
			}
			bag := gathered.IndexSelect(0, tensor.New(idx, count)) // (count, dim)
			rows[b] = bag.MaxAxis(0, false)                        // (dim)
		}
		return tensor.Stack(0, rows...) // (numBags, dim)
	}
}

// Forward2D treats each row of a (B, L) index tensor as one bag of fixed
// length L (PyTorch's 2-D EmbeddingBag input form). Returns (B, dim).
func (e *EmbeddingBag) Forward2D(input *tensor.Tensor) *tensor.Tensor {
	if len(input.Shape) != 2 {
		panic("EmbeddingBag.Forward2D: input must be (B, L)")
	}
	B, L := input.Shape[0], input.Shape[1]
	offsets := make([]float64, B)
	for b := range offsets {
		offsets[b] = float64(b * L)
	}
	return e.Forward(input.Reshape(B*L), tensor.New(offsets, B))
}
