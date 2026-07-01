package nn

import (
	"fmt"
	"math"

	"gonn/tensor"
)

// EmbeddingOpt configures NewEmbedding (PyTorch nn.Embedding keyword args).
type EmbeddingOpt func(*embeddingOpts)

type embeddingOpts struct {
	paddingIdx int
	hasPad     bool
	maxNorm    float64
	hasMaxNorm bool
	normType   float64
}

// WithPaddingIdx marks one row as the padding vector (PyTorch padding_idx).
// The row is zeroed at construction (after the full weight draw, exactly like
// PyTorch, which fills N(0,1) and then zeroes the row — so the RNG sequence is
// unchanged) and receives no gradient: Forward still emits the row's current
// values at padded positions, but Backward writes exact zeros into
// Weight.Grad at that row. Negative idx counts from the end, Python-style.
func WithPaddingIdx(idx int) EmbeddingOpt {
	return func(o *embeddingOpts) { o.paddingIdx = idx; o.hasPad = true }
}

// WithMaxNorm enables PyTorch max_norm: at the start of every Forward, each
// weight row referenced by the input whose norm exceeds maxNorm is rescaled
// IN PLACE to have norm exactly maxNorm. Like PyTorch's embedding_renorm_,
// this mutates Weight.Data outside autograd (non-differentiable).
func WithMaxNorm(maxNorm float64) EmbeddingOpt {
	return func(o *embeddingOpts) { o.maxNorm = maxNorm; o.hasMaxNorm = true }
}

// WithNormType sets the p of the p-norm used by WithMaxNorm (default 2).
func WithNormType(p float64) EmbeddingOpt {
	return func(o *embeddingOpts) { o.normType = p }
}

// Embedding maps integer indices to dense vectors.
type Embedding struct {
	Base
	NumEmbeddings int
	EmbeddingDim  int
	PaddingIdx    int            // normalized padding row, or -1 when unset
	MaxNorm       float64        // renorm ceiling; active only when set via WithMaxNorm
	NormType      float64        // p of the p-norm for MaxNorm (default 2)
	Weight        *tensor.Tensor // (NumEmbeddings, EmbeddingDim)

	hasMaxNorm bool
}

// NewEmbedding constructs an Embedding with N(0, 1) initialized weights.
// The plain 2-arg call is bit-identical to the historical constructor: all
// NumEmbeddings*EmbeddingDim normals are drawn first, and option effects
// (padding-row zeroing) apply only afterwards, so the global-RNG draw order
// is unchanged.
func NewEmbedding(numEmbeddings, embeddingDim int, opts ...EmbeddingOpt) *Embedding {
	o := embeddingOpts{paddingIdx: -1, normType: 2}
	for _, fn := range opts {
		fn(&o)
	}
	pad := -1
	if o.hasPad {
		pad = o.paddingIdx
		if pad < 0 {
			pad += numEmbeddings
		}
		if pad < 0 || pad >= numEmbeddings {
			panic(fmt.Sprintf("NewEmbedding: padding_idx %d out of range [-%d,%d)",
				o.paddingIdx, numEmbeddings, numEmbeddings))
		}
	}
	if o.hasMaxNorm && o.maxNorm < 0 {
		panic(fmt.Sprintf("NewEmbedding: max_norm must be non-negative, got %g", o.maxNorm))
	}
	if o.normType <= 0 {
		panic(fmt.Sprintf("NewEmbedding: norm_type must be positive, got %g", o.normType))
	}
	e := &Embedding{
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
		PaddingIdx:    pad,
		MaxNorm:       o.maxNorm,
		NormType:      o.normType,
		hasMaxNorm:    o.hasMaxNorm,
	}
	e.Weight = e.reg("weight", tensor.Randn(numEmbeddings, embeddingDim).SetRequiresGrad(true))
	if pad >= 0 {
		// PyTorch also zeroes the padding row post-init (reset_parameters
		// fills N(0,1), then _fill_padding_idx_with_zero).
		zeroRow(e.Weight, pad, embeddingDim)
	}
	return e
}

// Forward looks up rows of Weight by integer indices (cast from float64)
// via IndexSelect — O(n·dim) with a scatter-add backward, instead of the
// historical O(n·vocab·dim) one-hot matmul. Indices are not differentiable.
//
// With WithMaxNorm, referenced rows are first renormalized in place (see
// renormRowsInPlace). With WithPaddingIdx, padded positions still emit the
// padding row's current values but contribute no gradient to it.
func (e *Embedding) Forward(indices *tensor.Tensor) *tensor.Tensor {
	// Strict bounds check: IndexSelect wraps negative indices Python-style,
	// but Embedding has always rejected them.
	for _, v := range indices.Data {
		if idx := int(v); idx < 0 || idx >= e.NumEmbeddings {
			panic(fmt.Sprintf("Embedding: index %d out of range [0,%d)", idx, e.NumEmbeddings))
		}
	}
	if e.hasMaxNorm {
		renormRowsInPlace(e.Weight, indices.Data, e.MaxNorm, e.NormType)
	}
	flat := indices.Reshape(indices.Numel())
	out := e.Weight.IndexSelect(0, flat) // (n, EmbeddingDim)
	if e.PaddingIdx >= 0 {
		out = e.maskPadding(out, flat)
	}
	outShape := append(append([]int(nil), indices.Shape...), e.EmbeddingDim)
	return out.Reshape(outShape...)
}

// maskPadding blocks gradient flow into the padding row while leaving the
// output values bit-identical: out = gathered·m + constRows, where m is a
// constant 0/1 (n,1) row mask (0 at padded positions) and constRows is a
// constant (n,dim) tensor carrying the current padding-row values at those
// positions (zeros elsewhere; no grad). At non-pad positions this is
// gathered·1 + 0 = gathered exactly; at pad positions 0·x + row = row
// exactly. In Backward, Mul scales the upstream grad by m, so the
// IndexSelect scatter-add deposits exact zeros into Weight.Grad[PaddingIdx].
//
// Note the OUTPUT still depends on the padding row — PyTorch semantics: the
// row is emitted verbatim; only its gradient is defined as excluded.
func (e *Embedding) maskPadding(gathered, flat *tensor.Tensor) *tensor.Tensor {
	n, dim := flat.Numel(), e.EmbeddingDim
	mask := make([]float64, n)
	rows := make([]float64, n*dim)
	padRow := e.Weight.Data[e.PaddingIdx*dim : (e.PaddingIdx+1)*dim]
	anyPad := false
	for i, v := range flat.Data {
		if int(v) == e.PaddingIdx {
			anyPad = true
			copy(rows[i*dim:(i+1)*dim], padRow)
		} else {
			mask[i] = 1
		}
	}
	if !anyPad {
		return gathered
	}
	m := tensor.New(mask, n, 1)           // constant, no grad
	constRows := tensor.New(rows, n, dim) // constant, no grad
	return gathered.Mul(m).Add(constRows)
}

// zeroRow zeroes row r of a (rows, dim) weight tensor.
func zeroRow(w *tensor.Tensor, r, dim int) {
	row := w.Data[r*dim : (r+1)*dim]
	for i := range row {
		row[i] = 0
	}
}

// renormRowsInPlace renormalizes — in place and OUTSIDE autograd, plain
// writes to weight.Data exactly like PyTorch's non-differentiable
// embedding_renorm_ — every weight row referenced by idxData whose p-norm
// exceeds maxNorm: row *= maxNorm/norm. Indices are deduplicated so each row
// is renormalized at most once per call; unreferenced rows are untouched.
func renormRowsInPlace(weight *tensor.Tensor, idxData []float64, maxNorm, p float64) {
	dim := weight.Shape[1]
	seen := make(map[int]struct{}, len(idxData))
	for _, v := range idxData {
		r := int(v)
		if _, dup := seen[r]; dup {
			continue
		}
		seen[r] = struct{}{}
		row := weight.Data[r*dim : (r+1)*dim]
		var norm float64
		switch p {
		case 2:
			for _, x := range row {
				norm += x * x
			}
			norm = math.Sqrt(norm)
		case 1:
			for _, x := range row {
				norm += math.Abs(x)
			}
		default:
			for _, x := range row {
				norm += math.Pow(math.Abs(x), p)
			}
			norm = math.Pow(norm, 1/p)
		}
		if norm > maxNorm {
			scale := maxNorm / norm
			for i := range row {
				row[i] *= scale
			}
		}
	}
}

// EmbeddingBagOpt configures EmbeddingBag.
type EmbeddingBagOpt func(*embeddingBagOpts)

type embeddingBagOpts struct {
	mode       string
	paddingIdx int
	hasPad     bool
	maxNorm    float64
	hasMaxNorm bool
}

// WithBagMode sets the per-bag reduction: "sum", "mean" (default), or "max"
// (PyTorch EmbeddingBag modes).
func WithBagMode(mode string) EmbeddingBagOpt {
	return func(o *embeddingBagOpts) { o.mode = mode }
}

// WithBagPaddingIdx marks one row as padding (PyTorch EmbeddingBag
// padding_idx): input entries equal to it are EXCLUDED from the bag
// reduction (sum skips them, mean divides by the count of non-pad entries,
// max ignores them; a bag left empty after exclusion follows the empty-bag
// rule and yields a zero row). The row is zeroed at construction, after the
// full weight draw. Negative idx counts from the end, Python-style.
func WithBagPaddingIdx(idx int) EmbeddingBagOpt {
	return func(o *embeddingBagOpts) { o.paddingIdx = idx; o.hasPad = true }
}

// WithBagMaxNorm enables PyTorch max_norm for EmbeddingBag: at the start of
// every Forward, each weight row referenced by the input whose L2 norm
// exceeds v is rescaled IN PLACE (outside autograd) to have norm exactly v.
func WithBagMaxNorm(v float64) EmbeddingBagOpt {
	return func(o *embeddingBagOpts) { o.maxNorm = v; o.hasMaxNorm = true }
}

// EmbeddingBag looks up embeddings and reduces them per bag ("sum", "mean",
// or "max") without materializing the intermediate (N, dim) result in the
// caller — the PyTorch torch.nn.EmbeddingBag equivalent.
type EmbeddingBag struct {
	Base
	NumEmbeddings int
	EmbeddingDim  int
	Mode          string
	PaddingIdx    int            // normalized padding row, or -1 when unset
	MaxNorm       float64        // renorm ceiling; active only when set via WithBagMaxNorm
	Weight        *tensor.Tensor // (NumEmbeddings, EmbeddingDim)

	hasMaxNorm bool
}

// NewEmbeddingBag constructs an EmbeddingBag with N(0, 1) initialized
// weights. Default mode is "mean"; use WithBagMode("sum"|"mean"|"max").
// Like NewEmbedding, all weights are drawn before any option effect
// (padding-row zeroing) applies, so the RNG draw order matches the
// option-free constructor.
func NewEmbeddingBag(numEmbeddings, dim int, opts ...EmbeddingBagOpt) *EmbeddingBag {
	o := embeddingBagOpts{mode: "mean", paddingIdx: -1}
	for _, fn := range opts {
		fn(&o)
	}
	switch o.mode {
	case "sum", "mean", "max":
	default:
		panic(fmt.Sprintf("EmbeddingBag: unknown mode %q (want sum|mean|max)", o.mode))
	}
	pad := -1
	if o.hasPad {
		pad = o.paddingIdx
		if pad < 0 {
			pad += numEmbeddings
		}
		if pad < 0 || pad >= numEmbeddings {
			panic(fmt.Sprintf("NewEmbeddingBag: padding_idx %d out of range [-%d,%d)",
				o.paddingIdx, numEmbeddings, numEmbeddings))
		}
	}
	if o.hasMaxNorm && o.maxNorm < 0 {
		panic(fmt.Sprintf("NewEmbeddingBag: max_norm must be non-negative, got %g", o.maxNorm))
	}
	e := &EmbeddingBag{
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  dim,
		Mode:          o.mode,
		PaddingIdx:    pad,
		MaxNorm:       o.maxNorm,
		hasMaxNorm:    o.hasMaxNorm,
	}
	e.Weight = e.reg("weight", tensor.Randn(numEmbeddings, dim).SetRequiresGrad(true))
	if pad >= 0 {
		zeroRow(e.Weight, pad, dim)
	}
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

// isPad reports whether flat input position j holds the padding index.
func (e *EmbeddingBag) isPad(input *tensor.Tensor, j int) bool {
	return e.PaddingIdx >= 0 && int(input.Data[j]) == e.PaddingIdx
}

// Forward reduces bags of embeddings. input is a flat 1-D tensor of indices;
// offsets is a 1-D tensor of bag start positions (PyTorch semantics:
// offsets[0] must be 0, offsets must be nondecreasing, and the last bag runs
// to the end of input). Returns (numBags, dim). Empty bags produce zero rows
// in every mode; with WithBagPaddingIdx, entries equal to the padding index
// are excluded from the reduction, and a bag that is entirely padding is
// treated as empty (zero row).
//
// "sum" and "mean" are computed as one GEMM against a constant (numBags, N)
// bag matrix, so gradients flow to Weight through the IndexSelect backward;
// "max" uses a differentiable per-bag MaxAxis. Padded positions get zero
// bag-matrix weight (sum/mean) or are dropped from the per-bag gather (max),
// so the padding row receives an exactly-zero gradient.
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
	if e.hasMaxNorm {
		renormRowsInPlace(e.Weight, input.Data, e.MaxNorm, 2)
	}
	gathered := e.Weight.IndexSelect(0, input) // (N, dim)

	switch e.Mode {
	case "sum", "mean":
		// One GEMM: out = bagMat @ gathered, bagMat (numBags, N) constant.
		// Padding entries get weight 0, so their grad contribution is an
		// exact zero and "mean" divides by the non-pad count only.
		bagMat := tensor.Zeros(numBags, N)
		for b := 0; b < numBags; b++ {
			count := 0
			for j := starts[b]; j < ends[b]; j++ {
				if !e.isPad(input, j) {
					count++
				}
			}
			if count == 0 {
				continue // empty (or all-padding) bag -> zero row
			}
			w := 1.0
			if e.Mode == "mean" {
				w = 1.0 / float64(count)
			}
			for j := starts[b]; j < ends[b]; j++ {
				if !e.isPad(input, j) {
					bagMat.Data[b*N+j] = w
				}
			}
		}
		return bagMat.MatMul(gathered) // (numBags, dim)
	default: // "max"
		rows := make([]*tensor.Tensor, numBags)
		for b := 0; b < numBags; b++ {
			var idx []float64
			for j := starts[b]; j < ends[b]; j++ {
				if e.isPad(input, j) {
					continue // padding entries are excluded from the max
				}
				idx = append(idx, float64(j))
			}
			if len(idx) == 0 {
				rows[b] = tensor.Zeros(e.EmbeddingDim) // empty/all-pad bag -> zeros
				continue
			}
			bag := gathered.IndexSelect(0, tensor.New(idx, len(idx))) // (count, dim)
			rows[b] = bag.MaxAxis(0, false)                           // (dim)
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
