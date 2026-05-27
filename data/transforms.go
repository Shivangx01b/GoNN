package data

import (
	"fmt"
	"math/rand"

	"gonn/tensor"
)

// Transform is a stateless (apart from RNG) tensor-to-tensor operation
// applied to a single sample. Implementations should return a new tensor
// and avoid mutating their input.
type Transform interface {
	Apply(t *tensor.Tensor) *tensor.Tensor
}

// Normalize subtracts a per-channel mean and divides by a per-channel
// standard deviation. The channel dimension is the first dimension of the
// input tensor (e.g. (C, H, W) images). Mean and Std must have length
// equal to that channel dimension.
type Normalize struct {
	Mean []float64
	Std  []float64
}

// Apply normalizes t in-place semantics (returns a new tensor with
// per-channel (t - mean) / std).
func (n Normalize) Apply(t *tensor.Tensor) *tensor.Tensor {
	if len(t.Shape) == 0 {
		panic("data.Normalize: cannot normalize a 0-D tensor")
	}
	c := t.Shape[0]
	if len(n.Mean) != c || len(n.Std) != c {
		panic(fmt.Sprintf("data.Normalize: mean/std length %d/%d must match channel dim %d", len(n.Mean), len(n.Std), c))
	}
	per := len(t.Data) / c
	out := tensor.Zeros(t.Shape...)
	for ch := 0; ch < c; ch++ {
		m := n.Mean[ch]
		s := n.Std[ch]
		if s == 0 {
			panic(fmt.Sprintf("data.Normalize: std[%d] is zero", ch))
		}
		base := ch * per
		for k := 0; k < per; k++ {
			out.Data[base+k] = (t.Data[base+k] - m) / s
		}
	}
	return out
}

// RandomHorizontalFlip flips the last dimension (width) of the input with
// probability P. Useful for image augmentation on (C, H, W) tensors.
type RandomHorizontalFlip struct {
	P    float64
	Rand *rand.Rand // optional; if nil, the global rand is used
}

// Apply randomly flips along the last dim.
func (rf RandomHorizontalFlip) Apply(t *tensor.Tensor) *tensor.Tensor {
	p := rf.P
	if p <= 0 {
		return t.Copy()
	}
	var r float64
	if rf.Rand != nil {
		r = rf.Rand.Float64()
	} else {
		r = rand.Float64()
	}
	if r >= p {
		return t.Copy()
	}
	return flipLastDim(t)
}

func flipLastDim(t *tensor.Tensor) *tensor.Tensor {
	if len(t.Shape) == 0 {
		return t.Copy()
	}
	w := t.Shape[len(t.Shape)-1]
	out := tensor.Zeros(t.Shape...)
	outer := len(t.Data) / w
	for o := 0; o < outer; o++ {
		base := o * w
		for i := 0; i < w; i++ {
			out.Data[base+i] = t.Data[base+(w-1-i)]
		}
	}
	return out
}

// RandomCrop selects a random contiguous window of the given Size from the
// trailing dimensions of the input. Size must have the same length as the
// number of dimensions to crop, applied to the *last* len(Size) dims.
type RandomCrop struct {
	Size []int
	Rand *rand.Rand
}

// Apply picks a random crop. If the input is already exactly Size along
// those dims, returns a copy.
func (rc RandomCrop) Apply(t *tensor.Tensor) *tensor.Tensor {
	if len(rc.Size) == 0 {
		return t.Copy()
	}
	if len(rc.Size) > len(t.Shape) {
		panic(fmt.Sprintf("data.RandomCrop: size %v has more dims than tensor shape %v", rc.Size, t.Shape))
	}
	rank := len(t.Shape)
	offset := rank - len(rc.Size)

	// Validate sizes and pick offsets.
	starts := make([]int, rank)
	outShape := append([]int(nil), t.Shape...)
	for i, s := range rc.Size {
		dim := t.Shape[offset+i]
		if s > dim || s <= 0 {
			panic(fmt.Sprintf("data.RandomCrop: crop size %d invalid for dim %d (size %d)", s, offset+i, dim))
		}
		span := dim - s
		var off int
		if span == 0 {
			off = 0
		} else if rc.Rand != nil {
			off = rc.Rand.Intn(span + 1)
		} else {
			off = rand.Intn(span + 1)
		}
		starts[offset+i] = off
		outShape[offset+i] = s
	}

	out := tensor.Zeros(outShape...)
	inStrides := contiguousStrides(t.Shape)
	outStrides := contiguousStrides(outShape)
	idx := make([]int, rank)
	total := len(out.Data)
	for k := 0; k < total; k++ {
		inOff := 0
		for d := 0; d < rank; d++ {
			inOff += (idx[d] + starts[d]) * inStrides[d]
		}
		out.Data[k] = t.Data[inOff]
		// increment idx
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < outShape[d] {
				break
			}
			idx[d] = 0
		}
		_ = outStrides
	}
	return out
}

// ToFloat is a no-op placeholder for converting tensors to float (we
// already store float64 everywhere). It exists so transform pipelines
// match the conventional shape.
type ToFloat struct{}

// Apply returns t unchanged.
func (ToFloat) Apply(t *tensor.Tensor) *tensor.Tensor { return t }

// Compose chains a sequence of Transforms, applying them left-to-right.
type Compose struct {
	Transforms []Transform
}

// Apply runs each transform in sequence.
func (c Compose) Apply(t *tensor.Tensor) *tensor.Tensor {
	out := t
	for _, tr := range c.Transforms {
		out = tr.Apply(out)
	}
	return out
}

// contiguousStrides is a local copy of tensor.contiguousStrides since the
// tensor package's helper is unexported.
func contiguousStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	s := make([]int, len(shape))
	s[len(shape)-1] = 1
	for i := len(shape) - 2; i >= 0; i-- {
		s[i] = s[i+1] * shape[i+1]
	}
	return s
}
