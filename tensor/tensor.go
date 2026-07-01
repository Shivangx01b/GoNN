// Package tensor provides the core Tensor type with automatic differentiation.
//
// Design follows PyTorch/tinygrad: a Tensor is flat float64 data plus a shape
// and strides. Operations build a DAG (Tensor.creator); calling Backward()
// on a scalar walks the graph in reverse topological order and accumulates
// gradients into leaf tensors that have RequiresGrad set.
//
// Alias policy: some methods have historical aliases kept for compatibility.
// The canonical names are SiLU (alias Swish), AsType (alias To), Reshape
// (alias View), Transpose (alias T; swaps the last two dims of any rank >= 2
// tensor), Unsqueeze, and Repeat (alias Tile).
package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Tensor is the core n-dimensional array with autograd.
type Tensor struct {
	Data         []float64
	Shape        []int
	Strides      []int
	Dtype        DType // logical element precision; zero value = Float64
	RequiresGrad bool
	Grad         *Tensor
	creator      *Function
}

// New creates a tensor wrapping data with the given shape.
func New(data []float64, shape ...int) *Tensor {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	n := numel(shape)
	if len(data) != n {
		opError("New", "data length %d does not match shape %v (numel=%d)", len(data), shape, n)
	}
	return &Tensor{Data: data, Shape: append([]int(nil), shape...), Strides: contiguousStrides(shape)}
}

// NewLike returns a tensor with the same shape as t, filled with zeros.
func NewLike(t *Tensor) *Tensor { return Zeros(t.Shape...) }

// Zeros returns a tensor of zeros with the given shape.
func Zeros(shape ...int) *Tensor {
	return &Tensor{Data: make([]float64, numel(shape)), Shape: append([]int(nil), shape...), Strides: contiguousStrides(shape)}
}

// Ones returns a tensor of ones with the given shape.
func Ones(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = 1
	}
	return t
}

// Full returns a tensor filled with v.
func Full(v float64, shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = v
	}
	return t
}

// Randn returns a tensor sampled from N(0, 1).
func Randn(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64()
	}
	return t
}

// Uniform returns a tensor sampled uniformly from [low, high).
func Uniform(low, high float64, shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = low + rand.Float64()*(high-low)
	}
	return t
}

// Arange returns [start, start+step, ..., stop).
func Arange(start, stop, step float64) *Tensor {
	if step == 0 {
		opError("Arange", "step must be non-zero")
	}
	n := int(math.Ceil((stop - start) / step))
	if n < 0 {
		n = 0
	}
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = start + float64(i)*step
	}
	return New(d, n)
}

// Eye returns the n x n identity matrix.
func Eye(n int) *Tensor {
	t := Zeros(n, n)
	for i := 0; i < n; i++ {
		t.Data[i*n+i] = 1
	}
	return t
}

// Scalar returns a 0-d tensor holding v.
func Scalar(v float64) *Tensor {
	return &Tensor{Data: []float64{v}, Shape: []int{}, Strides: []int{}}
}

// Copy makes a deep copy of t (does not copy creator/grad).
func (t *Tensor) Copy() *Tensor {
	d := make([]float64, len(t.Data))
	copy(d, t.Data)
	return &Tensor{
		Data:         d,
		Shape:        append([]int(nil), t.Shape...),
		Strides:      append([]int(nil), t.Strides...),
		Dtype:        t.Dtype,
		RequiresGrad: t.RequiresGrad,
	}
}

// SetRequiresGrad enables gradient tracking and returns t for chaining.
func (t *Tensor) SetRequiresGrad(b bool) *Tensor {
	t.RequiresGrad = b
	return t
}

// Numel returns the number of elements.
func (t *Tensor) Numel() int { return len(t.Data) }

// Dim returns the number of dimensions.
func (t *Tensor) Dim() int { return len(t.Shape) }

// Item returns the scalar value (only valid for tensors with 1 element).
func (t *Tensor) Item() float64 {
	if len(t.Data) != 1 {
		opError("Item", "only valid for single-element tensors, got shape %v", t.Shape)
	}
	return t.Data[0]
}

// String formats the tensor for printing.
func (t *Tensor) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Tensor(shape=%v, data=", t.Shape))
	const maxPrint = 16
	if len(t.Data) <= maxPrint {
		sb.WriteString(fmt.Sprintf("%v", t.Data))
	} else {
		sb.WriteString(fmt.Sprintf("[%v ... %v]", t.Data[:8], t.Data[len(t.Data)-8:]))
	}
	if t.Dtype != Float64 {
		sb.WriteString(", dtype=" + t.Dtype.String())
	}
	if t.RequiresGrad {
		sb.WriteString(", requires_grad=true")
	}
	sb.WriteString(")")
	return sb.String()
}

// ZeroGrad zeros the gradient.
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad.Data {
			t.Grad.Data[i] = 0
		}
	}
}
