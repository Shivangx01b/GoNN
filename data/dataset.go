// Package data provides dataset abstractions and data loading utilities
// for GoNN. It mirrors the pattern of torch.utils.data: a Dataset is an
// indexable collection of (x, y) samples, and a DataLoader batches and
// optionally shuffles a Dataset for training.
package data

import (
	"fmt"

	"gonn/tensor"
)

// Dataset is an indexable collection of (input, target) samples. Indices
// range from 0 to Len()-1. Get must return tensors that DO NOT share
// storage with the dataset's internal buffers if the caller is expected to
// mutate them.
type Dataset interface {
	Len() int
	Get(i int) (x, y *tensor.Tensor)
}

// TensorDataset wraps two tensors X and Y whose leading dimension indexes
// samples. Get(i) returns the i-th slice along dim 0 of each.
type TensorDataset struct {
	X *tensor.Tensor
	Y *tensor.Tensor
}

// NewTensorDataset constructs a TensorDataset. X and Y must share the same
// length along dimension 0.
func NewTensorDataset(X, Y *tensor.Tensor) *TensorDataset {
	if X == nil || Y == nil {
		panic("data.NewTensorDataset: X and Y must be non-nil")
	}
	if len(X.Shape) == 0 || len(Y.Shape) == 0 {
		panic("data.NewTensorDataset: X and Y must have at least 1 dimension")
	}
	if X.Shape[0] != Y.Shape[0] {
		panic(fmt.Sprintf("data.NewTensorDataset: X.Shape[0]=%d != Y.Shape[0]=%d", X.Shape[0], Y.Shape[0]))
	}
	return &TensorDataset{X: X, Y: Y}
}

// Len returns the number of samples (size of dim 0).
func (d *TensorDataset) Len() int { return d.X.Shape[0] }

// Get returns a deep-copied slice of X and Y at row i. Shapes drop the
// leading sample dimension. If Y is 1-D, the returned y is a scalar (0-D).
func (d *TensorDataset) Get(i int) (x, y *tensor.Tensor) {
	if i < 0 || i >= d.Len() {
		panic(fmt.Sprintf("data.TensorDataset.Get: index %d out of range [0,%d)", i, d.Len()))
	}
	x = sliceRow(d.X, i)
	y = sliceRow(d.Y, i)
	return x, y
}

// sliceRow returns a deep copy of t[i, ...]. The returned tensor has shape
// t.Shape[1:]. If t is 1-D, the result is a 0-D scalar tensor.
func sliceRow(t *tensor.Tensor, i int) *tensor.Tensor {
	if len(t.Shape) == 0 {
		panic("data: sliceRow on 0-D tensor")
	}
	if len(t.Shape) == 1 {
		return tensor.Scalar(t.Data[i])
	}
	rowShape := append([]int(nil), t.Shape[1:]...)
	rowSize := 1
	for _, d := range rowShape {
		rowSize *= d
	}
	buf := make([]float64, rowSize)
	copy(buf, t.Data[i*rowSize:(i+1)*rowSize])
	return tensor.New(buf, rowShape...)
}

// subsetDataset is a view of another Dataset selecting only a subset of
// indices (in arbitrary order, with possible repeats).
type subsetDataset struct {
	base    Dataset
	indices []int
}

// Subset returns a Dataset that exposes only the samples at the given
// indices of the underlying dataset, in the given order. The slice is
// copied; mutating it after the call has no effect.
func Subset(d Dataset, indices []int) Dataset {
	if d == nil {
		panic("data.Subset: dataset must be non-nil")
	}
	n := d.Len()
	idx := make([]int, len(indices))
	for i, j := range indices {
		if j < 0 || j >= n {
			panic(fmt.Sprintf("data.Subset: index %d out of range [0,%d)", j, n))
		}
		idx[i] = j
	}
	return &subsetDataset{base: d, indices: idx}
}

func (s *subsetDataset) Len() int { return len(s.indices) }
func (s *subsetDataset) Get(i int) (*tensor.Tensor, *tensor.Tensor) {
	return s.base.Get(s.indices[i])
}

// concatDataset is the concatenation of several datasets, indexed in
// order: samples of ds[0], then ds[1], etc.
type concatDataset struct {
	datasets []Dataset
	cum      []int // cumulative lengths; cum[k] is total length of datasets[:k+1]
	total    int
}

// ConcatDataset returns a Dataset that is the concatenation of the given
// datasets in order.
func ConcatDataset(ds ...Dataset) Dataset {
	if len(ds) == 0 {
		panic("data.ConcatDataset: at least one dataset required")
	}
	cum := make([]int, len(ds))
	total := 0
	for i, d := range ds {
		if d == nil {
			panic(fmt.Sprintf("data.ConcatDataset: dataset %d is nil", i))
		}
		total += d.Len()
		cum[i] = total
	}
	return &concatDataset{datasets: ds, cum: cum, total: total}
}

func (c *concatDataset) Len() int { return c.total }
func (c *concatDataset) Get(i int) (*tensor.Tensor, *tensor.Tensor) {
	if i < 0 || i >= c.total {
		panic(fmt.Sprintf("data.ConcatDataset.Get: index %d out of range [0,%d)", i, c.total))
	}
	// Binary search for the dataset containing i.
	lo, hi := 0, len(c.cum)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if c.cum[mid] <= i {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	local := i
	if lo > 0 {
		local = i - c.cum[lo-1]
	}
	return c.datasets[lo].Get(local)
}
