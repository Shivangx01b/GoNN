package data

import (
	"math/rand"

	"gonn/tensor"
)

// Batch is one mini-batch produced by a DataLoader. X and Y each have a
// leading batch dimension.
type Batch struct {
	X *tensor.Tensor
	Y *tensor.Tensor
}

// DataLoader iterates over a Dataset in mini-batches, optionally shuffling
// the order each epoch.
type DataLoader struct {
	Dataset   Dataset
	BatchSize int
	Shuffle   bool
	// DropLast, when true, drops the final batch if it would contain fewer
	// than BatchSize samples.
	DropLast bool
	// Seed controls shuffle order. If zero, a fresh random source is used
	// each iteration.
	Seed int64
}

// NewDataLoader constructs a DataLoader with sensible defaults
// (DropLast=false, Seed=0).
func NewDataLoader(d Dataset, batchSize int, shuffle bool) *DataLoader {
	if d == nil {
		panic("data.NewDataLoader: dataset must be non-nil")
	}
	if batchSize <= 0 {
		panic("data.NewDataLoader: batchSize must be > 0")
	}
	return &DataLoader{Dataset: d, BatchSize: batchSize, Shuffle: shuffle}
}

// Len returns the number of batches produced per iteration.
func (dl *DataLoader) Len() int {
	n := dl.Dataset.Len()
	if dl.DropLast {
		return n / dl.BatchSize
	}
	return (n + dl.BatchSize - 1) / dl.BatchSize
}

// Iter returns a channel of mini-batches. The channel is closed once all
// batches have been emitted. Each call to Iter starts a fresh epoch and
// reshuffles if Shuffle is true.
func (dl *DataLoader) Iter() <-chan Batch {
	out := make(chan Batch)

	n := dl.Dataset.Len()
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	if dl.Shuffle {
		var r *rand.Rand
		if dl.Seed != 0 {
			r = rand.New(rand.NewSource(dl.Seed))
		} else {
			r = rand.New(rand.NewSource(rand.Int63()))
		}
		r.Shuffle(n, func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	}

	go func() {
		defer close(out)
		bs := dl.BatchSize
		for start := 0; start < n; start += bs {
			end := start + bs
			if end > n {
				if dl.DropLast {
					return
				}
				end = n
			}
			batch := collateBatch(dl.Dataset, indices[start:end])
			out <- batch
		}
	}()

	return out
}

// collateBatch fetches samples for the given indices and stacks them into
// a single Batch along a new leading dimension.
func collateBatch(d Dataset, idxs []int) Batch {
	xs := make([]*tensor.Tensor, len(idxs))
	ys := make([]*tensor.Tensor, len(idxs))
	for i, j := range idxs {
		xs[i], ys[i] = d.Get(j)
	}
	return Batch{
		X: stackOrUnsqueeze(xs),
		Y: stackOrUnsqueeze(ys),
	}
}

// stackOrUnsqueeze stacks the given tensors along a new leading axis. If
// the samples are 0-D scalars, the result is a 1-D tensor of length
// len(ts); otherwise the result has shape (len(ts), ...samples[0].Shape).
func stackOrUnsqueeze(ts []*tensor.Tensor) *tensor.Tensor {
	if len(ts) == 0 {
		panic("data: cannot stack empty batch")
	}
	if len(ts[0].Shape) == 0 {
		// 0-D scalars: build a 1-D tensor directly.
		buf := make([]float64, len(ts))
		for i, t := range ts {
			buf[i] = t.Data[0]
		}
		return tensor.New(buf, len(ts))
	}
	return tensor.Stack(0, ts...)
}
