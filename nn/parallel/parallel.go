// Package parallel provides multi-core CPU data parallelism: an honest Go
// adaptation of torch.nn.DataParallel semantics using goroutines instead of
// GPU replicas.
//
// # How it maps to PyTorch
//
// torch.nn.DataParallel splits the batch along dim 0, runs a model replica
// per device, and sums the replicas' gradients into the original module. In
// GoNN the model needs no replication at all: separate goroutines build
// separate forward graphs over the SAME parameter tensors (interior autograd
// nodes are graph-local; parameters are shared leaves), and concurrent
// Backward calls sum gradients into those shared leaves — serialized by the
// tensor package's leaf-accumulation lock. What PyTorch achieves with
// replicate + gather + grad-sum falls out of the shared-leaf design directly.
//
// # Gradient equivalence
//
// DataParallel.Step scales each shard's loss by shardSize/batchSize. For any
// mean-reduced loss (MSELoss, CrossEntropyLoss, ... — a mean over dim-0
// samples, or over all elements with a constant per-sample element count):
//
//	sum_k (n_k/N) * mean_shard_k(loss) = (1/N) * sum_i loss_i = full-batch mean
//
// Gradients are linear, so the per-shard Backward calls — which SUM into the
// shared parameter leaves — leave each param.Grad equal to the gradient of
// the full-batch mean loss, exactly up to float addition order (~1e-15
// relative). The returned aggregate loss is the full-batch mean loss.
//
// # Example
//
//	model := nn.NewSequential(nn.NewLinear(4, 16, true), nn.ReLU(), nn.NewLinear(16, 1, true))
//	opt := optim.NewSGD(model.Parameters(), 0.1)
//	dp := parallel.DataParallel{Workers: 4}
//	for epoch := 0; epoch < 100; epoch++ {
//		loss := dp.Step(opt, batchX, batchY, func(x, y *tensor.Tensor) *tensor.Tensor {
//			return nn.MSELoss(model.Forward(x), y)
//		})
//		fmt.Println("loss:", loss)
//	}
//
// # Limitations (read before using)
//
//   - The loss closure must be deterministic and mean-reduced. A sum-reduced
//     loss breaks the shardSize/batchSize scaling — use Gradients directly
//     and scale yourself.
//   - BatchNorm running statistics are NOT replicated per shard: concurrent
//     training-mode shards write the shared running-stat buffers without
//     synchronization — a data race the race detector will flag, and the
//     resulting stats are shard statistics, not batch statistics. Put BN
//     layers in eval mode under DataParallel, or prefer GroupNorm/LayerNorm
//     (which have no cross-sample state).
//   - Dropout uses the global math/rand source, which is internally locked —
//     memory-safe, but mask sequences interleave across shards, so runs are
//     not reproducible even with a fixed seed.
//   - This is CPU-core parallelism. The single-GPU acceleration path (backend
//     GEMM dispatch) is orthogonal and composes with it, but true multi-GPU
//     replica parallelism is out of scope here — see gonn/distributed for
//     multi-process data parallelism.
package parallel

import (
	"runtime"
	"sync"

	"gonn/optim"
	"gonn/tensor"
)

// Gradients runs loss(0) .. loss(workers-1) concurrently, one goroutine per
// shard. Each closure must build its own forward graph (over the shared model
// parameters — that is the point) and return a scalar loss tensor; Gradients
// calls Backward on each, so gradients from all shards SUM into the shared
// parameter leaves. It returns the per-shard loss values, indexed by shard.
//
// The caller is responsible for zeroing gradients beforehand (opt.ZeroGrad())
// and for the scaling convention: summed shard gradients equal full-batch
// mean-loss gradients only if each shard loss is pre-scaled by
// shardSize/batchSize (see the package doc). DataParallel.Step wraps all of
// that; use Gradients directly only for custom sharding or scaling schemes
// (e.g. gradient accumulation across calls).
func Gradients(workers int, loss func(shard int) *tensor.Tensor) []float64 {
	losses := make([]float64, workers)
	var wg sync.WaitGroup
	for s := 0; s < workers; s++ {
		wg.Add(1)
		go func(s int) {
			defer wg.Done()
			l := loss(s)
			losses[s] = l.Item()
			l.Backward()
		}(s)
	}
	wg.Wait()
	return losses
}

// DataParallel shards a batch across CPU cores, mirroring
// torch.nn.DataParallel's split-along-dim-0 / sum-gradients semantics.
type DataParallel struct {
	// Workers is the number of shards (goroutines). <= 0 means
	// runtime.GOMAXPROCS(0). It is additionally clamped to the batch size.
	Workers int
}

// Step performs one full optimization step:
//
//  1. opt.ZeroGrad()
//  2. split batchX/batchY along dim 0 into near-equal shards (row copies —
//     the shards share no data or autograd state with the batch tensors)
//  3. run lossFn on every shard concurrently, each loss scaled by
//     shardSize/batchSize, and Backward each (gradients sum into the shared
//     parameters; see the package doc for why that equals the full-batch
//     mean-loss gradient)
//  4. opt.Step() once, on the caller's goroutine
//
// It returns the aggregate (full-batch mean) loss. lossFn must return a
// scalar mean-reduced loss and may run concurrently with itself, so it must
// not mutate shared state beyond the model's forward pass (see the package
// doc's BatchNorm/Dropout caveats).
//
// Because Step includes ZeroGrad, it is a complete training step; it cannot
// be used for gradient accumulation across batches (use Gradients for that).
func (dp DataParallel) Step(opt optim.Optimizer, batchX, batchY *tensor.Tensor, lossFn func(x, y *tensor.Tensor) *tensor.Tensor) float64 {
	if batchX.Dim() < 1 || batchY.Dim() < 1 {
		panic("parallel.Step: batch tensors must have at least 1 dimension")
	}
	n := batchX.Shape[0]
	if batchY.Shape[0] != n {
		panic("parallel.Step: batchX and batchY disagree on batch size (dim 0)")
	}
	if n == 0 {
		panic("parallel.Step: empty batch")
	}
	w := dp.Workers
	if w <= 0 {
		w = runtime.GOMAXPROCS(0)
	}
	if w > n {
		w = n
	}
	bounds := shardBounds(n, w)

	opt.ZeroGrad()
	losses := Gradients(w, func(s int) *tensor.Tensor {
		lo, hi := bounds[s], bounds[s+1]
		x := shardRows(batchX, lo, hi)
		y := shardRows(batchY, lo, hi)
		scale := float64(hi-lo) / float64(n)
		return lossFn(x, y).MulScalar(scale)
	})
	opt.Step()

	total := 0.0
	for _, l := range losses {
		total += l
	}
	return total
}

// shardBounds splits n rows into w near-equal contiguous ranges; the first
// n%w shards get one extra row. Returns w+1 boundaries.
func shardBounds(n, w int) []int {
	bounds := make([]int, w+1)
	base, rem := n/w, n%w
	for s := 0; s < w; s++ {
		size := base
		if s < rem {
			size++
		}
		bounds[s+1] = bounds[s] + size
	}
	return bounds
}

// shardRows copies rows [lo, hi) of t into a fresh tensor with no autograd
// state. Rows are contiguous in row-major layout, so this is a single copy.
// t must be contiguous (every tensor constructor and op output in this
// framework is).
func shardRows(t *tensor.Tensor, lo, hi int) *tensor.Tensor {
	rowSize := 1
	for _, d := range t.Shape[1:] {
		rowSize *= d
	}
	data := make([]float64, (hi-lo)*rowSize)
	copy(data, t.Data[lo*rowSize:hi*rowSize])
	shape := append([]int{hi - lo}, t.Shape[1:]...)
	return tensor.New(data, shape...)
}
