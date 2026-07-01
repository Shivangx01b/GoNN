package parallel

import (
	"math"
	"math/rand"
	"testing"

	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

// seededRandn returns a deterministic N(0,1) tensor from a private stream.
func seededRandn(seed int64, shape ...int) *tensor.Tensor {
	rng := rand.New(rand.NewSource(seed))
	t := tensor.Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = rng.NormFloat64()
	}
	return t
}

// cloneLinear copies src's parameter values into a structurally identical
// fresh Linear so two models start bit-identical.
func cloneLinear(src *nn.Linear, in, out int) *nn.Linear {
	dst := nn.NewLinear(in, out, true)
	sp, dp := src.Parameters(), dst.Parameters()
	for i := range sp {
		copy(dp[i].Data, sp[i].Data)
	}
	return dst
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// TestGradientsMatchFullBatch: gradients accumulated by concurrent per-shard
// Backward calls (losses scaled by shardSize/batchSize) must equal the
// full-batch mean-loss gradients to float-addition-order precision.
func TestGradientsMatchFullBatch(t *testing.T) {
	model := nn.NewLinear(4, 3, true)
	x := seededRandn(101, 8, 4)
	y := seededRandn(102, 8, 3)

	// Reference: single full-batch backward.
	opt := optim.NewSGD(model.Parameters(), 0.1)
	opt.ZeroGrad()
	nn.MSELoss(model.Forward(x), y).Backward()
	ref := make([][]float64, 0)
	for _, p := range model.Parameters() {
		g := make([]float64, len(p.Grad.Data))
		copy(g, p.Grad.Data)
		ref = append(ref, g)
	}

	// DataParallel: 3 uneven shards (3,3,2) via Gradients directly.
	opt.ZeroGrad()
	const workers = 3
	bounds := shardBounds(8, workers)
	Gradients(workers, func(s int) *tensor.Tensor {
		lo, hi := bounds[s], bounds[s+1]
		xs, ys := shardRows(x, lo, hi), shardRows(y, lo, hi)
		return nn.MSELoss(model.Forward(xs), ys).MulScalar(float64(hi-lo) / 8.0)
	})

	for i, p := range model.Parameters() {
		if d := maxAbsDiff(p.Grad.Data, ref[i]); d > 1e-12 {
			t.Errorf("param %d: DataParallel grad differs from full-batch by %g", i, d)
		}
	}
}

// TestStepMatchesFullBatchTraining: dp.Step over several iterations must
// track a sequential full-batch training loop on an identically initialized
// model to 1e-12, and report the same (full-batch mean) loss.
func TestStepMatchesFullBatchTraining(t *testing.T) {
	x := seededRandn(201, 10, 4)
	y := seededRandn(202, 10, 2)

	mA := nn.NewLinear(4, 2, true)
	mB := cloneLinear(mA, 4, 2)
	optA := optim.NewSGD(mA.Parameters(), 0.05)
	optB := optim.NewSGD(mB.Parameters(), 0.05)

	dp := DataParallel{Workers: 4} // shards 3,3,2,2
	for step := 0; step < 5; step++ {
		lossA := dp.Step(optA, x, y, func(xs, ys *tensor.Tensor) *tensor.Tensor {
			return nn.MSELoss(mA.Forward(xs), ys)
		})

		optB.ZeroGrad()
		lb := nn.MSELoss(mB.Forward(x), y)
		lossB := lb.Item()
		lb.Backward()
		optB.Step()

		if math.Abs(lossA-lossB) > 1e-12 {
			t.Fatalf("step %d: dp loss %v != full-batch loss %v", step, lossA, lossB)
		}
		pa, pb := mA.Parameters(), mB.Parameters()
		for i := range pa {
			if d := maxAbsDiff(pa[i].Data, pb[i].Data); d > 1e-12 {
				t.Fatalf("step %d: param %d diverged by %g", step, i, d)
			}
		}
	}
}

// TestStepWorkerClamping: more workers than rows, and Workers <= 0
// (GOMAXPROCS default), must both behave like plain full-batch training.
func TestStepWorkerClamping(t *testing.T) {
	x := seededRandn(301, 2, 3)
	y := seededRandn(302, 2, 1)

	for _, workers := range []int{16, 0} {
		mA := nn.NewLinear(3, 1, true)
		mB := cloneLinear(mA, 3, 1)
		optA := optim.NewSGD(mA.Parameters(), 0.1)
		optB := optim.NewSGD(mB.Parameters(), 0.1)

		dp := DataParallel{Workers: workers}
		lossA := dp.Step(optA, x, y, func(xs, ys *tensor.Tensor) *tensor.Tensor {
			return nn.MSELoss(mA.Forward(xs), ys)
		})

		optB.ZeroGrad()
		lb := nn.MSELoss(mB.Forward(x), y)
		lb.Backward()
		optB.Step()

		if math.Abs(lossA-lb.Item()) > 1e-12 {
			t.Errorf("workers=%d: loss %v != %v", workers, lossA, lb.Item())
		}
		pa, pb := mA.Parameters(), mB.Parameters()
		for i := range pa {
			if d := maxAbsDiff(pa[i].Data, pb[i].Data); d > 1e-12 {
				t.Errorf("workers=%d: param %d diverged by %g", workers, i, d)
			}
		}
	}
}

// TestShardBounds sanity-checks the shard partitioning.
func TestShardBounds(t *testing.T) {
	cases := []struct {
		n, w int
		want []int
	}{
		{8, 3, []int{0, 3, 6, 8}},
		{10, 4, []int{0, 3, 6, 8, 10}},
		{2, 2, []int{0, 1, 2}},
		{5, 1, []int{0, 5}},
	}
	for _, c := range cases {
		got := shardBounds(c.n, c.w)
		if len(got) != len(c.want) {
			t.Fatalf("shardBounds(%d,%d) = %v, want %v", c.n, c.w, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("shardBounds(%d,%d) = %v, want %v", c.n, c.w, got, c.want)
			}
		}
	}
}
