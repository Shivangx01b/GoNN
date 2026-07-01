package distributed

// SyncBatchNorm tests: in-process goroutine ranks over a real localhost TCP
// star (same harness as distributed_test.go). The key invariant: K-rank
// SyncBatchNorm on batch shards must reproduce single-process nn.BatchNorm on
// the full batch — forward outputs, dx, summed dgamma/dbeta, and running
// statistics.

import (
	"math"
	"sync"
	"testing"

	"gonn/nn"
	"gonn/tensor"
)

// synthData produces deterministic O(1)-magnitude values with per-channel
// structure (nothing degenerate: no constant channels, no zero mean).
func synthData(n int) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = math.Sin(float64(i)*0.7+0.3)*1.4 + 0.2*math.Cos(float64(i)*1.9) + 0.05*float64(i%7)
	}
	return d
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

func world1Group(t *testing.T) *Group {
	t.Helper()
	g, err := Init(0, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestAllReduceSum: 3 ranks contribute (rank+1)*(i+1); the sum 6*(i+1) is
// exact in float64 and must be bit-identical on every rank.
func TestAllReduceSum(t *testing.T) {
	const world, n = 3, 5
	var mu sync.Mutex
	results := make(map[int][]float64)
	runRanks(t, world, func(rank int, g *Group) error {
		vals := make([]float64, n)
		for i := range vals {
			vals[i] = float64((rank + 1) * (i + 1))
		}
		if err := g.AllReduceSum(vals); err != nil {
			return err
		}
		mu.Lock()
		results[rank] = vals
		mu.Unlock()
		return nil
	})
	for rank := 0; rank < world; rank++ {
		for i := 0; i < n; i++ {
			if want := float64(6 * (i + 1)); results[rank][i] != want {
				t.Errorf("rank %d vals[%d] = %v, want %v", rank, i, results[rank][i], want)
			}
			if results[rank][i] != results[0][i] {
				t.Errorf("rank %d vals[%d] differs bitwise from rank 0", rank, i)
			}
		}
	}
}

// TestAllReduceSumWorldSize1 must be the identity with no networking.
func TestAllReduceSumWorldSize1(t *testing.T) {
	g := world1Group(t)
	defer g.Close()
	vals := []float64{3, -1.5, 0}
	if err := g.AllReduceSum(vals); err != nil {
		t.Fatal(err)
	}
	if vals[0] != 3 || vals[1] != -1.5 || vals[2] != 0 {
		t.Errorf("single-rank AllReduceSum changed vals: %v", vals)
	}
}

// syncBNResult is one rank's recorded state from the key test.
type syncBNResult struct {
	y, dx, dgamma, dbeta, runMean, runVar, yEval []float64
}

// TestSyncBatchNorm2dMatchesFullBatch — THE KEY TEST. 2 ranks each hold half
// of an (8,3,2,2) batch. Per rank: y = SyncBatchNorm2d(x_half), loss =
// sum(y^2)/96 (96 = FULL-batch numel, so the cross-rank sum of losses equals
// the single-process mean-of-squares loss and each rank's upstream dy matches
// the single-process dy on its half). Invariants vs single-process
// nn.BatchNorm2d on the full batch:
//
//   - forward outputs equal the corresponding half to 1e-12;
//   - dx equals the corresponding half of the full-batch gradient to 1e-10;
//   - dgamma/dbeta summed across ranks equal the single-process grads;
//   - running stats are bit-identical across ranks and match single-process;
//   - eval mode reproduces single-process eval WITH THE GROUP CLOSED,
//     proving the eval path issues no collectives.
func TestSyncBatchNorm2dMatchesFullBatch(t *testing.T) {
	const (
		world          = 2
		nFull, C, H, W = 8, 3, 2, 2
		full           = nFull * C * H * W // 96
		half           = full / world
	)
	fullData := synthData(full)
	gamma := []float64{1.5, 0.8, -1.2}
	beta := []float64{0.3, -0.5, 0.9}

	// Single-process full-batch reference.
	bn := nn.NewBatchNorm2d(C)
	copy(bn.Weight.Data, gamma)
	copy(bn.Bias.Data, beta)
	xFull := tensor.New(append([]float64(nil), fullData...), nFull, C, H, W).SetRequiresGrad(true)
	yFull := bn.Forward(xFull)
	yFull.Square().Sum().MulScalar(1.0 / float64(full)).Backward()
	bn.Eval()
	yFullEval := bn.Forward(tensor.New(append([]float64(nil), fullData...), nFull, C, H, W))

	// 2-rank SyncBatchNorm run on the batch halves.
	var mu sync.Mutex
	res := make(map[int]*syncBNResult)
	runRanks(t, world, func(rank int, g *Group) error {
		sbn := NewSyncBatchNorm2d(g, C)
		copy(sbn.Weight.Data, gamma)
		copy(sbn.Bias.Data, beta)
		shard := append([]float64(nil), fullData[rank*half:(rank+1)*half]...)
		x := tensor.New(shard, nFull/world, C, H, W).SetRequiresGrad(true)
		y := sbn.Forward(x)
		// Per-rank loss = sum(y_local^2)/N_total_elems, so summed cross-rank
		// losses equal the single-process mean loss.
		y.Square().Sum().MulScalar(1.0 / float64(full)).Backward()

		r := &syncBNResult{
			y:       append([]float64(nil), y.Data...),
			dx:      append([]float64(nil), x.Grad.Data...),
			dgamma:  append([]float64(nil), sbn.Weight.Grad.Data...),
			dbeta:   append([]float64(nil), sbn.Bias.Grad.Data...),
			runMean: append([]float64(nil), sbn.RunMean.Data...),
			runVar:  append([]float64(nil), sbn.RunVar.Data...),
		}

		// Eval with the group CLOSED: any collective would error (panic), so
		// passing proves eval mode communicates nothing.
		if err := g.Barrier(); err != nil {
			return err
		}
		g.Close()
		sbn.Eval()
		r.yEval = append([]float64(nil), sbn.Forward(tensor.New(append([]float64(nil), shard...), nFull/world, C, H, W)).Data...)

		mu.Lock()
		res[rank] = r
		mu.Unlock()
		return nil
	})

	for rank := 0; rank < world; rank++ {
		r := res[rank]
		lo, hi := rank*half, (rank+1)*half
		if d := maxAbsDiff(r.y, yFull.Data[lo:hi]); d > 1e-12 {
			t.Errorf("rank %d forward output vs full-batch BN: max diff %g > 1e-12", rank, d)
		}
		if d := maxAbsDiff(r.dx, xFull.Grad.Data[lo:hi]); d > 1e-10 {
			t.Errorf("rank %d dx vs full-batch BN: max diff %g > 1e-10", rank, d)
		}
		if d := maxAbsDiff(r.runMean, bn.RunMean.Data); d > 1e-12 {
			t.Errorf("rank %d running mean vs full-batch BN: max diff %g > 1e-12", rank, d)
		}
		if d := maxAbsDiff(r.runVar, bn.RunVar.Data); d > 1e-12 {
			t.Errorf("rank %d running var vs full-batch BN: max diff %g > 1e-12", rank, d)
		}
		if d := maxAbsDiff(r.yEval, yFullEval.Data[lo:hi]); d > 1e-12 {
			t.Errorf("rank %d eval output vs full-batch BN eval: max diff %g > 1e-12", rank, d)
		}
		// Running stats bit-identical across ranks (the DDP buffer invariant).
		for c := 0; c < C; c++ {
			if r.runMean[c] != res[0].runMean[c] || r.runVar[c] != res[0].runVar[c] {
				t.Errorf("rank %d running stats differ bitwise from rank 0 at channel %d", rank, c)
			}
		}
	}

	// dgamma/dbeta are LOCAL sums; the manual cross-rank sum must match the
	// single-process parameter gradients (this is what AllReduceMeanGrads
	// would reduce — PyTorch DDP's division of labor).
	sumG := make([]float64, C)
	sumB := make([]float64, C)
	for rank := 0; rank < world; rank++ {
		for c := 0; c < C; c++ {
			sumG[c] += res[rank].dgamma[c]
			sumB[c] += res[rank].dbeta[c]
		}
	}
	if d := maxAbsDiff(sumG, bn.Weight.Grad.Data); d > 1e-10 {
		t.Errorf("summed dgamma vs full-batch BN: max diff %g > 1e-10", d)
	}
	if d := maxAbsDiff(sumB, bn.Bias.Grad.Data); d > 1e-10 {
		t.Errorf("summed dbeta vs full-batch BN: max diff %g > 1e-10", d)
	}
}

// TestSyncBatchNormWorldSize1MatchesBatchNorm: with a degenerate group,
// SyncBatchNorm must be plain BatchNorm — forward, backward, running stats.
func TestSyncBatchNormWorldSize1MatchesBatchNorm(t *testing.T) {
	g := world1Group(t)
	defer g.Close()
	const n, C, H, W = 4, 3, 2, 2
	data := synthData(n * C * H * W)
	gamma := []float64{1.5, 0.8, -1.2}
	beta := []float64{0.3, -0.5, 0.9}

	sbn := NewSyncBatchNorm2d(g, C)
	bn := nn.NewBatchNorm2d(C)
	copy(sbn.Weight.Data, gamma)
	copy(sbn.Bias.Data, beta)
	copy(bn.Weight.Data, gamma)
	copy(bn.Bias.Data, beta)

	xs := tensor.New(append([]float64(nil), data...), n, C, H, W).SetRequiresGrad(true)
	xb := tensor.New(append([]float64(nil), data...), n, C, H, W).SetRequiresGrad(true)
	ys := sbn.Forward(xs)
	yb := bn.Forward(xb)
	if d := maxAbsDiff(ys.Data, yb.Data); d > 1e-12 {
		t.Errorf("world-1 forward vs BatchNorm2d: max diff %g > 1e-12", d)
	}

	total := float64(n * C * H * W)
	ys.Square().Sum().MulScalar(1 / total).Backward()
	yb.Square().Sum().MulScalar(1 / total).Backward()
	if d := maxAbsDiff(xs.Grad.Data, xb.Grad.Data); d > 1e-10 {
		t.Errorf("world-1 dx vs BatchNorm2d: max diff %g > 1e-10", d)
	}
	if d := maxAbsDiff(sbn.Weight.Grad.Data, bn.Weight.Grad.Data); d > 1e-10 {
		t.Errorf("world-1 dgamma vs BatchNorm2d: max diff %g > 1e-10", d)
	}
	if d := maxAbsDiff(sbn.Bias.Grad.Data, bn.Bias.Grad.Data); d > 1e-10 {
		t.Errorf("world-1 dbeta vs BatchNorm2d: max diff %g > 1e-10", d)
	}
	if d := maxAbsDiff(sbn.RunMean.Data, bn.RunMean.Data); d > 1e-12 {
		t.Errorf("world-1 running mean vs BatchNorm2d: max diff %g > 1e-12", d)
	}
	if d := maxAbsDiff(sbn.RunVar.Data, bn.RunVar.Data); d > 1e-12 {
		t.Errorf("world-1 running var vs BatchNorm2d: max diff %g > 1e-12", d)
	}

	// State-dict surface matches batchNormNd's exactly.
	np := sbn.NamedParameters()
	if len(np) != 2 || np[0].Name != "weight" || np[1].Name != "bias" {
		t.Errorf("NamedParameters = %v, want [weight bias]", np)
	}
	bufs := sbn.Buffers()
	if len(bufs) != 2 || bufs[0].Name != "running_mean" || bufs[1].Name != "running_var" {
		t.Errorf("Buffers = %v, want [running_mean running_var]", bufs)
	}

	// Opts flow through nn's resolver; non-affine registers no parameters.
	custom := NewSyncBatchNorm2d(g, C, nn.WithEps(1e-3), nn.WithMomentum(0.2), nn.WithAffine(false))
	if custom.Eps != 1e-3 || custom.Momentum != 0.2 {
		t.Errorf("opts not honored: eps %v momentum %v", custom.Eps, custom.Momentum)
	}
	if len(custom.Parameters()) != 0 {
		t.Errorf("non-affine SyncBatchNorm registered %d params, want 0", len(custom.Parameters()))
	}
}

// TestSyncBatchNormGradcheck: analytic dx vs central differences of the
// training-mode forward loss on a world-1 group.
func TestSyncBatchNormGradcheck(t *testing.T) {
	g := world1Group(t)
	defer g.Close()
	const n, C, H, W = 2, 2, 2, 2
	total := n * C * H * W
	data := synthData(total)

	sbn := NewSyncBatchNorm2d(g, C)
	copy(sbn.Weight.Data, []float64{1.3, -0.7})
	copy(sbn.Bias.Data, []float64{0.2, 0.5})

	loss := func(d []float64) float64 {
		x := tensor.New(append([]float64(nil), d...), n, C, H, W)
		y := sbn.Forward(x)
		return y.Square().Sum().Data[0] / float64(total)
	}

	x := tensor.New(append([]float64(nil), data...), n, C, H, W).SetRequiresGrad(true)
	sbn.Forward(x).Square().Sum().MulScalar(1 / float64(total)).Backward()

	const h = 1e-6
	for i := 0; i < total; i++ {
		pert := append([]float64(nil), data...)
		pert[i] = data[i] + h
		up := loss(pert)
		pert[i] = data[i] - h
		down := loss(pert)
		num := (up - down) / (2 * h)
		if d := math.Abs(num - x.Grad.Data[i]); d > 1e-7 {
			t.Errorf("gradcheck dx[%d]: analytic %v vs numeric %v (diff %g)", i, x.Grad.Data[i], num, d)
		}
	}
}

// TestSyncBatchNormEvalMatchesBatchNorm: with hand-set running statistics the
// eval path must reproduce nn.BatchNorm2d eval exactly (identical op
// sequence).
func TestSyncBatchNormEvalMatchesBatchNorm(t *testing.T) {
	g := world1Group(t)
	defer g.Close()
	const n, C, H, W = 3, 3, 2, 2
	data := synthData(n * C * H * W)
	gamma := []float64{1.5, 0.8, -1.2}
	beta := []float64{0.3, -0.5, 0.9}
	runMean := []float64{0.3, -0.2, 0.5}
	runVar := []float64{1.5, 0.7, 2.0}

	sbn := NewSyncBatchNorm2d(g, C)
	bn := nn.NewBatchNorm2d(C)
	copy(sbn.Weight.Data, gamma)
	copy(sbn.Bias.Data, beta)
	copy(sbn.RunMean.Data, runMean)
	copy(sbn.RunVar.Data, runVar)
	copy(bn.Weight.Data, gamma)
	copy(bn.Bias.Data, beta)
	copy(bn.RunMean.Data, runMean)
	copy(bn.RunVar.Data, runVar)
	sbn.Eval()
	bn.Eval()

	ys := sbn.Forward(tensor.New(append([]float64(nil), data...), n, C, H, W))
	yb := bn.Forward(tensor.New(append([]float64(nil), data...), n, C, H, W))
	for i := range ys.Data {
		if ys.Data[i] != yb.Data[i] {
			t.Errorf("eval output[%d] = %v, want %v (bitwise)", i, ys.Data[i], yb.Data[i])
		}
	}
	// Running stats untouched by eval.
	if d := maxAbsDiff(sbn.RunMean.Data, runMean); d != 0 {
		t.Errorf("eval modified running mean")
	}
}

// TestSyncBatchNorm1d3dShapes: the 1d wrapper matches nn.BatchNorm1d on both
// its accepted ranks, and 3d reduces exactly like 1d on the flattened spatial
// volume (the defining property of BatchNorm3d).
func TestSyncBatchNorm1d3dShapes(t *testing.T) {
	g := world1Group(t)
	defer g.Close()
	const C = 2
	gamma := []float64{1.4, -0.6}
	beta := []float64{-0.1, 0.7}

	// 1d on (N, C).
	d2 := synthData(5 * C)
	s1 := NewSyncBatchNorm1d(g, C)
	b1 := nn.NewBatchNorm1d(C)
	copy(s1.Weight.Data, gamma)
	copy(s1.Bias.Data, beta)
	copy(b1.Weight.Data, gamma)
	copy(b1.Bias.Data, beta)
	ys := s1.Forward(tensor.New(append([]float64(nil), d2...), 5, C))
	yb := b1.Forward(tensor.New(append([]float64(nil), d2...), 5, C))
	if d := maxAbsDiff(ys.Data, yb.Data); d > 1e-12 {
		t.Errorf("1d (N,C) vs BatchNorm1d: max diff %g > 1e-12", d)
	}

	// 1d on (N, C, L).
	d3 := synthData(4 * C * 3)
	s1b := NewSyncBatchNorm1d(g, C)
	b1b := nn.NewBatchNorm1d(C)
	copy(s1b.Weight.Data, gamma)
	copy(s1b.Bias.Data, beta)
	copy(b1b.Weight.Data, gamma)
	copy(b1b.Bias.Data, beta)
	ys = s1b.Forward(tensor.New(append([]float64(nil), d3...), 4, C, 3))
	yb = b1b.Forward(tensor.New(append([]float64(nil), d3...), 4, C, 3))
	if d := maxAbsDiff(ys.Data, yb.Data); d > 1e-12 {
		t.Errorf("1d (N,C,L) vs BatchNorm1d: max diff %g > 1e-12", d)
	}

	// 3d on (N, C, D, H, W) == 1d on (N, C, D*H*W) applied to the same bytes.
	d5 := synthData(2 * C * 2 * 3 * 2)
	s3 := NewSyncBatchNorm3d(g, C)
	b1c := nn.NewBatchNorm1d(C)
	copy(s3.Weight.Data, gamma)
	copy(s3.Bias.Data, beta)
	copy(b1c.Weight.Data, gamma)
	copy(b1c.Bias.Data, beta)
	ys = s3.Forward(tensor.New(append([]float64(nil), d5...), 2, C, 2, 3, 2))
	yb = b1c.Forward(tensor.New(append([]float64(nil), d5...), 2, C, 2*3*2))
	if d := maxAbsDiff(ys.Data, yb.Data); d > 1e-12 {
		t.Errorf("3d vs flattened BatchNorm1d: max diff %g > 1e-12", d)
	}
	if d := maxAbsDiff(s3.RunVar.Data, b1c.RunVar.Data); d > 1e-12 {
		t.Errorf("3d running var vs flattened BatchNorm1d: max diff %g > 1e-12", d)
	}
}
