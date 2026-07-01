package distributed

// In-process multi-goroutine "ranks" over localhost: each rank is a goroutine
// with its own model/optimizer, connected through a real TCP star on a free
// 127.0.0.1 port. These exercise the actual wire protocol end to end.

import (
	"fmt"
	"math"
	"net"
	"sync"
	"sync/atomic"
	"testing"

	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

// freeAddr grabs a free localhost port by listening and closing.
func freeAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("freeAddr: %v", err)
	}
	addr := l.Addr().String()
	l.Close()
	return addr
}

// runRanks starts world goroutine-ranks, each Init-ing into the same group
// and running fn; it fails the test on any rank error.
func runRanks(t *testing.T, world int, fn func(rank int, g *Group) error) {
	t.Helper()
	peers := []string{freeAddr(t)}
	errs := make([]error, world)
	var wg sync.WaitGroup
	for r := 0; r < world; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			g, err := Init(r, world, peers)
			if err != nil {
				errs[r] = fmt.Errorf("Init: %w", err)
				return
			}
			defer g.Close()
			errs[r] = fn(r, g)
		}(r)
	}
	wg.Wait()
	for r, err := range errs {
		if err != nil {
			t.Fatalf("rank %d: %v", r, err)
		}
	}
}

// TestAllReduceMeanGrads: 3 ranks with grads (rank+1)*base must all end with
// the exact mean 2*base (integers: exact). One param starts with a nil Grad
// on rank 1 to exercise zero-contribution and allocation.
func TestAllReduceMeanGrads(t *testing.T) {
	const world = 3
	var mu sync.Mutex
	results := make(map[int][][]float64)

	runRanks(t, world, func(rank int, g *Group) error {
		a := tensor.Zeros(2, 2)
		b := tensor.Zeros(3)
		a.Grad = tensor.New([]float64{
			float64(3 * (rank + 1)), float64(-3 * (rank + 1)),
			float64(6 * (rank + 1)), float64(0),
		}, 2, 2)
		if rank != 1 { // rank 1 contributes zeros via nil Grad
			b.Grad = tensor.New([]float64{
				float64(rank + 1), float64(2 * (rank + 1)), float64(4 * (rank + 1)),
			}, 3)
		}
		if err := g.AllReduceMeanGrads([]*tensor.Tensor{a, b}); err != nil {
			return err
		}
		mu.Lock()
		results[rank] = [][]float64{append([]float64(nil), a.Grad.Data...), append([]float64(nil), b.Grad.Data...)}
		mu.Unlock()
		return nil
	})

	// mean over ranks 0,1,2 of k*(r+1) = 2k; b skips rank 1 => (k*1+k*3)/3.
	wantA := []float64{6, -6, 12, 0}
	wantB := []float64{4.0 / 3.0, 8.0 / 3.0, 16.0 / 3.0}
	for rank := 0; rank < world; rank++ {
		got := results[rank]
		for i, w := range wantA {
			if got[0][i] != w {
				t.Errorf("rank %d a.Grad[%d] = %v, want %v", rank, i, got[0][i], w)
			}
		}
		for i, w := range wantB {
			if math.Abs(got[1][i]-w) > 1e-15 {
				t.Errorf("rank %d b.Grad[%d] = %v, want %v", rank, i, got[1][i], w)
			}
		}
	}
	// Bit-identical across ranks (the DDP invariant).
	for rank := 1; rank < world; rank++ {
		for pi := range results[0] {
			for i := range results[0][pi] {
				if results[rank][pi][i] != results[0][pi][i] {
					t.Errorf("rank %d param %d elem %d differs bitwise from rank 0", rank, pi, i)
				}
			}
		}
	}
}

// TestBroadcastParams checks both the direct (root 0) and relayed (root 2)
// broadcast paths.
func TestBroadcastParams(t *testing.T) {
	for _, root := range []int{0, 2} {
		root := root
		var mu sync.Mutex
		results := make(map[int][]float64)
		runRanks(t, 3, func(rank int, g *Group) error {
			// Every rank starts with different values; root's are 10*rank-derived.
			p := tensor.New([]float64{
				float64(100*rank + 1), float64(100*rank + 2), float64(100*rank + 3), float64(100*rank + 4),
			}, 2, 2)
			if err := g.BroadcastParams([]*tensor.Tensor{p}, root); err != nil {
				return err
			}
			mu.Lock()
			results[rank] = append([]float64(nil), p.Data...)
			mu.Unlock()
			return nil
		})
		want := []float64{float64(100*root + 1), float64(100*root + 2), float64(100*root + 3), float64(100*root + 4)}
		for rank := 0; rank < 3; rank++ {
			for i, w := range want {
				if results[rank][i] != w {
					t.Errorf("root %d: rank %d Data[%d] = %v, want %v", root, rank, i, results[rank][i], w)
				}
			}
		}
	}
}

// TestBarrier: no rank may observe fewer arrivals than worldSize after the
// barrier releases — every rank increments the counter strictly before
// entering the barrier, so a correct barrier guarantees the full count.
func TestBarrier(t *testing.T) {
	const world = 4
	var arrivals int32
	runRanks(t, world, func(rank int, g *Group) error {
		atomic.AddInt32(&arrivals, 1)
		if err := g.Barrier(); err != nil {
			return err
		}
		if n := atomic.LoadInt32(&arrivals); n != world {
			return fmt.Errorf("barrier released after only %d/%d ranks arrived", n, world)
		}
		return nil
	})
}

// TestSingleRankGroupNoOps: worldSize 1 must work without any networking.
func TestSingleRankGroupNoOps(t *testing.T) {
	g, err := Init(0, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	p := tensor.Ones(2)
	p.Grad = tensor.New([]float64{3, 5}, 2)
	if err := g.AllReduceMeanGrads([]*tensor.Tensor{p}); err != nil {
		t.Fatal(err)
	}
	if p.Grad.Data[0] != 3 || p.Grad.Data[1] != 5 {
		t.Errorf("single-rank all-reduce changed grads: %v", p.Grad.Data)
	}
	if err := g.BroadcastParams([]*tensor.Tensor{p}, 0); err != nil {
		t.Fatal(err)
	}
	if err := g.Barrier(); err != nil {
		t.Fatal(err)
	}
}

// newTestLinear builds a Linear(2,1) with FIXED weights so every rank (and
// the single-process reference) starts from the same values.
func newTestLinear() *nn.Linear {
	l := nn.NewLinear(2, 1, true)
	ps := l.Parameters()
	copy(ps[0].Data, []float64{0.5, -0.25}) // weight
	copy(ps[1].Data, []float64{0.1})        // bias
	return l
}

// TestDDPTrainingMatchesFullBatch: 2 ranks train the same tiny linear model
// on equal split shards with the DDP pattern. Invariants per step:
//   - parameters are BIT-IDENTICAL across ranks;
//   - they match a single-process full-batch run to 1e-12 (each rank's loss
//     is the mean over its shard; all-reduce MEAN of 2 equal shards equals
//     the full-batch mean).
func TestDDPTrainingMatchesFullBatch(t *testing.T) {
	const (
		world = 2
		steps = 4
		lr    = 0.1
	)
	// Deterministic data: 8 rows, 2 features -> 1 target.
	xFull := tensor.New([]float64{
		1, 2, -1, 0.5, 3, -2, 0.25, 1.5,
		-0.5, 2.5, 1, 1, -2, -1, 0.75, -0.25,
	}, 8, 2)
	yFull := tensor.New([]float64{1, -1, 2, 0.5, -0.5, 1.5, -2, 0.25}, 8, 1)
	shard := func(t8 *tensor.Tensor, rank, rowSize int) *tensor.Tensor {
		lo, hi := rank*4*rowSize, (rank+1)*4*rowSize
		d := append([]float64(nil), t8.Data[lo:hi]...)
		return tensor.New(d, 4, rowSize)
	}

	// Single-process full-batch reference.
	ref := newTestLinear()
	refOpt := optim.NewSGD(ref.Parameters(), lr)
	refSnaps := make([][]float64, steps)
	for s := 0; s < steps; s++ {
		refOpt.ZeroGrad()
		nn.MSELoss(ref.Forward(xFull), yFull).Backward()
		refOpt.Step()
		refSnaps[s] = snapshot(ref.Parameters())
	}

	// DDP run: snapshots[rank][step].
	var mu sync.Mutex
	snaps := make(map[int][][]float64)
	runRanks(t, world, func(rank int, g *Group) error {
		model := newTestLinear()
		opt := optim.NewSGD(model.Parameters(), lr)
		// Initial weight sync (a no-op here since init is fixed, but it is
		// part of the DDP pattern and exercises the code path).
		if err := g.BroadcastParams(model.Parameters(), 0); err != nil {
			return err
		}
		x, y := shard(xFull, rank, 2), shard(yFull, rank, 1)
		var mine [][]float64
		for s := 0; s < steps; s++ {
			opt.ZeroGrad()
			nn.MSELoss(model.Forward(x), y).Backward() // mean over the LOCAL shard
			if err := DDPStep(g, opt); err != nil {
				return err
			}
			mine = append(mine, snapshot(model.Parameters()))
		}
		mu.Lock()
		snaps[rank] = mine
		mu.Unlock()
		return nil
	})

	for s := 0; s < steps; s++ {
		// Bit-identical across ranks.
		for i := range snaps[0][s] {
			if snaps[0][s][i] != snaps[1][s][i] {
				t.Errorf("step %d elem %d: rank params differ bitwise: %v vs %v",
					s, i, snaps[0][s][i], snaps[1][s][i])
			}
		}
		// Match the full-batch reference to 1e-12.
		for i := range refSnaps[s] {
			if d := math.Abs(snaps[0][s][i] - refSnaps[s][i]); d > 1e-12 {
				t.Errorf("step %d elem %d: DDP %v vs full-batch %v (diff %g)",
					s, i, snaps[0][s][i], refSnaps[s][i], d)
			}
		}
	}
}

// snapshot flattens current parameter values.
func snapshot(params []*tensor.Tensor) []float64 {
	var out []float64
	for _, p := range params {
		out = append(out, p.Data...)
	}
	return out
}

// TestProtocolMismatchDetected: diverging parameter lists across ranks must
// produce an error, not silent corruption.
func TestProtocolMismatchDetected(t *testing.T) {
	const world = 2
	errs := make([]error, world)
	peers := []string{freeAddr(t)}
	var wg sync.WaitGroup
	for r := 0; r < world; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			g, err := Init(r, world, peers)
			if err != nil {
				errs[r] = err
				return
			}
			defer g.Close()
			// Rank 0 reduces a 4-elem grad, rank 1 a 2-elem grad.
			var p *tensor.Tensor
			if r == 0 {
				p = tensor.Zeros(4)
			} else {
				p = tensor.Zeros(2)
			}
			p.Grad = tensor.Zeros(p.Shape...)
			errs[r] = g.AllReduceMeanGrads([]*tensor.Tensor{p})
		}(r)
	}
	wg.Wait()
	if errs[0] == nil && errs[1] == nil {
		t.Fatal("mismatched parameter lists were not detected")
	}
}
