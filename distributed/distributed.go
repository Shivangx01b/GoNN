// Package distributed implements multi-process data-parallel training over
// TCP: an honest Go adaptation of torch.nn.parallel.DistributedDataParallel's
// gradient-averaging semantics, without the GPU/NCCL machinery.
//
// # Topology
//
// A Group is a STAR: rank 0 is the coordinator. Init on rank 0 listens on
// peers[0] and accepts one connection per non-zero rank; every other rank
// dials peers[0] (with retry, so start order does not matter) and identifies
// itself with a rank handshake. All collectives route through rank 0:
// AllReduceMeanGrads is gather-to-root, mean at root, broadcast — O(worldSize)
// messages through the coordinator rather than a ring. That is the right
// trade for the small world sizes this targets (a handful of CPU boxes); a
// ring all-reduce would halve coordinator bandwidth at the cost of worldSize
// serialized hops. Only peers[0] is consulted; further entries are accepted
// (and ignored) so callers may pass a full rank->address table.
//
// # Wire format
//
// Raw binary, deterministic (no encoding/gob type metadata): every collective
// message is
//
//	[1 byte op] [8 bytes big-endian element count] [count x 8 bytes big-endian IEEE-754 float64]
//
// Rank 0 reduces contributions in rank order (1, 2, ..., worldSize-1), so the
// reduction is deterministic run to run, and every rank — including rank 0 —
// applies the exact same averaged bytes, so parameters stay bit-identical
// across ranks after each optimizer step.
//
// # DDP training pattern
//
// After Backward on each rank, all-reduce the mean of the gradients, then
// step the optimizer — every rank computes the same update. With equal shard
// sizes and a mean-reduced per-rank loss, the all-reduced mean gradient
// equals the full-batch mean-loss gradient, so K-rank DDP training matches a
// single-process full-batch run (to float addition order):
//
//	// Identical program on every rank; only rank/shard differ.
//	group, err := distributed.Init(rank, worldSize, []string{"10.0.0.1:29500"})
//	if err != nil { log.Fatal(err) }
//	defer group.Close()
//
//	model := nn.NewSequential(nn.NewLinear(4, 16, true), nn.ReLU(), nn.NewLinear(16, 1, true))
//	opt := optim.NewSGD(model.Parameters(), 0.1)
//
//	// Initial weight sync: every rank adopts rank 0's parameters
//	// (DDP constructor semantics).
//	if err := group.BroadcastParams(model.Parameters(), 0); err != nil { log.Fatal(err) }
//
//	for epoch := 0; epoch < 100; epoch++ {
//		xShard, yShard := shardForRank(rank) // equal-size shards
//		opt.ZeroGrad()
//		loss := nn.MSELoss(model.Forward(xShard), yShard) // mean over the LOCAL shard
//		loss.Backward()
//		if err := distributed.DDPStep(group, opt); err != nil { // all-reduce mean grads + opt.Step()
//			log.Fatal(err)
//		}
//	}
//
// # SyncBatchNorm
//
// Plain nn.BatchNorm* under DDP normalizes with shard-local statistics and
// lets running stats drift apart across ranks. SyncBatchNorm{1,2,3}d (see
// syncbn.go) fixes both, matching torch.nn.SyncBatchNorm: in training mode
// the per-channel mean/variance are computed over the GLOBAL batch via
// AllReduceSum, running statistics are updated from those global statistics
// (so they stay bit-identical on every rank), and the backward pass
// all-reduces the two per-channel gradient sums so dx matches a
// single-process run on the concatenated batch. Convert a model by
// constructing the norm layers with the group — same per-rank architecture,
// exactly like torch.nn.SyncBatchNorm.convert_sync_batchnorm's result:
//
//	bn := distributed.NewSyncBatchNorm2d(group, 64) // instead of nn.NewBatchNorm2d(64)
//
// Because both the training-mode Forward and its Backward issue collectives,
// every rank must run the same number of forward AND backward passes in the
// same order (the usual SPMD rule; PyTorch DDP requires the same). Eval mode
// uses running statistics only and performs no communication.
//
// # What is NOT implemented (explicitly)
//
//   - Fault tolerance / elasticity: a dropped connection fails the collective
//     with an error; there is no rejoin.
//   - Overlapped bucketed all-reduce (PyTorch's Reducer): gradients are
//     reduced in one flat message after the full Backward, not overlapped
//     with it.
//
// # Usage rules
//
// Collectives are SPMD and blocking: every rank must call the same sequence
// of collective operations with identically shaped parameter lists, or ranks
// deadlock/error. A Group is not safe for concurrent use by multiple
// goroutines within one process.
package distributed

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net"
	"time"

	"gonn/optim"
	"gonn/tensor"
)

// Collective op codes (wire protocol).
const (
	opAllReduce    byte = 1
	opBroadcast    byte = 2
	opBarrier      byte = 3
	opAllReduceSum byte = 4
)

// handshakeMagic guards against a stray client connecting to the coordinator.
const handshakeMagic uint32 = 0x60DD_DD90

// DialTimeout bounds how long a non-zero rank's Init retries dialing the
// coordinator before giving up (the coordinator may not be listening yet).
var DialTimeout = 15 * time.Second

// Group is a connected process group. See the package doc for topology and
// usage rules.
type Group struct {
	rank  int
	world int
	lis   net.Listener // rank 0 only
	// rank 0: conns[r] is the connection to rank r (index 0 unused).
	// rank > 0: conns[0] is the connection to rank 0.
	conns []net.Conn
}

// Rank returns this process's rank in [0, WorldSize).
func (g *Group) Rank() int { return g.rank }

// WorldSize returns the number of ranks in the group.
func (g *Group) WorldSize() int { return g.world }

// Init connects the process group. rank 0 listens on peers[0] and accepts
// worldSize-1 identified connections; every other rank dials peers[0],
// retrying until DialTimeout so ranks may start in any order. Init returns on
// every rank only after the full star is connected (it doubles as a startup
// barrier). worldSize == 1 yields a degenerate group whose collectives are
// all no-ops, so single-process runs need no special casing.
func Init(rank, worldSize int, peers []string) (*Group, error) {
	if worldSize < 1 {
		return nil, fmt.Errorf("distributed.Init: worldSize %d < 1", worldSize)
	}
	if rank < 0 || rank >= worldSize {
		return nil, fmt.Errorf("distributed.Init: rank %d out of range [0,%d)", rank, worldSize)
	}
	g := &Group{rank: rank, world: worldSize}
	if worldSize == 1 {
		return g, nil
	}
	if len(peers) < 1 {
		return nil, fmt.Errorf("distributed.Init: need peers[0] (coordinator address)")
	}

	if rank == 0 {
		lis, err := net.Listen("tcp", peers[0])
		if err != nil {
			return nil, fmt.Errorf("distributed.Init: rank 0 listen %s: %w", peers[0], err)
		}
		g.lis = lis
		g.conns = make([]net.Conn, worldSize)
		for i := 1; i < worldSize; i++ {
			c, err := lis.Accept()
			if err != nil {
				g.Close()
				return nil, fmt.Errorf("distributed.Init: accept: %w", err)
			}
			var hs [8]byte
			if _, err := io.ReadFull(c, hs[:]); err != nil {
				g.Close()
				return nil, fmt.Errorf("distributed.Init: handshake read: %w", err)
			}
			if binary.BigEndian.Uint32(hs[:4]) != handshakeMagic {
				g.Close()
				return nil, fmt.Errorf("distributed.Init: bad handshake magic from %s", c.RemoteAddr())
			}
			r := int(binary.BigEndian.Uint32(hs[4:]))
			if r < 1 || r >= worldSize || g.conns[r] != nil {
				g.Close()
				return nil, fmt.Errorf("distributed.Init: invalid or duplicate rank %d in handshake", r)
			}
			g.conns[r] = c
		}
		// Ack every rank so their Init returns only once the whole group is up.
		for r := 1; r < worldSize; r++ {
			if err := writeFrame(g.conns[r], opBarrier, nil); err != nil {
				g.Close()
				return nil, fmt.Errorf("distributed.Init: ack rank %d: %w", r, err)
			}
		}
		return g, nil
	}

	// Non-zero rank: dial the coordinator with retry.
	deadline := time.Now().Add(DialTimeout)
	var c net.Conn
	for {
		var err error
		c, err = net.DialTimeout("tcp", peers[0], time.Second)
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("distributed.Init: rank %d could not reach coordinator %s: %w", rank, peers[0], err)
		}
		time.Sleep(20 * time.Millisecond)
	}
	var hs [8]byte
	binary.BigEndian.PutUint32(hs[:4], handshakeMagic)
	binary.BigEndian.PutUint32(hs[4:], uint32(rank))
	if _, err := c.Write(hs[:]); err != nil {
		c.Close()
		return nil, fmt.Errorf("distributed.Init: handshake write: %w", err)
	}
	if _, err := readFrame(c, opBarrier, 0); err != nil {
		c.Close()
		return nil, fmt.Errorf("distributed.Init: waiting for group ack: %w", err)
	}
	g.conns = []net.Conn{c}
	return g, nil
}

// Close tears down all connections (and rank 0's listener). Collectives on a
// closed group return errors on the remote ranks; Close is not itself a
// collective.
func (g *Group) Close() {
	if g.lis != nil {
		g.lis.Close()
	}
	for _, c := range g.conns {
		if c != nil {
			c.Close()
		}
	}
}

// AllReduceMeanGrads replaces every parameter's gradient with the element-wise
// MEAN of that gradient across all ranks (DDP semantics). A nil Grad
// contributes zeros and is allocated to receive the result. Every rank ends
// up with bit-identical gradient bytes: rank 0 gathers contributions in rank
// order, computes the mean once, applies it locally, and broadcasts the same
// values to everyone.
func (g *Group) AllReduceMeanGrads(params []*tensor.Tensor) error {
	if g.world == 1 {
		return nil // mean over one rank is the identity
	}
	flat := flatten(params, func(p *tensor.Tensor) []float64 {
		if p.Grad == nil {
			return nil // zeros
		}
		return p.Grad.Data
	})

	if g.rank == 0 {
		sum := flat
		for r := 1; r < g.world; r++ { // rank order: deterministic reduction
			v, err := readFrame(g.conns[r], opAllReduce, len(flat))
			if err != nil {
				return fmt.Errorf("distributed.AllReduceMeanGrads: recv from rank %d: %w", r, err)
			}
			for i := range sum {
				sum[i] += v[i]
			}
		}
		inv := 1.0 / float64(g.world)
		for i := range sum {
			sum[i] *= inv
		}
		for r := 1; r < g.world; r++ {
			if err := writeFrame(g.conns[r], opAllReduce, sum); err != nil {
				return fmt.Errorf("distributed.AllReduceMeanGrads: send to rank %d: %w", r, err)
			}
		}
		scatterGrads(params, sum)
		return nil
	}

	if err := writeFrame(g.conns[0], opAllReduce, flat); err != nil {
		return fmt.Errorf("distributed.AllReduceMeanGrads: send: %w", err)
	}
	avg, err := readFrame(g.conns[0], opAllReduce, len(flat))
	if err != nil {
		return fmt.Errorf("distributed.AllReduceMeanGrads: recv: %w", err)
	}
	scatterGrads(params, avg)
	return nil
}

// AllReduceSum replaces vals — in place, on every rank — with the
// element-wise SUM of vals across all ranks (the collective behind
// SyncBatchNorm's global statistics). It mirrors AllReduceMeanGrads'
// structure: gather-to-root in rank order (deterministic reduction), sum at
// rank 0, broadcast; every rank — including rank 0, which applies its own
// summed buffer — ends up with bit-identical bytes. Element counts and op
// codes are checked on every frame, so diverging call sequences or vector
// lengths across ranks fail loudly instead of corrupting data.
func (g *Group) AllReduceSum(vals []float64) error {
	if g.world == 1 {
		return nil // sum over one rank is the identity
	}
	if g.rank == 0 {
		for r := 1; r < g.world; r++ { // rank order: deterministic reduction
			v, err := readFrame(g.conns[r], opAllReduceSum, len(vals))
			if err != nil {
				return fmt.Errorf("distributed.AllReduceSum: recv from rank %d: %w", r, err)
			}
			for i := range vals {
				vals[i] += v[i]
			}
		}
		for r := 1; r < g.world; r++ {
			if err := writeFrame(g.conns[r], opAllReduceSum, vals); err != nil {
				return fmt.Errorf("distributed.AllReduceSum: send to rank %d: %w", r, err)
			}
		}
		return nil
	}
	if err := writeFrame(g.conns[0], opAllReduceSum, vals); err != nil {
		return fmt.Errorf("distributed.AllReduceSum: send: %w", err)
	}
	sum, err := readFrame(g.conns[0], opAllReduceSum, len(vals))
	if err != nil {
		return fmt.Errorf("distributed.AllReduceSum: recv: %w", err)
	}
	copy(vals, sum)
	return nil
}

// BroadcastParams copies every parameter's Data from rank root to all other
// ranks — the initial weight sync of the DDP pattern. With the star topology
// a non-zero root relays through rank 0 (root -> rank 0 -> everyone else).
func (g *Group) BroadcastParams(params []*tensor.Tensor, root int) error {
	if root < 0 || root >= g.world {
		return fmt.Errorf("distributed.BroadcastParams: root %d out of range [0,%d)", root, g.world)
	}
	if g.world == 1 {
		return nil
	}
	n := 0
	for _, p := range params {
		n += p.Numel()
	}

	switch {
	case g.rank == root:
		flat := flatten(params, func(p *tensor.Tensor) []float64 { return p.Data })
		if g.rank == 0 {
			for r := 1; r < g.world; r++ {
				if err := writeFrame(g.conns[r], opBroadcast, flat); err != nil {
					return fmt.Errorf("distributed.BroadcastParams: send to rank %d: %w", r, err)
				}
			}
		} else if err := writeFrame(g.conns[0], opBroadcast, flat); err != nil {
			return fmt.Errorf("distributed.BroadcastParams: send to coordinator: %w", err)
		}
		return nil

	case g.rank == 0: // coordinator relaying a non-zero root
		flat, err := readFrame(g.conns[root], opBroadcast, n)
		if err != nil {
			return fmt.Errorf("distributed.BroadcastParams: recv from root %d: %w", root, err)
		}
		scatterData(params, flat)
		for r := 1; r < g.world; r++ {
			if r == root {
				continue
			}
			if err := writeFrame(g.conns[r], opBroadcast, flat); err != nil {
				return fmt.Errorf("distributed.BroadcastParams: relay to rank %d: %w", r, err)
			}
		}
		return nil

	default: // non-root, non-coordinator: receive from rank 0
		flat, err := readFrame(g.conns[0], opBroadcast, n)
		if err != nil {
			return fmt.Errorf("distributed.BroadcastParams: recv: %w", err)
		}
		scatterData(params, flat)
		return nil
	}
}

// Barrier blocks until every rank has entered it: each non-zero rank sends an
// empty frame to rank 0, which releases everyone once all have arrived.
func (g *Group) Barrier() error {
	if g.world == 1 {
		return nil
	}
	if g.rank == 0 {
		for r := 1; r < g.world; r++ {
			if _, err := readFrame(g.conns[r], opBarrier, 0); err != nil {
				return fmt.Errorf("distributed.Barrier: recv from rank %d: %w", r, err)
			}
		}
		for r := 1; r < g.world; r++ {
			if err := writeFrame(g.conns[r], opBarrier, nil); err != nil {
				return fmt.Errorf("distributed.Barrier: release rank %d: %w", r, err)
			}
		}
		return nil
	}
	if err := writeFrame(g.conns[0], opBarrier, nil); err != nil {
		return fmt.Errorf("distributed.Barrier: send: %w", err)
	}
	if _, err := readFrame(g.conns[0], opBarrier, 0); err != nil {
		return fmt.Errorf("distributed.Barrier: recv: %w", err)
	}
	return nil
}

// DDPStep is the per-iteration DDP convenience: call it after loss.Backward()
// on every rank. It all-reduces the MEAN of the gradients of opt's parameters
// and then runs opt.Step() — every rank applies the identical update. See the
// package doc for the full training-loop example.
func DDPStep(g *Group, opt optim.Optimizer) error {
	if err := g.AllReduceMeanGrads(opt.Parameters()); err != nil {
		return err
	}
	opt.Step()
	return nil
}

// flatten concatenates get(p) for every param into one vector, in param
// order; a nil slice from get contributes p.Numel() zeros.
func flatten(params []*tensor.Tensor, get func(*tensor.Tensor) []float64) []float64 {
	n := 0
	for _, p := range params {
		n += p.Numel()
	}
	flat := make([]float64, n)
	off := 0
	for _, p := range params {
		if d := get(p); d != nil {
			copy(flat[off:off+p.Numel()], d)
		}
		off += p.Numel()
	}
	return flat
}

// scatterGrads writes flat back into the parameters' Grad tensors,
// allocating any that are nil.
func scatterGrads(params []*tensor.Tensor, flat []float64) {
	off := 0
	for _, p := range params {
		if p.Grad == nil {
			p.Grad = tensor.Zeros(p.Shape...)
		}
		copy(p.Grad.Data, flat[off:off+p.Numel()])
		off += p.Numel()
	}
}

// scatterData writes flat back into the parameters' Data.
func scatterData(params []*tensor.Tensor, flat []float64) {
	off := 0
	for _, p := range params {
		copy(p.Data, flat[off:off+p.Numel()])
		off += p.Numel()
	}
}

// writeFrame sends [op][count][count x float64 big-endian] in one Write.
func writeFrame(c net.Conn, op byte, data []float64) error {
	buf := make([]byte, 9+8*len(data))
	buf[0] = op
	binary.BigEndian.PutUint64(buf[1:9], uint64(len(data)))
	for i, v := range data {
		binary.BigEndian.PutUint64(buf[9+8*i:], math.Float64bits(v))
	}
	_, err := c.Write(buf)
	return err
}

// readFrame receives one frame, checking both the op code and the element
// count against expectations (a mismatch means the ranks' collective call
// sequences or parameter lists diverged — fail loudly).
func readFrame(c net.Conn, wantOp byte, wantN int) ([]float64, error) {
	var hdr [9]byte
	if _, err := io.ReadFull(c, hdr[:]); err != nil {
		return nil, err
	}
	if hdr[0] != wantOp {
		return nil, fmt.Errorf("protocol mismatch: got op %d, want %d (collective call sequences diverged across ranks?)", hdr[0], wantOp)
	}
	n := binary.BigEndian.Uint64(hdr[1:])
	if n != uint64(wantN) {
		return nil, fmt.Errorf("protocol mismatch: got %d elements, want %d (parameter lists diverged across ranks?)", n, wantN)
	}
	raw := make([]byte, 8*wantN)
	if _, err := io.ReadFull(c, raw); err != nil {
		return nil, err
	}
	out := make([]float64, wantN)
	for i := range out {
		out[i] = math.Float64frombits(binary.BigEndian.Uint64(raw[8*i:]))
	}
	return out, nil
}
