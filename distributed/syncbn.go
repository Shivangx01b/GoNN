package distributed

// SyncBatchNorm: torch.nn.SyncBatchNorm semantics on top of the Group
// collectives. In training mode the per-channel batch statistics are computed
// over the GLOBAL batch (all ranks) with one AllReduceSum, and the backward
// pass all-reduces the two per-channel gradient sums, so both the outputs and
// dx on every rank match a single-process BatchNorm run on the concatenated
// batch. Running statistics are updated from the global statistics, so they
// stay bit-identical across ranks. Eval mode normalizes with the running
// statistics and performs no communication.
//
// # Backward derivation
//
// Per channel, over the global batch of Ntot elements (batch x spatial across
// all ranks): mu = mean(x), var = biased variance, std = sqrt(var + eps),
// xhat = (x - mu)/std, y = gamma*xhat + beta. For the total loss L (the sum of
// the per-rank losses) with upstream gradient dy = dL/dy:
//
//	dL/dbeta  = sum_global(dy)
//	dL/dgamma = sum_global(dy * xhat)
//	dL/dx     = gamma/std * (dy - S1/Ntot - xhat * S2/Ntot)
//	            where S1 = sum_global(dy), S2 = sum_global(dy * xhat)
//
// dx needs the GLOBAL S1 and S2 (each rank's x influenced mu/var, which every
// rank's y depends on), so the backward closure all-reduces the concatenated
// [S1 | S2] vector. dgamma/dbeta are returned as the LOCAL sums: DDP
// all-reduces parameter gradients separately (AllReduceMeanGrads /DDPStep),
// exactly as PyTorch's DDP+SyncBatchNorm divides the work.
//
// # SPMD rules
//
// The training-mode Forward and its Backward each issue one collective, so
// every rank must call Forward and Backward the same number of times in the
// same order (PyTorch DDP imposes the same). A collective failure inside
// Forward/Backward panics — those paths cannot return an error.

import (
	"fmt"
	"math"

	"gonn/nn"
	"gonn/tensor"
)

// syncBatchNorm is the shared core; SyncBatchNorm1d/2d/3d are thin wrappers
// fixing the accepted input ranks. The field set mirrors nn's batchNormNd so
// state_dict layout (weight, bias, running_mean, running_var) is identical.
type syncBatchNorm struct {
	nn.Base
	group       *Group
	NumFeatures int
	Eps         float64
	Momentum    float64
	Weight      *tensor.Tensor // (C,) gamma; registered only when affine
	Bias        *tensor.Tensor // (C,) beta; registered only when affine
	RunMean     *tensor.Tensor // (C,) buffer
	RunVar      *tensor.Tensor // (C,) buffer
}

// newSyncBatchNorm resolves opts through nn's own machinery: it constructs a
// throwaway nn.BatchNorm1d and adopts its resolved Eps/Momentum and its
// freshly initialized weight/bias/running-stat tensors, so defaults,
// initialization, and the affine convention can never drift from
// batchNormNd's. Affine is detected by whether nn registered trainable
// parameters.
func newSyncBatchNorm(group *Group, numFeatures int, opts []nn.NormOpt) syncBatchNorm {
	probe := nn.NewBatchNorm1d(numFeatures, opts...)
	b := syncBatchNorm{
		group:       group,
		NumFeatures: numFeatures,
		Eps:         probe.Eps,
		Momentum:    probe.Momentum,
		Weight:      probe.Weight,
		Bias:        probe.Bias,
		RunMean:     probe.RunMean,
		RunVar:      probe.RunVar,
	}
	if len(probe.Parameters()) > 0 { // affine: weight/bias are trainable
		b.RegisterParam("weight", b.Weight)
		b.RegisterParam("bias", b.Bias)
	}
	b.RegisterBuffer("running_mean", b.RunMean)
	b.RegisterBuffer("running_var", b.RunVar)
	return b
}

// forward is the shared path for channels-first input (N, C, spatial...):
// rank >= 2 with x.Shape[1] == NumFeatures (rank 2 means no spatial dims).
func (b *syncBatchNorm) forward(x *tensor.Tensor) *tensor.Tensor {
	C := b.NumFeatures
	if len(x.Shape) < 2 || x.Shape[1] != C {
		panic(fmt.Sprintf("SyncBatchNorm: expected channels-first input with %d channels, got shape %v", C, x.Shape))
	}
	if !b.Training() {
		return b.forwardEval(x)
	}

	N := x.Shape[0]
	rest := 1
	for i := 2; i < len(x.Shape); i++ {
		rest *= x.Shape[i]
	}

	// Local per-channel sum and sum of squares, plus the local element count,
	// packed into one vector => one collective. The wire-level element-count
	// check doubles as a NumFeatures-consistency check across ranks.
	stats := make([]float64, 2*C+1)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			base := (n*C + c) * rest
			for s := 0; s < rest; s++ {
				v := x.Data[base+s]
				stats[c] += v
				stats[C+c] += v * v
			}
		}
	}
	stats[2*C] = float64(N * rest)
	if err := b.group.AllReduceSum(stats); err != nil {
		panic(fmt.Sprintf("SyncBatchNorm forward: %v", err))
	}

	// Global biased statistics: mean = S/Ntot, var = SS/Ntot - mean^2
	// (PyTorch's SyncBatchNorm computes var from the reduced sums the same way).
	nTot := stats[2*C]
	mean := make([]float64, C)
	biasedVar := make([]float64, C)
	invStd := make([]float64, C)
	for c := 0; c < C; c++ {
		m := stats[c] / nTot
		v := stats[C+c]/nTot - m*m
		if v < 0 {
			v = 0 // guard float cancellation for near-constant channels
		}
		mean[c] = m
		biasedVar[c] = v
		invStd[c] = 1 / math.Sqrt(v+b.Eps)
	}

	// y = gamma * xhat + beta with xhat = (x - mean) * invStd; xhat is kept
	// for the backward closure.
	xhat := make([]float64, len(x.Data))
	outData := make([]float64, len(x.Data))
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			base := (n*C + c) * rest
			gm, bt := b.Weight.Data[c], b.Bias.Data[c]
			mu, is := mean[c], invStd[c]
			for s := 0; s < rest; s++ {
				h := (x.Data[base+s] - mu) * is
				xhat[base+s] = h
				outData[base+s] = gm*h + bt
			}
		}
	}
	out := tensor.New(outData, x.Shape...)

	// Running stats: GLOBAL mean and GLOBAL UNBIASED variance (Bessel with
	// Ntot), folded in with Momentum — same convention as batchNormNd's
	// updateRunningStats, but over the global batch, so the buffers stay
	// bit-identical on every rank.
	unbiased := 1.0
	if nTot > 1 {
		unbiased = nTot / (nTot - 1)
	}
	for c := 0; c < C; c++ {
		b.RunMean.Data[c] = (1-b.Momentum)*b.RunMean.Data[c] + b.Momentum*mean[c]
		b.RunVar.Data[c] = (1-b.Momentum)*b.RunVar.Data[c] + b.Momentum*biasedVar[c]*unbiased
	}

	group := b.group
	weight := b.Weight
	tensor.MakeNode(out, "SyncBatchNorm", []*tensor.Tensor{x, b.Weight, b.Bias}, func(grad *tensor.Tensor) []*tensor.Tensor {
		// Local per-channel sums: red = [S1 | S2] = [sum(dy) | sum(dy*xhat)].
		red := make([]float64, 2*C)
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				base := (n*C + c) * rest
				for s := 0; s < rest; s++ {
					dy := grad.Data[base+s]
					red[c] += dy
					red[C+c] += dy * xhat[base+s]
				}
			}
		}
		// Parameter grads stay LOCAL (copied before the in-place all-reduce):
		// DDP all-reduces parameter gradients separately, like PyTorch.
		dbeta := tensor.New(append([]float64(nil), red[:C]...), C)
		dgamma := tensor.New(append([]float64(nil), red[C:]...), C)
		// dx needs the GLOBAL sums — every rank must run Backward in step.
		if err := group.AllReduceSum(red); err != nil {
			panic(fmt.Sprintf("SyncBatchNorm backward: %v", err))
		}
		dxData := make([]float64, len(grad.Data))
		for n := 0; n < N; n++ {
			for c := 0; c < C; c++ {
				base := (n*C + c) * rest
				k := weight.Data[c] * invStd[c]
				m1 := red[c] / nTot
				m2 := red[C+c] / nTot
				for s := 0; s < rest; s++ {
					dxData[base+s] = k * (grad.Data[base+s] - m1 - xhat[base+s]*m2)
				}
			}
		}
		dx := tensor.New(dxData, x.Shape...)
		return []*tensor.Tensor{dx, dgamma, dbeta}
	})
	return out
}

// forwardEval normalizes with the running statistics — no collectives, pure
// tensor ops, mirroring batchNormNd's eval path op for op (including the
// permute dance for spatial inputs) so eval outputs are bit-identical to
// nn.BatchNorm*'s.
func (b *syncBatchNorm) forwardEval(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) == 2 {
		return b.eval2D(x)
	}
	N, C := x.Shape[0], x.Shape[1]
	rest := 1
	for i := 2; i < len(x.Shape); i++ {
		rest *= x.Shape[i]
	}
	xp := x.Reshape(N, C, rest).Permute(0, 2, 1).Reshape(N*rest, C)
	out := b.eval2D(xp)
	return out.Reshape(N, rest, C).Permute(0, 2, 1).Reshape(x.Shape...)
}

func (b *syncBatchNorm) eval2D(x *tensor.Tensor) *tensor.Tensor {
	mean := b.RunMean.Reshape(1, b.NumFeatures)
	v := b.RunVar.Reshape(1, b.NumFeatures)
	xc := x.Sub(mean)
	std := v.AddScalar(b.Eps).Sqrt()
	norm := xc.Div(std)
	return norm.Mul(b.Weight).Add(b.Bias)
}

// SyncBatchNorm1d is nn.BatchNorm1d with cross-rank statistics: (N, C) or
// (N, C, L) input, normalized per channel over the GLOBAL batch (and length).
type SyncBatchNorm1d struct{ syncBatchNorm }

// NewSyncBatchNorm1d constructs a SyncBatchNorm1d with C features over group.
func NewSyncBatchNorm1d(group *Group, numFeatures int, opts ...nn.NormOpt) *SyncBatchNorm1d {
	return &SyncBatchNorm1d{newSyncBatchNorm(group, numFeatures, opts)}
}

// Forward applies synchronized batch norm over the channel dim.
func (b *SyncBatchNorm1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 2 && len(x.Shape) != 3 {
		panic(fmt.Sprintf("SyncBatchNorm1d: expected 2D or 3D input, got shape %v", x.Shape))
	}
	return b.forward(x)
}

// SyncBatchNorm2d is nn.BatchNorm2d with cross-rank statistics: (N, C, H, W)
// input, normalized per channel over the GLOBAL N, H, W.
type SyncBatchNorm2d struct{ syncBatchNorm }

// NewSyncBatchNorm2d constructs a SyncBatchNorm2d with C features over group.
func NewSyncBatchNorm2d(group *Group, numFeatures int, opts ...nn.NormOpt) *SyncBatchNorm2d {
	return &SyncBatchNorm2d{newSyncBatchNorm(group, numFeatures, opts)}
}

// Forward applies synchronized batch norm over (N, H, W) per channel.
func (b *SyncBatchNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic(fmt.Sprintf("SyncBatchNorm2d: expected 4D input, got shape %v", x.Shape))
	}
	return b.forward(x)
}

// SyncBatchNorm3d is BatchNorm3d with cross-rank statistics: (N, C, D, H, W)
// input, normalized per channel over the GLOBAL N, D, H, W.
type SyncBatchNorm3d struct{ syncBatchNorm }

// NewSyncBatchNorm3d constructs a SyncBatchNorm3d with C features over group.
func NewSyncBatchNorm3d(group *Group, numFeatures int, opts ...nn.NormOpt) *SyncBatchNorm3d {
	return &SyncBatchNorm3d{newSyncBatchNorm(group, numFeatures, opts)}
}

// Forward applies synchronized batch norm over (N, D, H, W) per channel.
func (b *SyncBatchNorm3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic(fmt.Sprintf("SyncBatchNorm3d: expected 5D input, got shape %v", x.Shape))
	}
	return b.forward(x)
}

var (
	_ nn.Module = (*SyncBatchNorm1d)(nil)
	_ nn.Module = (*SyncBatchNorm2d)(nil)
	_ nn.Module = (*SyncBatchNorm3d)(nil)
)
