package nn

import (
	"fmt"

	"gonn/tensor"
)

// Sequence helpers shared by the recurrent layers and Seq2Seq. These are
// built on the native indexing/stacking ops (IndexSelect, Stack), which have
// scatter-add backwards — replacing the historical selector-matmul versions
// that cost O(B·F·T) per slice and O(T²) for stacking.

// sliceTime returns x[:, t, :] for a (B, T, F) tensor as (B, F).
func sliceTime(x *tensor.Tensor, t int) *tensor.Tensor {
	B, F := x.Shape[0], x.Shape[2]
	return x.IndexSelect(1, tensor.New([]float64{float64(t)}, 1)).Reshape(B, F)
}

// stackTime stacks T tensors of shape (B, H) into (B, T, H).
func stackTime(hs []*tensor.Tensor, B, T, H int) *tensor.Tensor {
	return tensor.Stack(1, hs...)
}

// sliceCol returns x[:, lo:hi] for a 2D tensor (B, N).
func sliceCol(x *tensor.Tensor, lo, hi int) *tensor.Tensor {
	idx := make([]float64, hi-lo)
	for j := range idx {
		idx[j] = float64(lo + j)
	}
	return x.IndexSelect(1, tensor.New(idx, len(idx)))
}

// PadSequence pads a list of variable-length sequences, each of shape
// (T_i, F), to the longest length with padValue and batches them: with
// batchFirst the result is (B, Tmax, F), otherwise (Tmax, B, F) — the
// torch.nn.utils.rnn.pad_sequence equivalent.
//
// It is fully differentiable back to each input sequence (padding is
// Full + Concat along time, then Stack). GoNN's RNN/LSTM/GRU consume padded
// (B, T, F) batches directly; use UnpadSequence (or masking) to drop the
// padded steps, or PackedSequence/ForwardPacked for PyTorch-style
// length-aware outputs and final states.
func PadSequence(seqs []*tensor.Tensor, batchFirst bool, padValue float64) *tensor.Tensor {
	if len(seqs) == 0 {
		panic("PadSequence: need at least one sequence")
	}
	F := -1
	Tmax := 0
	for i, s := range seqs {
		if len(s.Shape) != 2 || s.Shape[0] < 1 {
			panic(fmt.Sprintf("PadSequence: seqs[%d] must be (T, F) with T >= 1, got shape %v", i, s.Shape))
		}
		if F == -1 {
			F = s.Shape[1]
		} else if s.Shape[1] != F {
			panic(fmt.Sprintf("PadSequence: seqs[%d] has F=%d, want %d", i, s.Shape[1], F))
		}
		if s.Shape[0] > Tmax {
			Tmax = s.Shape[0]
		}
	}
	padded := make([]*tensor.Tensor, len(seqs))
	for i, s := range seqs {
		if s.Shape[0] == Tmax {
			padded[i] = s
			continue
		}
		pad := tensor.Full(padValue, Tmax-s.Shape[0], F)
		padded[i] = tensor.Concat(0, s, pad)
	}
	if batchFirst {
		return tensor.Stack(0, padded...) // (B, Tmax, F)
	}
	return tensor.Stack(1, padded...) // (Tmax, B, F)
}

// UnpadSequence is the inverse of batch-first PadSequence: it slices a
// padded (B, Tmax, F) tensor back into B sequences of shape (lengths[b], F),
// dropping the padded steps. Differentiable via IndexSelect.
func UnpadSequence(padded *tensor.Tensor, lengths []int) []*tensor.Tensor {
	if len(padded.Shape) != 3 {
		panic("UnpadSequence: padded must be (B, Tmax, F) (batch-first)")
	}
	B, Tmax, F := padded.Shape[0], padded.Shape[1], padded.Shape[2]
	if len(lengths) != B {
		panic(fmt.Sprintf("UnpadSequence: %d lengths for batch of %d", len(lengths), B))
	}
	out := make([]*tensor.Tensor, B)
	for b := 0; b < B; b++ {
		L := lengths[b]
		if L < 1 || L > Tmax {
			panic(fmt.Sprintf("UnpadSequence: lengths[%d]=%d out of range [1,%d]", b, L, Tmax))
		}
		row := padded.IndexSelect(0, tensor.New([]float64{float64(b)}, 1)).Reshape(Tmax, F)
		idx := make([]float64, L)
		for t := range idx {
			idx[t] = float64(t)
		}
		out[b] = row.IndexSelect(0, tensor.New(idx, L)) // (L, F)
	}
	return out
}

// PackedSequence is GoNN's honest adaptation of
// torch.nn.utils.rnn.PackedSequence.
//
// DEVIATION (read this): unlike PyTorch, which stores a flattened
// time-major buffer and lets cuDNN skip computation on finished sequences,
// GoNN's PackedSequence stores the PADDED tensor plus per-sequence lengths.
// There are NO compute savings — only the SEMANTICS of packing are
// reproduced: RNN/LSTM/GRU ForwardPacked zeroes outputs at t >= length_i and
// returns final states taken at each sequence's true last step
// (t = length_i - 1), exactly as PyTorch does for packed input.
type PackedSequence struct {
	// Padded holds the padded batch: (B, Tmax, F) if BatchFirst, else
	// (Tmax, B, F).
	Padded *tensor.Tensor
	// Lengths[i] is the true (unpadded) length of sequence i, 1 <= L <= Tmax.
	Lengths []int
	// BatchFirst records the layout of Padded (and of ForwardPacked outputs).
	BatchFirst bool
}

// PackPaddedSequence wraps an already-padded batch and its lengths into a
// PackedSequence — the torch.nn.utils.rnn.pack_padded_sequence equivalent.
// Lengths need not be sorted (PyTorch enforce_sorted=False semantics); each
// must satisfy 1 <= lengths[i] <= Tmax. padded: (B, Tmax, F) with batchFirst,
// else (Tmax, B, F). No data is moved — see the PackedSequence deviation note.
func PackPaddedSequence(padded *tensor.Tensor, lengths []int, batchFirst bool) PackedSequence {
	if len(padded.Shape) != 3 {
		panic(fmt.Sprintf("PackPaddedSequence: padded must be 3-D, got shape %v", padded.Shape))
	}
	B, T := padded.Shape[0], padded.Shape[1]
	if !batchFirst {
		B, T = padded.Shape[1], padded.Shape[0]
	}
	if len(lengths) != B {
		panic(fmt.Sprintf("PackPaddedSequence: %d lengths for batch of %d", len(lengths), B))
	}
	for i, L := range lengths {
		if L < 1 || L > T {
			panic(fmt.Sprintf("PackPaddedSequence: lengths[%d]=%d out of range [1, %d]", i, L, T))
		}
	}
	ls := make([]int, len(lengths))
	copy(ls, lengths)
	return PackedSequence{Padded: padded, Lengths: ls, BatchFirst: batchFirst}
}

// PackSequence packs a list of variable-length (T_i, F) sequences — the
// torch.nn.utils.rnn.pack_sequence equivalent (enforce_sorted=False). It pads
// with zeros via PadSequence (batch-first) and records the lengths.
func PackSequence(seqs []*tensor.Tensor) PackedSequence {
	padded := PadSequence(seqs, true, 0)
	lengths := make([]int, len(seqs))
	for i, s := range seqs {
		lengths[i] = s.Shape[0]
	}
	return PackedSequence{Padded: padded, Lengths: lengths, BatchFirst: true}
}

// PadPackedSequence unwraps a PackedSequence back into its padded tensor and
// lengths — the torch.nn.utils.rnn.pad_packed_sequence equivalent. The tensor
// is returned in the layout recorded by ps.BatchFirst.
func PadPackedSequence(ps PackedSequence) (*tensor.Tensor, []int) {
	lengths := make([]int, len(ps.Lengths))
	copy(lengths, ps.Lengths)
	return ps.Padded, lengths
}

// batchFirstPadded returns the padded data as (B, Tmax, F), permuting
// (differentiably) when the stored layout is time-first.
func (ps PackedSequence) batchFirstPadded() *tensor.Tensor {
	if ps.BatchFirst {
		return ps.Padded
	}
	return ps.Padded.Permute(1, 0, 2)
}

// checkAgainst validates the packed batch against a consumer ("RNN.ForwardPacked").
func (ps PackedSequence) checkAgainst(kind string) {
	if ps.Padded == nil || len(ps.Padded.Shape) != 3 {
		panic(fmt.Sprintf("%s: packed data must be 3-D, got %v", kind, ps.Padded))
	}
	B, T := ps.Padded.Shape[0], ps.Padded.Shape[1]
	if !ps.BatchFirst {
		B, T = ps.Padded.Shape[1], ps.Padded.Shape[0]
	}
	if len(ps.Lengths) != B {
		panic(fmt.Sprintf("%s: %d lengths for batch of %d", kind, len(ps.Lengths), B))
	}
	for i, L := range ps.Lengths {
		if L < 1 || L > T {
			panic(fmt.Sprintf("%s: lengths[%d]=%d out of range [1, %d]", kind, i, L, T))
		}
	}
}

// packedOutputMask builds the constant (B, T, 1) 0/1 mask that zeroes padded
// timesteps: mask[b, t] = 1 iff t < lengths[b]. Multiplying by it is the
// differentiable "zero beyond length" step of ForwardPacked.
func packedOutputMask(B, T int, lengths []int) *tensor.Tensor {
	m := tensor.Zeros(B, T, 1)
	for b, L := range lengths {
		for t := 0; t < L; t++ {
			m.Data[b*T+t] = 1
		}
	}
	return m
}

// gatherLastSteps picks out[b, lengths[b]-1, :] for every batch element of a
// (B, T, H) tensor, returning (B, H). Differentiable via IndexSelect.
func gatherLastSteps(out *tensor.Tensor, lengths []int) *tensor.Tensor {
	B, T, H := out.Shape[0], out.Shape[1], out.Shape[2]
	rows := make([]*tensor.Tensor, B)
	for b := 0; b < B; b++ {
		row := out.IndexSelect(0, tensor.New([]float64{float64(b)}, 1)).Reshape(T, H)
		rows[b] = row.IndexSelect(0, tensor.New([]float64{float64(lengths[b] - 1)}, 1)).Reshape(H)
	}
	return tensor.Stack(0, rows...) // (B, H)
}
