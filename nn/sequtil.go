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
// Full + Concat along time, then Stack). PackedSequence is intentionally
// not provided: GoNN's RNN/LSTM/GRU consume padded (B, T, F) batches
// directly; use UnpadSequence (or masking) to drop the padded steps.
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
