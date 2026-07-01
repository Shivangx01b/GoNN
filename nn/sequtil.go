package nn

import "gonn/tensor"

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
