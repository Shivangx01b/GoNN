package nn

import (
	"fmt"

	"gonn/tensor"
)

// BatchNorm3d normalizes a 5D input (N, C, D, H, W) per channel over the
// (N, D, H, W) axes, matching torch.nn.BatchNorm3d: batch statistics in
// training mode, running statistics (updated with momentum, PyTorch default
// 0.1, unbiased variance) in eval mode. Defaults: eps 1e-5, affine true.
//
// It is a thin wrapper over the shared batchNormNd core, which lowers to
// MatMul-free reshape/permute + reduction ops, so autograd works by
// construction.
type BatchNorm3d struct{ batchNormNd }

// NewBatchNorm3d constructs a BatchNorm3d with C features.
func NewBatchNorm3d(c int, opts ...NormOpt) *BatchNorm3d {
	return &BatchNorm3d{newBatchNorm(c, opts)}
}

// Forward applies BN over (N, D, H, W) per channel.
func (b *BatchNorm3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic(fmt.Sprintf("BatchNorm3d: expected 5D input, got shape %v", x.Shape))
	}
	return b.forwardChannelsFirst(x)
}

// InstanceNorm3d normalizes (N, C, D, H, W) per (sample, channel) over the
// spatial axes (D, H, W), matching torch.nn.InstanceNorm3d with its defaults:
// affine off (enable with WithAffine(true)), no running statistics, eps 1e-5.
//
// Deviation from PyTorch: unbatched 4D input (C, D, H, W) is not accepted;
// add a leading batch dim of 1 instead (consistent with InstanceNorm1d/2d in
// this package).
type InstanceNorm3d struct{ instanceNormNd }

// NewInstanceNorm3d constructs an InstanceNorm3d.
func NewInstanceNorm3d(c int, opts ...NormOpt) *InstanceNorm3d {
	return &InstanceNorm3d{newInstanceNorm(5, c, opts)}
}
