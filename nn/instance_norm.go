package nn

import "gonn/tensor"

// InstanceNorm1d normalizes (N, C, L) per (sample, channel) over the L axis.
// Affine parameters Weight/Bias are per-channel (C,). If Affine is false,
// they are not used.
type InstanceNorm1d struct {
	NumFeatures int
	Eps         float64
	Affine      bool
	Weight      *tensor.Tensor // (C,) or nil
	Bias        *tensor.Tensor // (C,) or nil
}

// NewInstanceNorm1d constructs an InstanceNorm1d.
func NewInstanceNorm1d(c int, affine bool) *InstanceNorm1d {
	in := &InstanceNorm1d{NumFeatures: c, Eps: 1e-5, Affine: affine}
	if affine {
		in.Weight = tensor.Ones(c).SetRequiresGrad(true)
		in.Bias = tensor.Zeros(c).SetRequiresGrad(true)
	}
	return in
}

// Forward applies instance normalization on a (N, C, L) input.
func (in *InstanceNorm1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 3 {
		panic("InstanceNorm1d: expected 3D input (N,C,L)")
	}
	N, C, L := x.Shape[0], x.Shape[1], x.Shape[2]
	if C != in.NumFeatures {
		panic("InstanceNorm1d: channel mismatch")
	}
	mean := x.MeanAxis(2, true) // (N, C, 1)
	xc := x.Sub(mean)
	v := xc.Square().MeanAxis(2, true)
	std := v.AddScalar(in.Eps).Sqrt()
	norm := xc.Div(std)
	if in.Affine {
		w := in.Weight.Reshape(1, C, 1)
		b := in.Bias.Reshape(1, C, 1)
		norm = norm.Mul(w).Add(b)
	}
	_ = N
	_ = L
	return norm
}

// Parameters returns affine params (or nothing).
func (in *InstanceNorm1d) Parameters() []*tensor.Tensor {
	if in.Affine {
		return []*tensor.Tensor{in.Weight, in.Bias}
	}
	return nil
}

// InstanceNorm2d normalizes (N, C, H, W) per (sample, channel) over (H, W).
type InstanceNorm2d struct {
	NumFeatures int
	Eps         float64
	Affine      bool
	Weight      *tensor.Tensor // (C,) or nil
	Bias        *tensor.Tensor // (C,) or nil
}

// NewInstanceNorm2d constructs an InstanceNorm2d.
func NewInstanceNorm2d(c int, affine bool) *InstanceNorm2d {
	in := &InstanceNorm2d{NumFeatures: c, Eps: 1e-5, Affine: affine}
	if affine {
		in.Weight = tensor.Ones(c).SetRequiresGrad(true)
		in.Bias = tensor.Zeros(c).SetRequiresGrad(true)
	}
	return in
}

// Forward applies 2D instance normalization.
func (in *InstanceNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("InstanceNorm2d: expected 4D input (N,C,H,W)")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	if C != in.NumFeatures {
		panic("InstanceNorm2d: channel mismatch")
	}
	// Reduce over H then W with keepDim.
	mean := x.MeanAxis(3, true).MeanAxis(2, true) // (N, C, 1, 1)
	xc := x.Sub(mean)
	sq := xc.Square()
	v := sq.MeanAxis(3, true).MeanAxis(2, true) // (N, C, 1, 1)
	std := v.AddScalar(in.Eps).Sqrt()
	norm := xc.Div(std)
	if in.Affine {
		w := in.Weight.Reshape(1, C, 1, 1)
		b := in.Bias.Reshape(1, C, 1, 1)
		norm = norm.Mul(w).Add(b)
	}
	_ = N
	_ = H
	_ = W
	return norm
}

// Parameters returns affine params (or nothing).
func (in *InstanceNorm2d) Parameters() []*tensor.Tensor {
	if in.Affine {
		return []*tensor.Tensor{in.Weight, in.Bias}
	}
	return nil
}
