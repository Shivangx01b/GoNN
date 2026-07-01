package nn

import (
	"fmt"
	"math"
	"math/rand"

	"gonn/tensor"
)

// Channel-wise dropout (torch.nn.Dropout1d/2d/3d): zero ENTIRE channels of a
// (N, C, spatial...) tensor with probability p during training, scaling the
// surviving channels by 1/(1-p) so expectations are preserved. Identity in
// eval mode. The mask has shape (N, C, 1, ...) and is applied with a
// broadcast multiply, so gradients flow through the kept elements only.
//
// Deviation from PyTorch: unbatched inputs ((C, L), (C, H, W), ...) are not
// accepted — add a leading batch dim of 1 — and Dropout2d does not replicate
// PyTorch's deprecated "3D input treated as (N, C, L)" behavior; use
// Dropout1d for (N, C, L).

// channelMask returns a (N, C, 1, ...) 0/1 keep-mask (1 with probability
// keep) matching the rank of shape, for broadcasting over the spatial dims.
func channelMask(shape []int, keep float64) *tensor.Tensor {
	maskShape := make([]int, len(shape))
	for i := range maskShape {
		maskShape[i] = 1
	}
	maskShape[0], maskShape[1] = shape[0], shape[1]
	mask := tensor.Zeros(maskShape...)
	for i := range mask.Data {
		if rand.Float64() < keep {
			mask.Data[i] = 1
		}
	}
	return mask
}

// channelDropout applies channel-wise dropout with drop probability p.
func channelDropout(x *tensor.Tensor, p float64) *tensor.Tensor {
	if p <= 0 {
		return x
	}
	if p >= 1 {
		return x.MulScalar(0)
	}
	keep := 1.0 - p
	mask := channelMask(x.Shape, keep).MulScalar(1.0 / keep)
	return x.Mul(mask)
}

func checkChannelRank(name string, x *tensor.Tensor, rank int) {
	if len(x.Shape) != rank {
		panic(fmt.Sprintf("%s: expected %dD input (N, C, spatial...), got shape %v", name, rank, x.Shape))
	}
}

// Dropout1d zeroes entire channels of a (N, C, L) input with probability P
// during training (each channel is a 1D feature map). Survivors are scaled
// by 1/(1-P). Identity in eval mode.
type Dropout1d struct {
	Base
	P float64
}

// NewDropout1d returns a Dropout1d module (training mode by default).
func NewDropout1d(p float64) *Dropout1d { return &Dropout1d{P: p} }

// Forward applies channel dropout to a 3D input in training mode.
func (d *Dropout1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkChannelRank("Dropout1d", x, 3)
	if !d.Training() {
		return x
	}
	return channelDropout(x, d.P)
}

// Dropout2d zeroes entire channels of a (N, C, H, W) input with probability
// P during training (each channel is a 2D feature map). Survivors are scaled
// by 1/(1-P). Identity in eval mode.
type Dropout2d struct {
	Base
	P float64
}

// NewDropout2d returns a Dropout2d module (training mode by default).
func NewDropout2d(p float64) *Dropout2d { return &Dropout2d{P: p} }

// Forward applies channel dropout to a 4D input in training mode.
func (d *Dropout2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkChannelRank("Dropout2d", x, 4)
	if !d.Training() {
		return x
	}
	return channelDropout(x, d.P)
}

// Dropout3d zeroes entire channels of a (N, C, D, H, W) input with
// probability P during training (each channel is a 3D feature map).
// Survivors are scaled by 1/(1-P). Identity in eval mode.
type Dropout3d struct {
	Base
	P float64
}

// NewDropout3d returns a Dropout3d module (training mode by default).
func NewDropout3d(p float64) *Dropout3d { return &Dropout3d{P: p} }

// Forward applies channel dropout to a 5D input in training mode.
func (d *Dropout3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkChannelRank("Dropout3d", x, 5)
	if !d.Training() {
		return x
	}
	return channelDropout(x, d.P)
}

// alphaPrime is the negative saturation value of SELU:
// -selu_lambda * selu_alpha = -1.0507009873554805 * 1.6732632423543772.
// PyTorch hardcodes the same constant in ATen's alpha_dropout.
const alphaPrime = -1.7580993408473766

// alphaDropoutTransform applies the SELU-compatible affine-corrected dropout
// given a 0/1 keep mask (elementwise or broadcastable channel mask):
//
//	out = a * (x*mask + alpha' * (1-mask)) + b
//	a   = (q + alpha'^2 * q * (1-q))^(-1/2),  q = 1-p
//	b   = -a * alpha' * (1-q)
//
// which keeps zero mean and unit variance for zero-mean unit-variance inputs
// (Klambauer et al., "Self-Normalizing Neural Networks"; identical to
// PyTorch's ATen formula a = 1/sqrt((alpha^2*p + 1)*(1-p))).
func alphaDropoutTransform(x, mask *tensor.Tensor, p float64) *tensor.Tensor {
	q := 1.0 - p
	a := math.Pow(q+alphaPrime*alphaPrime*q*(1-q), -0.5)
	b := -a * alphaPrime * (1 - q)
	// alpha' * (1 - mask), built from the constant mask (no grad needed).
	fill := mask.MulScalar(-alphaPrime).AddScalar(alphaPrime)
	return x.Mul(mask).Add(fill).MulScalar(a).AddScalar(b)
}

// bernoulliMask returns a 0/1 tensor of the given shape, 1 with probability
// keep.
func bernoulliMask(keep float64, shape ...int) *tensor.Tensor {
	m := tensor.Zeros(shape...)
	for i := range m.Data {
		if rand.Float64() < keep {
			m.Data[i] = 1
		}
	}
	return m
}

// AlphaDropout randomly masks elements with probability P during training,
// setting them to the negative SELU saturation value alpha' and applying an
// affine correction so that zero-mean/unit-variance inputs keep zero mean and
// unit variance (torch.nn.AlphaDropout; pair with SELU activations).
// Identity in eval mode. Like PyTorch, P >= 1 yields all zeros.
type AlphaDropout struct {
	Base
	P float64
}

// NewAlphaDropout returns an AlphaDropout module (training mode by default).
func NewAlphaDropout(p float64) *AlphaDropout { return &AlphaDropout{P: p} }

// Forward applies alpha dropout in training mode; identity otherwise.
func (d *AlphaDropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !d.Training() || d.P <= 0 {
		return x
	}
	if d.P >= 1 {
		return x.MulScalar(0)
	}
	return alphaDropoutTransform(x, bernoulliMask(1.0-d.P, x.Shape...), d.P)
}

// FeatureAlphaDropout is the channel-wise variant of AlphaDropout
// (torch.nn.FeatureAlphaDropout): entire channels of a (N, C, spatial...)
// input are masked to the negative SELU saturation value with probability P
// during training, with the same mean/variance-preserving affine correction.
// Identity in eval mode. Requires rank >= 2 (batched input; unbatched inputs
// are not accepted, unlike PyTorch).
type FeatureAlphaDropout struct {
	Base
	P float64
}

// NewFeatureAlphaDropout returns a FeatureAlphaDropout module (training mode
// by default).
func NewFeatureAlphaDropout(p float64) *FeatureAlphaDropout { return &FeatureAlphaDropout{P: p} }

// Forward applies channel-wise alpha dropout in training mode.
func (d *FeatureAlphaDropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) < 2 {
		panic(fmt.Sprintf("FeatureAlphaDropout: expected input of rank >= 2 (N, C, ...), got shape %v", x.Shape))
	}
	if !d.Training() || d.P <= 0 {
		return x
	}
	if d.P >= 1 {
		return x.MulScalar(0)
	}
	return alphaDropoutTransform(x, channelMask(x.Shape, 1.0-d.P), d.P)
}
