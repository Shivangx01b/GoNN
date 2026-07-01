package nn

import "gonn/tensor"

// PixelShuffle rearranges a (N, C*r^2, H, W) tensor into (N, C, H*r, W*r),
// where r is the UpscaleFactor.
type PixelShuffle struct {
	Base
	UpscaleFactor int
}

// NewPixelShuffle constructs a PixelShuffle with the given upscale factor.
func NewPixelShuffle(r int) *PixelShuffle {
	if r <= 0 {
		panic("PixelShuffle: upscale factor must be positive")
	}
	return &PixelShuffle{UpscaleFactor: r}
}

// Forward applies the pixel-shuffle reshape via reshape+permute+reshape.
func (p *PixelShuffle) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("PixelShuffle: expected 4D input (N,C*r^2,H,W)")
	}
	N, Cin, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	r := p.UpscaleFactor
	if Cin%(r*r) != 0 {
		panic("PixelShuffle: channel count not divisible by r*r")
	}
	Cout := Cin / (r * r)
	// (N, Cout, r, r, H, W)
	y := x.Reshape(N, Cout, r, r, H, W)
	// -> (N, Cout, H, r, W, r)
	y = y.Permute(0, 1, 4, 2, 5, 3)
	// -> (N, Cout, H*r, W*r)
	return y.Reshape(N, Cout, H*r, W*r)
}

// PixelUnshuffle is the inverse: (N, C, H*r, W*r) -> (N, C*r^2, H, W).
type PixelUnshuffle struct {
	Base
	DownscaleFactor int
}

// NewPixelUnshuffle constructs a PixelUnshuffle.
func NewPixelUnshuffle(r int) *PixelUnshuffle {
	if r <= 0 {
		panic("PixelUnshuffle: downscale factor must be positive")
	}
	return &PixelUnshuffle{DownscaleFactor: r}
}

// Forward applies the inverse pixel-shuffle rearrangement.
func (p *PixelUnshuffle) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("PixelUnshuffle: expected 4D input")
	}
	N, C, Hin, Win := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	r := p.DownscaleFactor
	if Hin%r != 0 || Win%r != 0 {
		panic("PixelUnshuffle: spatial dims not divisible by r")
	}
	H := Hin / r
	W := Win / r
	// (N, C, H, r, W, r)
	y := x.Reshape(N, C, H, r, W, r)
	// -> (N, C, r, r, H, W)
	y = y.Permute(0, 1, 3, 5, 2, 4)
	// -> (N, C*r*r, H, W)
	return y.Reshape(N, C*r*r, H, W)
}
