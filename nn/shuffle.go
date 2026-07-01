package nn

import (
	"fmt"

	"gonn/tensor"
)

// ChannelShuffle divides the channels of a (N, C, spatial...) tensor into
// Groups groups and interleaves them (torch.nn.ChannelShuffle): output
// channel i*g + j takes input channel j*(C/g) + i, for i in [0, C/g) and
// j in [0, g). Implemented as a pure reshape -> permute -> reshape, so it is
// differentiable by construction:
//
//	(N, C, ...) -> (N, g, C/g, ...) -> (N, C/g, g, ...) -> (N, C, ...)
type ChannelShuffle struct {
	Base
	Groups int
}

// NewChannelShuffle constructs a ChannelShuffle with the given group count.
func NewChannelShuffle(groups int) *ChannelShuffle {
	if groups <= 0 {
		panic("ChannelShuffle: groups must be positive")
	}
	return &ChannelShuffle{Groups: groups}
}

// Forward shuffles the channel axis of a rank >= 3 input.
func (c *ChannelShuffle) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) < 3 {
		panic(fmt.Sprintf("ChannelShuffle: expected input of rank >= 3 (N, C, spatial...), got shape %v", x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	g := c.Groups
	if C%g != 0 {
		panic(fmt.Sprintf("ChannelShuffle: channel count %d not divisible by groups %d", C, g))
	}
	rest := 1
	for i := 2; i < len(x.Shape); i++ {
		rest *= x.Shape[i]
	}
	y := x.Reshape(N, g, C/g, rest).Permute(0, 2, 1, 3)
	return y.Reshape(x.Shape...)
}
