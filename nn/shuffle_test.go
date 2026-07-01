package nn

import (
	"testing"

	"gonn/tensor"
)

func TestChannelShufflePyTorchExample(t *testing.T) {
	// The torch.nn.ChannelShuffle doc example: (1, 4, 2, 2) arange input with
	// groups=2 reorders channels [0 1 2 3] -> [0 2 1 3].
	x := tensor.Arange(1, 17, 1).Reshape(1, 4, 2, 2)
	y := NewChannelShuffle(2).Forward(x)
	want := []float64{
		1, 2, 3, 4, // channel 0
		9, 10, 11, 12, // channel 2
		5, 6, 7, 8, // channel 1
		13, 14, 15, 16, // channel 3
	}
	if !shapeEq(y.Shape, []int{1, 4, 2, 2}) {
		t.Fatalf("ChannelShuffle shape: got %v", y.Shape)
	}
	if !dataClose(y.Data, want, 0) {
		t.Fatalf("ChannelShuffle: got %v, want %v", y.Data, want)
	}
}

func TestChannelShuffleRoundTrip(t *testing.T) {
	// Shuffling with g groups then C/g groups is the identity.
	x := seededRandn(130, 2, 6, 3, 2)
	y := NewChannelShuffle(3).Forward(NewChannelShuffle(2).Forward(x))
	if !dataClose(y.Data, x.Data, 0) {
		t.Fatalf("ChannelShuffle(3)(ChannelShuffle(2)(x)) != x")
	}
}

func TestChannelShuffleTrivialGroups(t *testing.T) {
	x := seededRandn(131, 2, 4, 3)
	for _, g := range []int{1, 4} {
		y := NewChannelShuffle(g).Forward(x)
		if !dataClose(y.Data, x.Data, 0) {
			t.Errorf("ChannelShuffle(groups=%d) must be identity", g)
		}
	}
}

func TestChannelShuffleRejectsIndivisibleChannels(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("ChannelShuffle: expected panic when C %% groups != 0")
		}
	}()
	NewChannelShuffle(3).Forward(tensor.Zeros(1, 4, 2, 2))
}

func TestGradCheckChannelShuffle(t *testing.T) {
	cs := NewChannelShuffle(3)
	x := seededRandn(132, 2, 6, 2).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return cs.Forward(x).Square().Mean() }
	gradCheck(t, "ChannelShuffle", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}
