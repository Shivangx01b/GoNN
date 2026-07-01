package nn

import (
	"testing"

	"gonn/tensor"
)

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestConv1dShape(t *testing.T) {
	c := NewConv1d(3, 8, 3, WithPad(1))
	x := tensor.Randn(2, 3, 10)
	y := c.Forward(x)
	// outL = (10 + 2*1 - 3)/1 + 1 = 10
	if !shapeEq(y.Shape, []int{2, 8, 10}) {
		t.Fatalf("Conv1d shape: got %v want [2 8 10]", y.Shape)
	}
}

func TestConv1dStride2Shape(t *testing.T) {
	c := NewConv1d(2, 4, 3, WithStride(2), WithNoBias())
	x := tensor.Randn(1, 2, 9)
	y := c.Forward(x)
	// outL = (9 - 3)/2 + 1 = 4
	if !shapeEq(y.Shape, []int{1, 4, 4}) {
		t.Fatalf("Conv1d stride2 shape: got %v", y.Shape)
	}
}

func TestConv3dShape(t *testing.T) {
	c := NewConv3d(2, 4, 3, WithPad(1))
	x := tensor.Randn(1, 2, 5, 6, 7)
	y := c.Forward(x)
	// outs: (5+2-3)+1=5, (6+2-3)+1=6, (7+2-3)+1=7
	if !shapeEq(y.Shape, []int{1, 4, 5, 6, 7}) {
		t.Fatalf("Conv3d shape: got %v want [1 4 5 6 7]", y.Shape)
	}
}

func TestConvTranspose1dShape(t *testing.T) {
	c := NewConvTranspose1d(3, 5, 3, WithStride(2), WithPad(1))
	x := tensor.Randn(2, 3, 4)
	y := c.Forward(x)
	// outL = (4-1)*2 - 2*1 + 3 = 7
	if !shapeEq(y.Shape, []int{2, 5, 7}) {
		t.Fatalf("ConvTranspose1d shape: got %v want [2 5 7]", y.Shape)
	}
}

func TestConvTranspose2dShape(t *testing.T) {
	c := NewConvTranspose2d(2, 4, 3, WithStride(2), WithPad(1))
	x := tensor.Randn(1, 2, 4, 4)
	y := c.Forward(x)
	// outH = outW = (4-1)*2 - 2 + 3 = 7
	if !shapeEq(y.Shape, []int{1, 4, 7, 7}) {
		t.Fatalf("ConvTranspose2d shape: got %v want [1 4 7 7]", y.Shape)
	}
}

func TestConvTranspose3dShape(t *testing.T) {
	c := NewConvTranspose3d(2, 3, 3, WithStride(2), WithPad(1), WithNoBias())
	x := tensor.Randn(1, 2, 3, 3, 3)
	y := c.Forward(x)
	// each spatial: (3-1)*2 - 2 + 3 = 5
	if !shapeEq(y.Shape, []int{1, 3, 5, 5, 5}) {
		t.Fatalf("ConvTranspose3d shape: got %v want [1 3 5 5 5]", y.Shape)
	}
}

func TestAdaptiveAvgPool1d(t *testing.T) {
	p := NewAdaptiveAvgPool1d(4)
	x := tensor.Randn(2, 3, 10)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{2, 3, 4}) {
		t.Fatalf("AdaptiveAvgPool1d shape: got %v", y.Shape)
	}
}

func TestAdaptiveMaxPool1d(t *testing.T) {
	p := NewAdaptiveMaxPool1d(3)
	x := tensor.Randn(1, 2, 7)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 2, 3}) {
		t.Fatalf("AdaptiveMaxPool1d shape: got %v", y.Shape)
	}
}

func TestAdaptiveAvgPool2d(t *testing.T) {
	p := NewAdaptiveAvgPool2d(2, 3)
	x := tensor.Randn(1, 2, 5, 7)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 2, 2, 3}) {
		t.Fatalf("AdaptiveAvgPool2d shape: got %v", y.Shape)
	}
}

func TestAdaptiveMaxPool2d(t *testing.T) {
	p := NewAdaptiveMaxPool2d(1, 1)
	x := tensor.Randn(2, 4, 6, 6)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{2, 4, 1, 1}) {
		t.Fatalf("AdaptiveMaxPool2d shape: got %v", y.Shape)
	}
}

func TestAdaptiveAvgPool3d(t *testing.T) {
	p := NewAdaptiveAvgPool3d(2, 2, 2)
	x := tensor.Randn(1, 2, 4, 5, 6)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 2, 2, 2, 2}) {
		t.Fatalf("AdaptiveAvgPool3d shape: got %v", y.Shape)
	}
}

func TestAdaptiveMaxPool3d(t *testing.T) {
	p := NewAdaptiveMaxPool3d(1, 2, 2)
	x := tensor.Randn(1, 2, 3, 4, 5)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 2, 1, 2, 2}) {
		t.Fatalf("AdaptiveMaxPool3d shape: got %v", y.Shape)
	}
}

func TestZeroPad2d(t *testing.T) {
	p := NewZeroPad2d(1, 2, 3, 4)
	x := tensor.Randn(2, 3, 5, 6)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{2, 3, 5 + 1 + 2, 6 + 3 + 4}) {
		t.Fatalf("ZeroPad2d shape: got %v", y.Shape)
	}
	// Corner cells should be zero.
	if y.Data[0] != 0 {
		t.Fatalf("ZeroPad2d top-left not zero, got %v", y.Data[0])
	}
}

func TestConstantPad2d(t *testing.T) {
	p := NewConstantPad2d(1, 1, 1, 1, 3.5)
	x := tensor.Ones(1, 1, 2, 2)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 1, 4, 4}) {
		t.Fatalf("ConstantPad2d shape: got %v", y.Shape)
	}
	// Corner should be 3.5.
	if y.Data[0] != 3.5 {
		t.Fatalf("ConstantPad2d corner: got %v want 3.5", y.Data[0])
	}
	// Center should be 1.
	if y.Data[1*4+1] != 1 {
		t.Fatalf("ConstantPad2d center: got %v want 1", y.Data[1*4+1])
	}
}

func TestReflectionPad2d(t *testing.T) {
	p := NewReflectionPad2d(1, 1, 1, 1)
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 1, 4, 4}) {
		t.Fatalf("ReflectionPad2d shape: got %v", y.Shape)
	}
}

func TestReplicationPad2d(t *testing.T) {
	p := NewReplicationPad2d(2, 2, 2, 2)
	x := tensor.Randn(1, 1, 3, 3)
	y := p.Forward(x)
	if !shapeEq(y.Shape, []int{1, 1, 7, 7}) {
		t.Fatalf("ReplicationPad2d shape: got %v", y.Shape)
	}
}

func TestUpsampleNearest(t *testing.T) {
	u := NewUpsample(2, "nearest")
	x := tensor.New([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	y := u.Forward(x)
	if !shapeEq(y.Shape, []int{1, 1, 4, 4}) {
		t.Fatalf("Upsample nearest shape: got %v", y.Shape)
	}
	// Top-left 2x2 should all be 1.
	if y.Data[0] != 1 || y.Data[1] != 1 || y.Data[4] != 1 || y.Data[5] != 1 {
		t.Fatalf("Upsample nearest top-left tile mismatch: %v", y.Data[:8])
	}
}

func TestUpsampleBilinear(t *testing.T) {
	u := NewUpsample(2, "bilinear")
	x := tensor.Randn(1, 2, 3, 3)
	y := u.Forward(x)
	if !shapeEq(y.Shape, []int{1, 2, 6, 6}) {
		t.Fatalf("Upsample bilinear shape: got %v", y.Shape)
	}
}

func TestPixelShuffle(t *testing.T) {
	ps := NewPixelShuffle(2)
	x := tensor.Randn(1, 4*2*2, 3, 3)
	y := ps.Forward(x)
	if !shapeEq(y.Shape, []int{1, 4, 6, 6}) {
		t.Fatalf("PixelShuffle shape: got %v want [1 4 6 6]", y.Shape)
	}
}

func TestPixelUnshuffle(t *testing.T) {
	pu := NewPixelUnshuffle(2)
	x := tensor.Randn(1, 4, 6, 6)
	y := pu.Forward(x)
	if !shapeEq(y.Shape, []int{1, 16, 3, 3}) {
		t.Fatalf("PixelUnshuffle shape: got %v want [1 16 3 3]", y.Shape)
	}
}

func TestInstanceNorm1d(t *testing.T) {
	in := NewInstanceNorm1d(4, WithAffine(true))
	x := tensor.Randn(2, 4, 7)
	y := in.Forward(x)
	if !shapeEq(y.Shape, []int{2, 4, 7}) {
		t.Fatalf("InstanceNorm1d shape: got %v", y.Shape)
	}
}

func TestInstanceNorm2d(t *testing.T) {
	in := NewInstanceNorm2d(3)
	x := tensor.Randn(2, 3, 5, 6)
	y := in.Forward(x)
	if !shapeEq(y.Shape, []int{2, 3, 5, 6}) {
		t.Fatalf("InstanceNorm2d shape: got %v", y.Shape)
	}
}
