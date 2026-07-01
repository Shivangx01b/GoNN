package nn

import (
	"testing"

	"gonn/tensor"
)

func TestUnfoldShapes(t *testing.T) {
	// L = prod over dims of (in + 2p - d*(k-1) - 1)/s + 1.
	u := NewUnfold(2, WithKernel(2, 3))
	y := u.Forward(seededRandn(61, 2, 3, 4, 5))
	// L = (4-2+1)*(5-3+1) = 3*3 = 9; C*prod(kernel) = 3*6 = 18.
	if !intsEqual(y.Shape, []int{2, 18, 9}) {
		t.Fatalf("Unfold shape %v, want [2 18 9]", y.Shape)
	}

	u2 := NewUnfold(2, WithStride(2), WithPad(1), WithDilation(1))
	y2 := u2.Forward(seededRandn(62, 1, 2, 4, 4))
	// per dim: (4+2-1-1)/2+1 = 3 -> L = 9; C*prod(kernel) = 2*4 = 8.
	if !intsEqual(y2.Shape, []int{1, 8, 9}) {
		t.Fatalf("Unfold shape %v, want [1 8 9]", y2.Shape)
	}
}

// Hand check against PyTorch's layout: output[n][c*KH*KW + k][l] is kernel
// offset k of block l (blocks row-major over output positions), rows ordered
// (C, kh, kw).
func TestUnfoldHandCheck(t *testing.T) {
	xData := make([]float64, 18)
	for i := 0; i < 9; i++ {
		xData[i] = float64(i)        // channel 0: 0..8
		xData[9+i] = float64(i) + 10 // channel 1: 10..18
	}
	x := tensor.New(xData, 1, 2, 3, 3)
	y := NewUnfold(2).Forward(x)
	if !intsEqual(y.Shape, []int{1, 8, 4}) {
		t.Fatalf("shape %v, want [1 8 4]", y.Shape)
	}
	want := []float64{
		// channel 0, kernel offsets (0,0),(0,1),(1,0),(1,1) x blocks (0,0),(0,1),(1,0),(1,1)
		0, 1, 3, 4,
		1, 2, 4, 5,
		3, 4, 6, 7,
		4, 5, 7, 8,
		// channel 1 (same pattern + 10)
		10, 11, 13, 14,
		11, 12, 14, 15,
		13, 14, 16, 17,
		14, 15, 17, 18,
	}
	requireSameTensor(t, "Unfold values", y, tensor.New(want, 1, 8, 4), 1e-12)
}

func TestFoldShape(t *testing.T) {
	f := NewFold([2]int{4, 5}, 2, WithKernel(2, 3))
	// L = (4-2+1)*(5-3+1) = 9, input dim1 = C*6.
	y := f.Forward(seededRandn(63, 2, 2*6, 9))
	if !intsEqual(y.Shape, []int{2, 2, 4, 5}) {
		t.Fatalf("Fold shape %v, want [2 2 4 5]", y.Shape)
	}
}

// PyTorch invariant: Fold(Unfold(x)) == x * divisor, where the divisor
// (per-position block-overlap count) is computed literally as
// Fold(Unfold(ones)).
func TestFoldUnfoldDivisorIdentity(t *testing.T) {
	geoms := []struct {
		name string
		opts []ConvOpt
	}{
		{"k2s1", nil},
		{"k2s2", []ConvOpt{WithStride(2)}},
		{"k3s1p1", []ConvOpt{WithKernel(3), WithStride(1), WithPad(1)}},
		{"k2s1d2", []ConvOpt{WithDilation(2)}},
	}
	for _, g := range geoms {
		kernel := 2
		u := NewUnfold(kernel, g.opts...)
		f := NewFold([2]int{4, 4}, kernel, g.opts...)
		x := seededRandn(64, 2, 3, 4, 4)
		ones := tensor.Ones(2, 3, 4, 4)
		divisor := f.Forward(u.Forward(ones))
		got := f.Forward(u.Forward(x))
		requireSameTensor(t, "Fold(Unfold(x)) == x*divisor ["+g.name+"]",
			got, x.Mul(divisor), 1e-10)
	}
}

func TestGradCheckUnfold(t *testing.T) {
	u := NewUnfold(2, WithStride(1), WithPad(1))
	x := seededRandn(65, 2, 2, 3, 3).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return u.Forward(x).Square().Mean() }
	gradCheck(t, "Unfold", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckFold(t *testing.T) {
	f := NewFold([2]int{3, 3}, 2)
	// L = 2*2 = 4, dim1 = C*4 = 8.
	x := seededRandn(66, 2, 8, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return f.Forward(x).Square().Mean() }
	gradCheck(t, "Fold", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestFoldValidation(t *testing.T) {
	expectPanic(t, "Fold wrong L", func() {
		NewFold([2]int{3, 3}, 2).Forward(tensor.Zeros(1, 4, 5))
	})
	expectPanic(t, "Fold dim1 not divisible", func() {
		NewFold([2]int{3, 3}, 2).Forward(tensor.Zeros(1, 5, 4))
	})
	expectPanic(t, "Unfold non-4D", func() {
		NewUnfold(2).Forward(tensor.Zeros(2, 3, 3))
	})
	expectPanic(t, "Unfold groups", func() { NewUnfold(2, WithGroups(2)) })
	expectPanic(t, "Fold output_padding", func() { NewFold([2]int{3, 3}, 2, WithOutputPadding(1)) })
}
