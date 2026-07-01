package nn

import (
	"math"
	"testing"

	"gonn/optim"
	"gonn/tensor"
)

func TestLazyLinearInfersInFeaturesAndTrains(t *testing.T) {
	ll := NewLazyLinear(1, true)
	if !IsLazy(ll) {
		t.Fatal("IsLazy(LazyLinear) = false")
	}
	if ll.IsInitialized() {
		t.Fatal("LazyLinear initialized before first forward")
	}
	if n := len(ll.Parameters()); n != 0 {
		t.Fatalf("Parameters before init: %d, want 0 (run a dummy forward before the optimizer)", n)
	}
	if ll.Inner() != nil {
		t.Fatal("Inner() non-nil before init")
	}

	x := seededRandn(70, 16, 5)
	// Ground-truth target: a fixed linear map of x.
	wTrue := tensor.New([]float64{0.5, -1, 2, 0.25, -0.75}, 5, 1)
	target := x.MatMul(wTrue)

	y := ll.Forward(x)
	if y.Shape[0] != 16 || y.Shape[1] != 1 {
		t.Fatalf("output shape %v, want [16 1]", y.Shape)
	}
	if !ll.IsInitialized() || ll.Inner() == nil || ll.Inner().InFeatures != 5 {
		t.Fatalf("inner not materialized with InFeatures=5")
	}
	if n := len(ll.Parameters()); n != 2 {
		t.Fatalf("Parameters after init: %d, want 2", n)
	}

	opt := optim.NewSGD(ll.Parameters(), 0.05)
	loss0 := MSELoss(ll.Forward(x), target).Item()
	var last float64
	for i := 0; i < 100; i++ {
		opt.ZeroGrad()
		loss := MSELoss(ll.Forward(x), target)
		loss.Backward()
		opt.Step()
		last = loss.Item()
	}
	if !(last < loss0*0.1) {
		t.Fatalf("training did not reduce loss: start %v end %v", loss0, last)
	}
}

func TestGradCheckLazyLinearPostInit(t *testing.T) {
	ll := NewLazyLinear(3, true)
	x := seededRandn(71, 2, 4).SetRequiresGrad(true)
	ll.Forward(x) // materialize
	loss := func() *tensor.Tensor { return ll.Forward(x).Square().Mean() }
	gradCheck(t, "LazyLinear", loss, append(ll.Parameters(), x), gcEps, gcTol, 0)
}

func TestLazyInsideSequential(t *testing.T) {
	seq := NewSequential(NewLazyLinear(4, true), ReLU(), NewLazyLinear(2, true))
	if n := len(seq.Parameters()); n != 0 {
		t.Fatalf("Sequential of lazy modules has %d params before init, want 0", n)
	}

	x := seededRandn(72, 3, 6)
	y := seq.Forward(x) // container forward runs through nn.Call
	if y.Shape[0] != 3 || y.Shape[1] != 2 {
		t.Fatalf("output shape %v, want [3 2]", y.Shape)
	}
	if n := len(seq.Parameters()); n != 4 {
		t.Fatalf("Parameters after init: %d, want 4", n)
	}
	np := seq.NamedParameters()
	if np[0].Name != "0.inner.weight" {
		t.Fatalf("NamedParameters[0].Name = %q, want 0.inner.weight", np[0].Name)
	}

	// Direct Call on an initialized lazy module keeps working.
	y2 := Call(seq, x)
	if y2.Shape[0] != 3 || y2.Shape[1] != 2 {
		t.Fatalf("Call output shape %v, want [3 2]", y2.Shape)
	}
}

func TestLazyEvalModeBeforeInitPropagates(t *testing.T) {
	lb := NewLazyBatchNorm1d()
	lb.Eval()
	x := seededRandn(73, 6, 3)
	y := lb.Forward(x)
	inner := lb.InnerModule().(*BatchNorm1d)
	if inner.Training() {
		t.Fatal("inner module of an eval-mode lazy norm should be in eval mode")
	}
	// Eval-mode BN with fresh running stats (mean 0, var 1) is ~identity.
	for i := range x.Data {
		if math.Abs(y.Data[i]-x.Data[i]) > 1e-3 {
			t.Fatalf("eval BN output[%d] = %v, want ~%v", i, y.Data[i], x.Data[i])
		}
	}
}

func TestLazyConvVariantsInferChannelsAndShapes(t *testing.T) {
	cases := []struct {
		name    string
		m       Module
		in      []int
		wantOut []int
		inC     func(m Module) int
	}{
		{"LazyConv1d", NewLazyConv1d(4, 3, WithPad(1)), []int{2, 3, 8}, []int{2, 4, 8},
			func(m Module) int { return m.(*Conv1d).InC }},
		{"LazyConv2d", NewLazyConv2d(5, 3, WithStride(2), WithPad(1)), []int{2, 3, 8, 8}, []int{2, 5, 4, 4},
			func(m Module) int { return m.(*Conv2d).InC }},
		{"LazyConv3d", NewLazyConv3d(2, 2), []int{1, 3, 3, 4, 4}, []int{1, 2, 2, 3, 3},
			func(m Module) int { return m.(*Conv3d).InC }},
		{"LazyConvTranspose1d", NewLazyConvTranspose1d(3, 3, WithStride(2), WithPad(1)), []int{2, 2, 5}, []int{2, 3, 9},
			func(m Module) int { return m.(*ConvTranspose1d).InC }},
		{"LazyConvTranspose2d", NewLazyConvTranspose2d(3, 2), []int{1, 2, 3, 3}, []int{1, 3, 4, 4},
			func(m Module) int { return m.(*ConvTranspose2d).InC }},
		{"LazyConvTranspose3d", NewLazyConvTranspose3d(2, 2, WithStride(2)), []int{1, 2, 2, 2, 2}, []int{1, 2, 4, 4, 4},
			func(m Module) int { return m.(*ConvTranspose3d).InC }},
	}
	for _, tc := range cases {
		lz, ok := tc.m.(lazyInitializer)
		if !ok || lz.IsInitialized() {
			t.Fatalf("%s: not lazy or initialized too early", tc.name)
		}
		if n := len(tc.m.Parameters()); n != 0 {
			t.Fatalf("%s: %d params before init, want 0", tc.name, n)
		}
		x := seededRandn(74, tc.in...)
		y := tc.m.Forward(x)
		if !intsEqual(y.Shape, tc.wantOut) {
			t.Fatalf("%s: output shape %v, want %v", tc.name, y.Shape, tc.wantOut)
		}
		inner := tc.m.(interface{ InnerModule() Module }).InnerModule()
		if got, want := tc.inC(inner), tc.in[1]; got != want {
			t.Fatalf("%s: inferred InC = %d, want %d", tc.name, got, want)
		}
		if n := len(tc.m.Parameters()); n != 2 {
			t.Fatalf("%s: %d params after init, want 2 (weight+bias)", tc.name, n)
		}
	}
}

func TestLazyNormVariantsInferChannelsAndShapes(t *testing.T) {
	cases := []struct {
		name       string
		m          Module
		in         []int
		wantParams int
		wantBufs   int
	}{
		{"LazyBatchNorm1d-2D", NewLazyBatchNorm1d(), []int{4, 3}, 2, 2},
		{"LazyBatchNorm1d-3D", NewLazyBatchNorm1d(), []int{4, 3, 5}, 2, 2},
		{"LazyBatchNorm2d", NewLazyBatchNorm2d(), []int{2, 3, 4, 4}, 2, 2},
		{"LazyBatchNorm3d", NewLazyBatchNorm3d(), []int{2, 3, 2, 4, 4}, 2, 2},
		{"LazyInstanceNorm1d", NewLazyInstanceNorm1d(), []int{2, 3, 6}, 0, 0},
		{"LazyInstanceNorm2d-affine", NewLazyInstanceNorm2d(WithAffine(true)), []int{2, 3, 4, 4}, 2, 0},
		{"LazyInstanceNorm3d", NewLazyInstanceNorm3d(), []int{2, 2, 3, 4, 4}, 0, 0},
	}
	for _, tc := range cases {
		if !IsLazy(tc.m.(Child)) {
			t.Fatalf("%s: IsLazy = false", tc.name)
		}
		if n := len(tc.m.Parameters()); n != 0 {
			t.Fatalf("%s: %d params before init, want 0", tc.name, n)
		}
		x := seededRandn(75, tc.in...)
		y := tc.m.Forward(x)
		if !intsEqual(y.Shape, tc.in) {
			t.Fatalf("%s: output shape %v, want %v (norms preserve shape)", tc.name, y.Shape, tc.in)
		}
		if n := len(tc.m.Parameters()); n != tc.wantParams {
			t.Fatalf("%s: %d params after init, want %d", tc.name, n, tc.wantParams)
		}
		if n := len(tc.m.(Child).Buffers()); n != tc.wantBufs {
			t.Fatalf("%s: %d buffers after init, want %d", tc.name, n, tc.wantBufs)
		}
	}
}

func TestLazyBatchNorm3dNormalizesPerChannel(t *testing.T) {
	m := NewLazyBatchNorm3d()
	x := seededRandn(76, 2, 3, 2, 4, 4)
	y := m.Forward(x) // training mode: batch statistics
	// Per channel c, the output over (N, D, H, W) must have ~zero mean and
	// ~unit variance.
	N, C, rest := 2, 3, 2*4*4
	for c := 0; c < C; c++ {
		var sum, sq float64
		var cnt int
		for n := 0; n < N; n++ {
			base := (n*C + c) * rest
			for i := 0; i < rest; i++ {
				v := y.Data[base+i]
				sum += v
				sq += v * v
				cnt++
			}
		}
		mean := sum / float64(cnt)
		variance := sq/float64(cnt) - mean*mean
		if math.Abs(mean) > 1e-8 || math.Abs(variance-1) > 1e-3 {
			t.Fatalf("channel %d: mean %v var %v, want ~0 / ~1", c, mean, variance)
		}
	}
}
