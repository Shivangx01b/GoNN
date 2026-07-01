package nn

import (
	"math"
	"math/rand"
	"testing"

	"gonn/tensor"
)

func TestDropout2dZeroesWholeChannels(t *testing.T) {
	rand.Seed(1234)
	d := NewDropout2d(0.5)
	x := tensor.Ones(4, 8, 3, 3)
	y := d.Forward(x)
	if !shapeEq(y.Shape, x.Shape) {
		t.Fatalf("Dropout2d shape: got %v", y.Shape)
	}
	const scale = 2.0 // 1/(1-p)
	zeroCh, keptCh := 0, 0
	for n := 0; n < 4; n++ {
		for c := 0; c < 8; c++ {
			base := (n*8 + c) * 9
			first := y.Data[base]
			if first != 0 && math.Abs(first-scale) > 1e-12 {
				t.Fatalf("channel (%d,%d): value %g, want 0 or %g", n, c, first, scale)
			}
			for s := 0; s < 9; s++ {
				if y.Data[base+s] != first {
					t.Fatalf("channel (%d,%d) not uniform: %g vs %g", n, c, y.Data[base+s], first)
				}
			}
			if first == 0 {
				zeroCh++
			} else {
				keptCh++
			}
		}
	}
	if zeroCh == 0 || keptCh == 0 {
		t.Fatalf("expected a mix of zeroed and kept channels, got %d/%d", zeroCh, keptCh)
	}
}

func TestChannelDropoutEvalIdentityAndEdgeCases(t *testing.T) {
	x3 := seededRandn(120, 2, 3, 4)
	x4 := seededRandn(121, 2, 3, 4, 2)
	x5 := seededRandn(122, 2, 3, 2, 2, 2)
	mods := []struct {
		name string
		m    Module
		x    *tensor.Tensor
	}{
		{"Dropout1d", NewDropout1d(0.5), x3},
		{"Dropout2d", NewDropout2d(0.5), x4},
		{"Dropout3d", NewDropout3d(0.5), x5},
		{"AlphaDropout", NewAlphaDropout(0.5), x4},
		{"FeatureAlphaDropout", NewFeatureAlphaDropout(0.5), x4},
	}
	for _, tc := range mods {
		tc.m.SetTraining(false)
		y := tc.m.Forward(tc.x)
		if !dataClose(y.Data, tc.x.Data, 0) {
			t.Errorf("%s: eval mode must be identity", tc.name)
		}
	}
	// p=0 is identity even in training; p=1 zeroes everything (PyTorch).
	if y := NewDropout3d(0).Forward(x5); !dataClose(y.Data, x5.Data, 0) {
		t.Errorf("Dropout3d(p=0) must be identity in training mode")
	}
	if y := NewAlphaDropout(1).Forward(x4); !dataClose(y.Data, make([]float64, len(x4.Data)), 0) {
		t.Errorf("AlphaDropout(p=1) must produce zeros")
	}
}

func TestDropout1dStatistics(t *testing.T) {
	rand.Seed(99)
	const p = 0.3
	d := NewDropout1d(p)
	x := tensor.Ones(1, 10, 4)
	trials := 2000
	zeroed, total := 0, 0
	sum := 0.0
	for i := 0; i < trials; i++ {
		y := d.Forward(x)
		for c := 0; c < 10; c++ {
			if y.Data[c*4] == 0 {
				zeroed++
			}
			total++
		}
		for _, v := range y.Data {
			sum += v
		}
	}
	frac := float64(zeroed) / float64(total)
	if math.Abs(frac-p) > 0.02 {
		t.Errorf("zeroed-channel fraction %g, want ~%g", frac, p)
	}
	mean := sum / float64(trials*40)
	if math.Abs(mean-1) > 0.05 {
		t.Errorf("survivor scaling: mean %g, want ~1", mean)
	}
}

func TestAlphaDropoutKeepsMeanAndVariance(t *testing.T) {
	rand.Seed(4242)
	d := NewAlphaDropout(0.5)
	// N(0,1) input: after alpha dropout, mean ~0 and variance ~1.
	x := seededRandn(123, 100, 10, 20) // 20000 elements
	y := d.Forward(x)
	mean, sq := 0.0, 0.0
	for _, v := range y.Data {
		mean += v
		sq += v * v
	}
	n := float64(len(y.Data))
	mean /= n
	variance := sq/n - mean*mean
	if math.Abs(mean) > 0.03 {
		t.Errorf("AlphaDropout mean %g, want ~0", mean)
	}
	if math.Abs(variance-1) > 0.06 {
		t.Errorf("AlphaDropout variance %g, want ~1", variance)
	}
}

func TestFeatureAlphaDropoutMasksWholeChannels(t *testing.T) {
	rand.Seed(31)
	const p = 0.5
	d := NewFeatureAlphaDropout(p)
	x := seededRandn(124, 3, 16, 5)
	y := d.Forward(x)
	q := 1 - p
	a := math.Pow(q+alphaPrime*alphaPrime*q*(1-q), -0.5)
	b := -a * alphaPrime * (1 - q)
	maskedVal := a*alphaPrime + b
	masked, kept := 0, 0
	for n := 0; n < 3; n++ {
		for c := 0; c < 16; c++ {
			base := (n*16 + c) * 5
			if math.Abs(y.Data[base]-maskedVal) < 1e-9 {
				masked++
				for s := 0; s < 5; s++ {
					if math.Abs(y.Data[base+s]-maskedVal) > 1e-12 {
						t.Fatalf("masked channel (%d,%d) not constant at %g", n, c, maskedVal)
					}
				}
			} else {
				kept++
				for s := 0; s < 5; s++ {
					want := a*x.Data[base+s] + b
					if math.Abs(y.Data[base+s]-want) > 1e-12 {
						t.Fatalf("kept channel (%d,%d)[%d]: got %g, want a*x+b=%g", n, c, s, y.Data[base+s], want)
					}
				}
			}
		}
	}
	if masked == 0 || kept == 0 {
		t.Fatalf("expected a mix of masked and kept channels, got %d/%d", masked, kept)
	}
}

// Gradchecks: the mask is re-drawn per forward, so the loss closures re-seed
// the global RNG to make every invocation produce the identical mask.

func TestGradCheckDropoutChannel(t *testing.T) {
	d1 := NewDropout1d(0.4)
	x1 := seededRandn(125, 2, 4, 3).SetRequiresGrad(true)
	gradCheck(t, "Dropout1d", func() *tensor.Tensor {
		rand.Seed(777)
		return d1.Forward(x1).Square().Mean()
	}, []*tensor.Tensor{x1}, gcEps, gcTol, 0)

	d2 := NewDropout2d(0.5)
	x2 := seededRandn(126, 2, 3, 2, 2).SetRequiresGrad(true)
	gradCheck(t, "Dropout2d", func() *tensor.Tensor {
		rand.Seed(778)
		return d2.Forward(x2).Square().Mean()
	}, []*tensor.Tensor{x2}, gcEps, gcTol, 0)

	d3 := NewDropout3d(0.5)
	x3 := seededRandn(127, 2, 3, 2, 2, 2).SetRequiresGrad(true)
	gradCheck(t, "Dropout3d", func() *tensor.Tensor {
		rand.Seed(779)
		return d3.Forward(x3).Square().Mean()
	}, []*tensor.Tensor{x3}, gcEps, gcTol, 0)
}

func TestGradCheckAlphaDropout(t *testing.T) {
	d := NewAlphaDropout(0.4)
	x := seededRandn(128, 3, 5).SetRequiresGrad(true)
	gradCheck(t, "AlphaDropout", func() *tensor.Tensor {
		rand.Seed(780)
		return d.Forward(x).Square().Mean()
	}, []*tensor.Tensor{x}, gcEps, gcTol, 0)

	f := NewFeatureAlphaDropout(0.4)
	xf := seededRandn(129, 2, 4, 3).SetRequiresGrad(true)
	gradCheck(t, "FeatureAlphaDropout", func() *tensor.Tensor {
		rand.Seed(781)
		return f.Forward(xf).Square().Mean()
	}, []*tensor.Tensor{xf}, gcEps, gcTol, 0)
}
