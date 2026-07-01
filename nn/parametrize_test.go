package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

func maxAbsDiff64(t *testing.T, a, b []float64) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("length mismatch: %d vs %d", len(a), len(b))
	}
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

// ---- weight norm ------------------------------------------------------------

func TestWeightNormLinearInitPreservesFunction(t *testing.T) {
	l := NewLinear(5, 3, true)
	x := seededRandn(201, 4, 5)
	want := l.Forward(x)
	wn := NewWeightNormLinear(l)
	got := wn.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("wrapped output differs from original at init: max diff %g", d)
	}
}

func TestWeightNormConv2dInitPreservesFunction(t *testing.T) {
	c := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
	x := seededRandn(202, 2, 2, 6, 7)
	want := c.Forward(x)
	wn := NewWeightNormConv2d(c)
	got := wn.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("wrapped conv output differs from original at init: max diff %g", d)
	}
}

func TestGradCheckWeightNormLinear(t *testing.T) {
	wn := NewWeightNormLinear(NewLinear(4, 3, true))
	x := seededRandn(203, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return wn.Forward(x).Square().Mean() }
	gradCheck(t, "WeightNormLinear", loss, append(wn.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckWeightNormConv2d(t *testing.T) {
	wn := NewWeightNormConv2d(NewConv2d(2, 3, 3, WithPad(1)))
	x := seededRandn(204, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return wn.Forward(x).Square().Mean() }
	gradCheck(t, "WeightNormConv2d", loss, append(wn.Parameters(), x), gcEps, gcTol, 40)
}

func TestWeightNormTrainingUpdatesGVNotWrappedWeight(t *testing.T) {
	l := NewLinear(4, 3, true)
	origW := append([]float64(nil), l.Weight.Data...)
	wn := NewWeightNormLinear(l)
	g0 := append([]float64(nil), wn.G.Data...)
	v0 := append([]float64(nil), wn.V.Data...)

	x := seededRandn(205, 6, 4)
	for _, p := range wn.Parameters() {
		p.ZeroGrad()
	}
	wn.Forward(x).Square().Mean().Backward()
	for _, p := range wn.Parameters() {
		if p.Grad == nil {
			t.Fatalf("parameter received no gradient")
		}
		for i := range p.Data {
			p.Data[i] -= 0.1 * p.Grad.Data[i]
		}
	}

	if d := maxAbsDiff64(t, l.Weight.Data, origW); d != 0 {
		t.Errorf("wrapped layer's registered weight changed during training: max diff %g", d)
	}
	if maxAbsDiff64(t, wn.G.Data, g0) == 0 {
		t.Errorf("g did not change after a training step")
	}
	if maxAbsDiff64(t, wn.V.Data, v0) == 0 {
		t.Errorf("v did not change after a training step")
	}
}

func TestRemoveWeightNorm(t *testing.T) {
	wn := NewWeightNormLinear(NewLinear(5, 3, true))
	// Move g off its init value so the baked weight is a non-trivial g*v/||v||.
	for i := range wn.G.Data {
		wn.G.Data[i] *= 1.0 + 0.1*float64(i+1)
	}
	x := seededRandn(206, 4, 5)
	want := wn.Forward(x)
	plain := RemoveWeightNormLinear(wn)
	got := plain.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("removed linear differs from parametrized forward: max diff %g", d)
	}

	wnc := NewWeightNormConv2d(NewConv2d(2, 3, 3, WithStride(2), WithPad(1)))
	for i := range wnc.G.Data {
		wnc.G.Data[i] *= 0.7 + 0.2*float64(i)
	}
	xc := seededRandn(207, 2, 2, 6, 6)
	wantC := wnc.Forward(xc)
	plainC := RemoveWeightNormConv2d(wnc)
	gotC := plainC.Forward(xc)
	if d := maxAbsDiff64(t, gotC.Data, wantC.Data); d > 1e-12 {
		t.Errorf("removed conv differs from parametrized forward: max diff %g", d)
	}
}

// ---- spectral norm ----------------------------------------------------------

// refTopSingularValue estimates the top singular value of the m x n row-major
// matrix w with an independent long power iteration (test-side reference).
func refTopSingularValue(w []float64, m, n, iters int) float64 {
	u := make([]float64, m)
	v := make([]float64, n)
	for i := range u {
		u[i] = 1 / math.Sqrt(float64(m))
	}
	norml := func(x []float64) {
		s := 0.0
		for _, e := range x {
			s += e * e
		}
		inv := 1 / math.Sqrt(s)
		for i := range x {
			x[i] *= inv
		}
	}
	for it := 0; it < iters; it++ {
		for j := 0; j < n; j++ {
			s := 0.0
			for i := 0; i < m; i++ {
				s += w[i*n+j] * u[i]
			}
			v[j] = s
		}
		norml(v)
		for i := 0; i < m; i++ {
			s := 0.0
			for j := 0; j < n; j++ {
				s += w[i*n+j] * v[j]
			}
			u[i] = s
		}
		norml(u)
	}
	sigma := 0.0
	for i := 0; i < m; i++ {
		row := 0.0
		for j := 0; j < n; j++ {
			row += w[i*n+j] * v[j]
		}
		sigma += u[i] * row
	}
	return sigma
}

// setSpectralTestWeight overwrites w (m x n) with a small random matrix plus
// a strong diagonal, giving a well-separated top singular value so power
// iteration converges fast and deterministically.
func setSpectralTestWeight(w []float64, m, n int, seed int64, diag []float64) {
	noise := seededRandn(seed, m, n)
	for i := range w {
		w[i] = 0.01 * noise.Data[i]
	}
	for i, d := range diag {
		w[i*n+i] += d
	}
}

func TestSpectralNormLinearSigma(t *testing.T) {
	l := NewLinear(6, 4, true)
	setSpectralTestWeight(l.Weight.Data, 4, 6, 208, []float64{5, 2, 1, 0.5})
	sn := NewSpectralNormLinear(l, WithNPowerIterations(2))
	x := seededRandn(209, 3, 6)
	for i := 0; i < 20; i++ {
		sn.Forward(x) // training mode: refines u/v
	}
	sigma := sn.EstimatedSigma()
	ref := refTopSingularValue(l.Weight.Data, 4, 6, 2000)
	if math.Abs(sigma-ref)/ref > 1e-8 {
		t.Errorf("sigma estimate %g does not match reference top singular value %g", sigma, ref)
	}
}

func TestSpectralNormConv2dSigma(t *testing.T) {
	c := NewConv2d(2, 3, 3, WithPad(1))
	setSpectralTestWeight(c.Weight.Data, 3, 18, 210, []float64{4, 2, 1})
	sn := NewSpectralNormConv2d(c, WithNPowerIterations(2))
	x := seededRandn(211, 2, 2, 5, 5)
	for i := 0; i < 20; i++ {
		sn.Forward(x)
	}
	sigma := sn.EstimatedSigma()
	ref := refTopSingularValue(c.Weight.Data, 3, 18, 2000)
	if math.Abs(sigma-ref)/ref > 1e-8 {
		t.Errorf("conv sigma estimate %g does not match reference %g", sigma, ref)
	}
}

func TestSpectralNormForwardScalesWeight(t *testing.T) {
	l := NewLinear(5, 3, true)
	sn := NewSpectralNormLinear(l)
	sn.Eval() // freeze u/v so sigma is a fixed constant
	sigma := sn.EstimatedSigma()

	x := seededRandn(212, 4, 5)
	got := sn.Forward(x)

	// Expected: plain linear forward with weight w/sigma.
	scaled := NewLinear(5, 3, true)
	for i, w := range l.Weight.Data {
		scaled.Weight.Data[i] = w / sigma
	}
	copy(scaled.Bias.Data, l.Bias.Data)
	want := scaled.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("spectral norm forward != x @ (W/sigma)^T + b: max diff %g", d)
	}
}

func TestSpectralNormBuffersUpdateOnlyInTrain(t *testing.T) {
	sn := NewSpectralNormLinear(NewLinear(5, 4, true))
	// Perturb u away from the power-iteration fixed point so a training
	// forward must move the buffers.
	pert := seededRandn(213, 4)
	copy(sn.U.Data, pert.Data)
	l2NormalizeInPlace(sn.U.Data, sn.Eps)

	u0 := append([]float64(nil), sn.U.Data...)
	v0 := append([]float64(nil), sn.V.Data...)
	x := seededRandn(214, 2, 5)

	sn.Eval()
	sn.Forward(x)
	if maxAbsDiff64(t, sn.U.Data, u0) != 0 || maxAbsDiff64(t, sn.V.Data, v0) != 0 {
		t.Errorf("u/v buffers changed during an eval-mode forward")
	}

	sn.Train()
	sn.Forward(x)
	if maxAbsDiff64(t, sn.U.Data, u0) == 0 && maxAbsDiff64(t, sn.V.Data, v0) == 0 {
		t.Errorf("u/v buffers did not change during a training-mode forward")
	}
}

func TestGradCheckSpectralNormLinear(t *testing.T) {
	sn := NewSpectralNormLinear(NewLinear(4, 3, true))
	sn.Eval() // buffers must stay fixed for a deterministic loss
	x := seededRandn(215, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return sn.Forward(x).Square().Mean() }
	gradCheck(t, "SpectralNormLinear", loss, append(sn.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckSpectralNormConv2d(t *testing.T) {
	sn := NewSpectralNormConv2d(NewConv2d(2, 3, 3, WithPad(1)))
	sn.Eval()
	x := seededRandn(216, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return sn.Forward(x).Square().Mean() }
	gradCheck(t, "SpectralNormConv2d", loss, append(sn.Parameters(), x), gcEps, gcTol, 40)
}

func TestRemoveSpectralNorm(t *testing.T) {
	sn := NewSpectralNormLinear(NewLinear(5, 3, true))
	sn.Eval()
	x := seededRandn(217, 4, 5)
	want := sn.Forward(x)
	plain := RemoveSpectralNormLinear(sn)
	got := plain.Forward(x)
	if d := maxAbsDiff64(t, got.Data, want.Data); d > 1e-12 {
		t.Errorf("removed spectral-norm linear differs from wrapper forward: max diff %g", d)
	}

	snc := NewSpectralNormConv2d(NewConv2d(2, 3, 3, WithStride(2), WithPad(1)))
	snc.Eval()
	xc := seededRandn(218, 2, 2, 6, 6)
	wantC := snc.Forward(xc)
	plainC := RemoveSpectralNormConv2d(snc)
	gotC := plainC.Forward(xc)
	if d := maxAbsDiff64(t, gotC.Data, wantC.Data); d > 1e-12 {
		t.Errorf("removed spectral-norm conv differs from wrapper forward: max diff %g", d)
	}
}
