package nn

import (
	"math"
	"math/rand"
	"testing"

	"gonn/tensor"
)

// gradCheck verifies autograd gradients against central finite differences.
// loss must be a deterministic scalar-producing closure that re-runs the full
// forward pass on every call (no dropout / stochastic ops). maxElems > 0
// limits the number of elements checked per tensor (sampled with a fixed
// seed) so large parameters stay cheap.
func gradCheck(t *testing.T, name string, loss func() *tensor.Tensor, wrt []*tensor.Tensor, eps, tol float64, maxElems int) {
	t.Helper()

	for _, p := range wrt {
		p.ZeroGrad()
	}
	out := loss()
	if out.Numel() != 1 {
		t.Fatalf("%s: loss must be scalar, got shape %v", name, out.Shape)
	}
	out.Backward()

	analytic := make([][]float64, len(wrt))
	for i, p := range wrt {
		analytic[i] = make([]float64, len(p.Data))
		if p.Grad != nil {
			copy(analytic[i], p.Grad.Data)
		}
	}

	rng := rand.New(rand.NewSource(42))
	for i, p := range wrt {
		idxs := sampleIndices(rng, len(p.Data), maxElems)
		for _, j := range idxs {
			orig := p.Data[j]
			p.Data[j] = orig + eps
			fp := loss().Item()
			p.Data[j] = orig - eps
			fm := loss().Item()
			p.Data[j] = orig

			num := (fp - fm) / (2 * eps)
			got := analytic[i][j]
			denom := math.Max(1, math.Abs(num)+math.Abs(got))
			if math.Abs(num-got)/denom > tol {
				t.Errorf("%s: wrt[%d].Data[%d]: analytic=%.8g numeric=%.8g relerr=%.3g",
					name, i, j, got, num, math.Abs(num-got)/denom)
			}
		}
	}
}

// sampleIndices returns up to maxElems distinct indices in [0, n); maxElems
// <= 0 means all indices.
func sampleIndices(rng *rand.Rand, n, maxElems int) []int {
	if maxElems <= 0 || n <= maxElems {
		idxs := make([]int, n)
		for i := range idxs {
			idxs[i] = i
		}
		return idxs
	}
	perm := rng.Perm(n)
	return perm[:maxElems]
}

const (
	gcEps = 1e-5
	gcTol = 1e-4
)

// seededRandn returns a deterministic N(0,1) tensor (private stream, does not
// disturb the global rand state used by module init).
func seededRandn(seed int64, shape ...int) *tensor.Tensor {
	rng := rand.New(rand.NewSource(seed))
	t := tensor.Zeros(shape...)
	for i := range t.Data {
		t.Data[i] = rng.NormFloat64()
	}
	return t
}

func TestGradCheckLinear(t *testing.T) {
	l := NewLinear(4, 3, true)
	x := seededRandn(1, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return l.Forward(x).Square().Mean() }
	gradCheck(t, "Linear", loss, append(l.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckBilinear(t *testing.T) {
	b := NewBilinear(3, 4, 2, true)
	x1 := seededRandn(2, 2, 3).SetRequiresGrad(true)
	x2 := seededRandn(3, 2, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return b.Forward(x1, x2).Square().Mean() }
	gradCheck(t, "Bilinear", loss, append(b.Parameters(), x1, x2), gcEps, gcTol, 0)
}

func TestGradCheckEmbedding(t *testing.T) {
	e := NewEmbedding(5, 3)
	idx := tensor.New([]float64{0, 2, 4, 2}, 4)
	loss := func() *tensor.Tensor { return e.Forward(idx).Square().Mean() }
	gradCheck(t, "Embedding", loss, e.Parameters(), gcEps, gcTol, 0)
}

func TestGradCheckConv1d(t *testing.T) {
	c := NewConv1d(2, 3, 3, 2, 1, true)
	x := seededRandn(4, 2, 2, 7).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "Conv1d", loss, append(c.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckConv2d(t *testing.T) {
	c := NewConv2d(2, 3, 3, 2, 1, true)
	x := seededRandn(5, 2, 2, 6, 7).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "Conv2d", loss, append(c.Parameters(), x), gcEps, gcTol, 40)
}

func TestGradCheckConv3d(t *testing.T) {
	c := NewConv3d(2, 2, 2, 1, 0, true)
	x := seededRandn(6, 1, 2, 3, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "Conv3d", loss, append(c.Parameters(), x), gcEps, gcTol, 30)
}

func TestGradCheckConvTranspose2d(t *testing.T) {
	c := NewConvTranspose2d(2, 3, 3, 2, 1, true)
	x := seededRandn(7, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "ConvTranspose2d", loss, append(c.Parameters(), x), gcEps, gcTol, 40)
}

func TestGradCheckMaxPool2d(t *testing.T) {
	p := NewMaxPool2d(2, 2)
	x := seededRandn(8, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "MaxPool2d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckAvgPool3d(t *testing.T) {
	p := NewAvgPool3d(2, 2)
	x := seededRandn(9, 2, 1, 4, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "AvgPool3d", loss, []*tensor.Tensor{x}, gcEps, gcTol, 40)
}

func TestGradCheckLayerNorm(t *testing.T) {
	ln := NewLayerNorm(6)
	x := seededRandn(10, 3, 6).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return ln.Forward(x).Square().Mean() }
	gradCheck(t, "LayerNorm", loss, append(ln.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckRMSNorm(t *testing.T) {
	rn := NewRMSNorm(6)
	x := seededRandn(11, 3, 6).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return rn.Forward(x).Square().Mean() }
	gradCheck(t, "RMSNorm", loss, append(rn.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckBatchNorm1d(t *testing.T) {
	bn := NewBatchNorm1d(4)
	// Training mode: normalizes with batch stats; running-stat updates are a
	// side effect that does not feed back into the training-mode output.
	x := seededRandn(12, 5, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return bn.Forward(x).Square().Mean() }
	gradCheck(t, "BatchNorm1d", loss, append(bn.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckGroupNorm(t *testing.T) {
	gn := NewGroupNorm(2, 4)
	x := seededRandn(13, 3, 4, 5).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return gn.Forward(x).Square().Mean() }
	gradCheck(t, "GroupNorm", loss, append(gn.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckInstanceNorm2d(t *testing.T) {
	in := NewInstanceNorm2d(2, true)
	x := seededRandn(14, 2, 2, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return in.Forward(x).Square().Mean() }
	gradCheck(t, "InstanceNorm2d", loss, append(in.Parameters(), x), gcEps, gcTol, 40)
}

func TestGradCheckPReLU(t *testing.T) {
	p := NewPReLU(3)
	x := seededRandn(15, 2, 3, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return p.Forward(x).Square().Mean() }
	gradCheck(t, "PReLU", loss, append(p.Parameters(), x), gcEps, gcTol, 0)
}

func TestGradCheckGLU(t *testing.T) {
	g := GLU{Dim: -1}
	x := seededRandn(16, 3, 8).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return g.Forward(x).Square().Mean() }
	gradCheck(t, "GLU", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}

func TestGradCheckActivations(t *testing.T) {
	acts := []struct {
		name string
		f    func(x *tensor.Tensor) *tensor.Tensor
	}{
		{"GELU", func(x *tensor.Tensor) *tensor.Tensor { return x.GELU() }},
		{"SiLU", func(x *tensor.Tensor) *tensor.Tensor { return x.SiLU() }},
		{"ELU", func(x *tensor.Tensor) *tensor.Tensor { return x.ELU(1.0) }},
		{"Softplus", func(x *tensor.Tensor) *tensor.Tensor { return x.Softplus() }},
		{"Mish", func(x *tensor.Tensor) *tensor.Tensor { return x.Mish() }},
		{"Tanh", func(x *tensor.Tensor) *tensor.Tensor { return x.Tanh() }},
		{"Sigmoid", func(x *tensor.Tensor) *tensor.Tensor { return x.Sigmoid() }},
		{"LeakyReLU", func(x *tensor.Tensor) *tensor.Tensor { return x.LeakyReLU(0.01) }},
	}
	for _, a := range acts {
		x := seededRandn(17, 3, 5).SetRequiresGrad(true)
		f := a.f
		loss := func() *tensor.Tensor { return f(x).Square().Mean() }
		gradCheck(t, a.name, loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
	}
}

func TestGradCheckMultiHeadAttention(t *testing.T) {
	m := NewMultiHeadAttention(8, 2)
	q := seededRandn(18, 2, 3, 8).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return m.Forward(q, q, q, true).Square().Mean() }
	gradCheck(t, "MultiHeadAttention", loss, append(m.Parameters(), q), gcEps, gcTol, 30)
}

func TestGradCheckRNNFamily(t *testing.T) {
	x := seededRandn(19, 2, 3, 3).SetRequiresGrad(true)

	r := NewRNN(3, 4)
	loss := func() *tensor.Tensor { return r.Forward(x).Square().Mean() }
	gradCheck(t, "RNN", loss, append(r.Parameters(), x), gcEps, gcTol, 20)

	l := NewLSTM(3, 4)
	loss = func() *tensor.Tensor { return l.Forward(x).Square().Mean() }
	gradCheck(t, "LSTM", loss, append(l.Parameters(), x), gcEps, gcTol, 20)

	g := NewGRU(3, 4)
	loss = func() *tensor.Tensor { return g.Forward(x).Square().Mean() }
	gradCheck(t, "GRU", loss, append(g.Parameters(), x), gcEps, gcTol, 20)
}

func TestGradCheckMultiLayerLSTM(t *testing.T) {
	l := NewMultiLayerLSTM(3, 4, 2, true)
	x := seededRandn(20, 2, 3, 3).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return l.Forward(x).Square().Mean() }
	gradCheck(t, "MultiLayerLSTM", loss, append(l.Parameters(), x), gcEps, gcTol, 20)
}

func TestGradCheckLosses(t *testing.T) {
	pred := seededRandn(21, 4, 3).SetRequiresGrad(true)
	target := seededRandn(22, 4, 3)

	gradCheck(t, "MSELoss",
		func() *tensor.Tensor { return MSELoss(pred, target) },
		[]*tensor.Tensor{pred}, gcEps, gcTol, 0)

	gradCheck(t, "HuberLoss",
		func() *tensor.Tensor { return HuberLoss(pred, target, 1.0) },
		[]*tensor.Tensor{pred}, gcEps, gcTol, 0)

	logits := seededRandn(23, 4, 3).SetRequiresGrad(true)
	classes := tensor.New([]float64{0, 2, 1, 2}, 4)
	gradCheck(t, "CrossEntropyLoss",
		func() *tensor.Tensor { return CrossEntropyLoss(logits, classes) },
		[]*tensor.Tensor{logits}, gcEps, gcTol, 0)

	z := seededRandn(24, 4, 3).SetRequiresGrad(true)
	bt := tensor.New([]float64{1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0}, 4, 3)
	gradCheck(t, "BCEWithLogitsLoss",
		func() *tensor.Tensor { return BCEWithLogitsLoss(z, bt) },
		[]*tensor.Tensor{z}, gcEps, gcTol, 0)

	a := seededRandn(25, 4, 3).SetRequiresGrad(true)
	p := seededRandn(26, 4, 3).SetRequiresGrad(true)
	n := seededRandn(27, 4, 3).SetRequiresGrad(true)
	gradCheck(t, "TripletMarginLoss",
		func() *tensor.Tensor { return TripletMarginLoss(a, p, n, 1.0) },
		[]*tensor.Tensor{a, p, n}, gcEps, gcTol, 0)
}
