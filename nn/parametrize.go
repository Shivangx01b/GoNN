package nn

// Weight parametrizations: weight normalization and spectral normalization,
// mirroring torch.nn.utils.parametrizations.{weight_norm, spectral_norm}.
//
// Design deviation from PyTorch (documented, deliberate): Go modules have no
// forward pre-hooks, so a parametrization here is an explicit wrapper module
// that OWNS the raw parameters (g/v for weight norm, weight_orig for spectral
// norm) and recomputes the wrapped layer's effective weight with
// differentiable tensor ops on every Forward. Wrap a trained layer with
// NewWeightNormLinear / NewSpectralNormLinear (or the Conv2d variants) and
// use the wrapper in place of the original layer from then on; the wrapper's
// Parameters() are what the optimizer should train. The original layer's own
// registered weight is never mutated. RemoveWeightNorm* / RemoveSpectralNorm*
// bake the current effective weight into a fresh plain layer, matching
// torch.nn.utils.remove_weight_norm / parametrize.remove_parametrizations.

import (
	"fmt"
	"math"

	"gonn/tensor"
)

// ---- shared forward helpers -------------------------------------------------

// applyLinear runs the Linear forward math (x @ W^T + b) with an explicit,
// possibly recomputed, weight tensor. Same shape handling as Linear.Forward.
func applyLinear(x, weight, bias *tensor.Tensor, in, out int) *tensor.Tensor {
	origShape := x.Shape
	feat := origShape[len(origShape)-1]
	if feat != in {
		panic("nn: parametrized linear forward: input last dim does not match InFeatures")
	}
	batch := 1
	for i := 0; i < len(origShape)-1; i++ {
		batch *= origShape[i]
	}
	x2 := x.Reshape(batch, feat)
	y := x2.MatMul(weight.Transpose())
	if bias != nil {
		y = y.Add(bias)
	}
	outShape := append([]int(nil), origShape[:len(origShape)-1]...)
	outShape = append(outShape, out)
	return y.Reshape(outShape...)
}

// convForwardWithWeight runs the shared im2col + GEMM convolution of convNd
// with an explicit weight tensor (non-transposed (OutC, InC, K...) layout).
// The geometry and gather cache come from c; c's own registered weight is not
// touched.
func convForwardWithWeight(c *convNd, x, w, bias *tensor.Tensor) *tensor.Tensor {
	rank := len(c.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: conv expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	if x.Shape[1] != c.InC {
		panic(fmt.Sprintf("nn: conv input channels %d != InC %d", x.Shape[1], c.InC))
	}
	N := x.Shape[0]
	g, out, numWin, winSize := c.gatherFor(x.Shape[2:])

	col := unfold(x, g, numWin, winSize)                          // (N*numWin, C*winSize)
	y := col.MatMul(w.Reshape(c.OutC, c.InC*winSize).Transpose()) // (N*numWin, OutC)
	if bias != nil {
		y = y.Add(bias)
	}
	shape := append(append([]int{N}, out...), c.OutC)
	perm := make([]int, rank+2)
	perm[0], perm[1] = 0, rank+1
	for d := 0; d < rank; d++ {
		perm[2+d] = 1 + d
	}
	return y.Reshape(shape...).Permute(perm...)
}

// rowNorms returns the per-row L2 norms of the (rows, cols) row-major data.
func rowNorms(data []float64, rows, cols int) []float64 {
	out := make([]float64, rows)
	for r := 0; r < rows; r++ {
		s := 0.0
		for c := 0; c < cols; c++ {
			w := data[r*cols+c]
			s += w * w
		}
		out[r] = math.Sqrt(s)
	}
	return out
}

// weightNormEffective computes w = g * v / ||v||_row differentiably, where v
// is viewed as a (rows, cols) matrix and g has shape (rows,). The result is
// reshaped back to v's original shape.
func weightNormEffective(g, v *tensor.Tensor, rows, cols int) *tensor.Tensor {
	vm := v.Reshape(rows, cols)
	norm := vm.Square().SumAxis(1, true).Sqrt() // (rows, 1)
	scale := g.Reshape(rows, 1).Div(norm)       // (rows, 1)
	return vm.Mul(scale).Reshape(v.Shape...)
}

// ---- weight norm ------------------------------------------------------------

// WeightNormLinear reparametrizes a Linear layer's weight as
//
//	w = g * v / ||v||        (norm taken per output row, dim=0 in PyTorch terms)
//
// g ("weight_g", shape (Out,)) and v ("weight_v", shape (Out, In)) are this
// module's trainable parameters; the wrapped layer's bias tensor is reused
// (shared, registered here as "bias"). The wrapped layer's own weight is left
// untouched and is not part of this module's parameters.
type WeightNormLinear struct {
	Base
	InFeatures  int
	OutFeatures int
	G           *tensor.Tensor // (Out,)
	V           *tensor.Tensor // (Out, In)
	Bias        *tensor.Tensor // shared with the wrapped layer, or nil
}

// NewWeightNormLinear wraps l with weight normalization, initializing
// g = ||w||_row and v = w from l's current weight (PyTorch weight_norm init,
// so the wrapper computes exactly the same function as l at wrap time).
func NewWeightNormLinear(l *Linear) *WeightNormLinear {
	out, in := l.OutFeatures, l.InFeatures
	v := make([]float64, out*in)
	copy(v, l.Weight.Data)
	m := &WeightNormLinear{InFeatures: in, OutFeatures: out}
	m.G = m.reg("weight_g", tensor.New(rowNorms(v, out, in), out).SetRequiresGrad(true))
	m.V = m.reg("weight_v", tensor.New(v, out, in).SetRequiresGrad(true))
	if l.Bias != nil {
		m.Bias = m.reg("bias", l.Bias)
	}
	return m
}

// EffectiveWeight recomputes w = g * v/||v||_row with differentiable ops.
func (m *WeightNormLinear) EffectiveWeight() *tensor.Tensor {
	return weightNormEffective(m.G, m.V, m.OutFeatures, m.InFeatures)
}

// Forward computes x @ w^T + b with the recomputed effective weight.
func (m *WeightNormLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return applyLinear(x, m.EffectiveWeight(), m.Bias, m.InFeatures, m.OutFeatures)
}

// RemoveWeightNormLinear bakes the current effective weight into a fresh
// plain Linear (the analog of torch.nn.utils.remove_weight_norm).
func RemoveWeightNormLinear(m *WeightNormLinear) *Linear {
	l := NewLinear(m.InFeatures, m.OutFeatures, m.Bias != nil)
	copy(l.Weight.Data, m.EffectiveWeight().Data)
	if m.Bias != nil {
		copy(l.Bias.Data, m.Bias.Data)
	}
	return l
}

// WeightNormConv2d reparametrizes a Conv2d weight as w = g * v/||v|| with the
// norm taken per output channel over the (InC, KH, KW) block (dim=0). The
// wrapped conv provides geometry (and its gather cache); its registered
// weight is left untouched.
type WeightNormConv2d struct {
	Base
	inner *Conv2d
	G     *tensor.Tensor // (OutC,)
	V     *tensor.Tensor // (OutC, InC, KH, KW)
	Bias  *tensor.Tensor // shared with the wrapped layer, or nil
}

// NewWeightNormConv2d wraps c with weight normalization initialized from its
// current weight (g = per-out-channel norm, v = weight).
func NewWeightNormConv2d(c *Conv2d) *WeightNormConv2d {
	rows := c.OutC
	cols := c.InC * prodInts(c.Kernel)
	v := make([]float64, rows*cols)
	copy(v, c.Weight.Data)
	m := &WeightNormConv2d{inner: c}
	m.G = m.reg("weight_g", tensor.New(rowNorms(v, rows, cols), rows).SetRequiresGrad(true))
	m.V = m.reg("weight_v", tensor.New(v, c.Weight.Shape...).SetRequiresGrad(true))
	if c.Bias != nil {
		m.Bias = m.reg("bias", c.Bias)
	}
	return m
}

// EffectiveWeight recomputes w = g * v/||v|| per output channel.
func (m *WeightNormConv2d) EffectiveWeight() *tensor.Tensor {
	return weightNormEffective(m.G, m.V, m.inner.OutC, m.inner.InC*prodInts(m.inner.Kernel))
}

// Forward runs the wrapped conv's im2col+GEMM math with the effective weight.
func (m *WeightNormConv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return convForwardWithWeight(&m.inner.convNd, x, m.EffectiveWeight(), m.Bias)
}

// RemoveWeightNormConv2d bakes the current effective weight into a fresh
// plain Conv2d with the same geometry.
func RemoveWeightNormConv2d(m *WeightNormConv2d) *Conv2d {
	c := newConv2dLike(m.inner, m.Bias != nil)
	copy(c.Weight.Data, m.EffectiveWeight().Data)
	if m.Bias != nil {
		copy(c.Bias.Data, m.Bias.Data)
	}
	return c
}

// newConv2dLike constructs a fresh Conv2d with the same geometry as c.
// The fresh constructor draws RNG; callers overwrite the data in place.
func newConv2dLike(c *Conv2d, bias bool) *Conv2d {
	opts := []ConvOpt{
		WithKernel(c.Kernel...), WithStride(c.Stride...),
		WithPad(c.Pad...), WithDilation(c.Dilation...),
	}
	if c.Groups > 1 {
		opts = append(opts, WithGroups(c.Groups))
	}
	if c.PaddingMode != "" && c.PaddingMode != "zeros" {
		opts = append(opts, WithPaddingMode(c.PaddingMode))
	}
	if !bias {
		opts = append(opts, WithNoBias())
	}
	return NewConv2d(c.InC, c.OutC, c.Kernel[0], opts...)
}

// ---- spectral norm ----------------------------------------------------------

// SpectralNormOpt configures spectral normalization wrappers.
type SpectralNormOpt func(*snOpts)

type snOpts struct {
	nPowerIterations int
	eps              float64
}

// WithNPowerIterations sets how many power iterations run per training-mode
// forward (default 1, like PyTorch).
func WithNPowerIterations(n int) SpectralNormOpt {
	return func(o *snOpts) { o.nPowerIterations = n }
}

// WithSpectralEps sets the normalization epsilon (default 1e-12).
func WithSpectralEps(eps float64) SpectralNormOpt {
	return func(o *snOpts) { o.eps = eps }
}

func resolveSNOpts(opts []SpectralNormOpt) snOpts {
	o := snOpts{nPowerIterations: 1, eps: 1e-12}
	for _, fn := range opts {
		fn(&o)
	}
	return o
}

// l2NormalizeInPlace scales x to unit L2 norm (with eps in the denominator).
func l2NormalizeInPlace(x []float64, eps float64) {
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	inv := 1.0 / (math.Sqrt(s) + eps)
	for i := range x {
		x[i] *= inv
	}
}

// powerIterate runs iters power iterations on the m x n row-major matrix w,
// updating the singular-vector estimates u (len m) and v (len n) in place.
// Plain float64 math: no autograd flows through the iteration, matching
// PyTorch (u and v are buffers, updated under no_grad).
func powerIterate(w []float64, m, n int, u, v []float64, iters int, eps float64) {
	for it := 0; it < iters; it++ {
		for j := 0; j < n; j++ { // v = normalize(W^T u)
			s := 0.0
			for i := 0; i < m; i++ {
				s += w[i*n+j] * u[i]
			}
			v[j] = s
		}
		l2NormalizeInPlace(v, eps)
		for i := 0; i < m; i++ { // u = normalize(W v)
			s := 0.0
			for j := 0; j < n; j++ {
				s += w[i*n+j] * v[j]
			}
			u[i] = s
		}
		l2NormalizeInPlace(u, eps)
	}
}

// snSigmaTensor builds sigma = u^T W v as a (1,1) tensor. u and v are
// materialized as constant (non-grad) tensors, so gradients flow to the
// weight only — exactly PyTorch's treatment, which differentiates sigma
// w.r.t. W while holding u, v fixed.
func snSigmaTensor(weight2d *tensor.Tensor, u, v []float64) *tensor.Tensor {
	m, n := weight2d.Shape[0], weight2d.Shape[1]
	ut := tensor.New(append([]float64(nil), u...), 1, m)
	vt := tensor.New(append([]float64(nil), v...), n, 1)
	return ut.MatMul(weight2d).MatMul(vt) // (1, 1)
}

// snSigmaValue computes u^T W v as a plain float64 (no graph).
func snSigmaValue(w []float64, m, n int, u, v []float64) float64 {
	s := 0.0
	for i := 0; i < m; i++ {
		row := 0.0
		for j := 0; j < n; j++ {
			row += w[i*n+j] * v[j]
		}
		s += u[i] * row
	}
	return s
}

// initSNVectors allocates randomly initialized, normalized u/v estimates and
// runs 15 warm-up power iterations (PyTorch's parametrization init does the
// same) so sigma starts near the true top singular value.
func initSNVectors(w []float64, m, n int, eps float64) (u, v []float64) {
	u = tensor.Randn(m).Data
	v = tensor.Randn(n).Data
	l2NormalizeInPlace(u, eps)
	l2NormalizeInPlace(v, eps)
	powerIterate(w, m, n, u, v, 15, eps)
	return u, v
}

// SpectralNormLinear reparametrizes a Linear layer's weight as
//
//	w_sn = w / sigma,   sigma = u^T w v
//
// where u, v are power-iteration estimates of the top singular vectors,
// stored as buffers and updated (without grad) only during training-mode
// forwards. The wrapped layer's weight tensor is registered here as
// "weight_orig" (shared) and keeps training; sigma is computed with tensor
// ops from it, so gradients flow through both the numerator and sigma,
// matching PyTorch.
type SpectralNormLinear struct {
	Base
	InFeatures       int
	OutFeatures      int
	Weight           *tensor.Tensor // shared with the wrapped layer ("weight_orig")
	Bias             *tensor.Tensor // shared with the wrapped layer, or nil
	U                *tensor.Tensor // buffer, (Out,)
	V                *tensor.Tensor // buffer, (In,)
	NPowerIterations int
	Eps              float64
}

// NewSpectralNormLinear wraps l with spectral normalization.
func NewSpectralNormLinear(l *Linear, opts ...SpectralNormOpt) *SpectralNormLinear {
	o := resolveSNOpts(opts)
	out, in := l.OutFeatures, l.InFeatures
	s := &SpectralNormLinear{
		InFeatures: in, OutFeatures: out,
		NPowerIterations: o.nPowerIterations, Eps: o.eps,
	}
	s.Weight = s.reg("weight_orig", l.Weight)
	if l.Bias != nil {
		s.Bias = s.reg("bias", l.Bias)
	}
	u, v := initSNVectors(l.Weight.Data, out, in, o.eps)
	s.U = s.regBuf("u", tensor.New(u, out))
	s.V = s.regBuf("v", tensor.New(v, in))
	return s
}

// EstimatedSigma returns the current sigma estimate u^T W v (no graph).
func (s *SpectralNormLinear) EstimatedSigma() float64 {
	return snSigmaValue(s.Weight.Data, s.OutFeatures, s.InFeatures, s.U.Data, s.V.Data)
}

// EffectiveWeight returns w / sigma with sigma differentiable w.r.t. w.
func (s *SpectralNormLinear) EffectiveWeight() *tensor.Tensor {
	sigma := snSigmaTensor(s.Weight, s.U.Data, s.V.Data)
	return s.Weight.Div(sigma)
}

// Forward updates u/v by power iteration (training mode only), then computes
// x @ (w/sigma)^T + b.
func (s *SpectralNormLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	if s.Training() {
		powerIterate(s.Weight.Data, s.OutFeatures, s.InFeatures,
			s.U.Data, s.V.Data, s.NPowerIterations, s.Eps)
	}
	return applyLinear(x, s.EffectiveWeight(), s.Bias, s.InFeatures, s.OutFeatures)
}

// RemoveSpectralNormLinear bakes the current normalized weight into a fresh
// plain Linear.
func RemoveSpectralNormLinear(s *SpectralNormLinear) *Linear {
	l := NewLinear(s.InFeatures, s.OutFeatures, s.Bias != nil)
	sigma := s.EstimatedSigma()
	for i, w := range s.Weight.Data {
		l.Weight.Data[i] = w / sigma
	}
	if s.Bias != nil {
		copy(l.Bias.Data, s.Bias.Data)
	}
	return l
}

// SpectralNormConv2d applies spectral normalization to a Conv2d weight viewed
// as the (OutC, InC*KH*KW) matrix (PyTorch flattens the same way for dim=0).
// See SpectralNormLinear for the update/gradient semantics.
type SpectralNormConv2d struct {
	Base
	inner            *Conv2d
	Weight           *tensor.Tensor // shared with the wrapped layer ("weight_orig")
	Bias             *tensor.Tensor // shared with the wrapped layer, or nil
	U                *tensor.Tensor // buffer, (OutC,)
	V                *tensor.Tensor // buffer, (InC*KH*KW,)
	NPowerIterations int
	Eps              float64
}

// NewSpectralNormConv2d wraps c with spectral normalization.
func NewSpectralNormConv2d(c *Conv2d, opts ...SpectralNormOpt) *SpectralNormConv2d {
	o := resolveSNOpts(opts)
	m := c.OutC
	n := c.InC * prodInts(c.Kernel)
	s := &SpectralNormConv2d{
		inner:            c,
		NPowerIterations: o.nPowerIterations, Eps: o.eps,
	}
	s.Weight = s.reg("weight_orig", c.Weight)
	if c.Bias != nil {
		s.Bias = s.reg("bias", c.Bias)
	}
	u, v := initSNVectors(c.Weight.Data, m, n, o.eps)
	s.U = s.regBuf("u", tensor.New(u, m))
	s.V = s.regBuf("v", tensor.New(v, n))
	return s
}

func (s *SpectralNormConv2d) matDims() (m, n int) {
	return s.inner.OutC, s.inner.InC * prodInts(s.inner.Kernel)
}

// EstimatedSigma returns the current sigma estimate u^T W v (no graph).
func (s *SpectralNormConv2d) EstimatedSigma() float64 {
	m, n := s.matDims()
	return snSigmaValue(s.Weight.Data, m, n, s.U.Data, s.V.Data)
}

// EffectiveWeight returns w / sigma with sigma differentiable w.r.t. w.
func (s *SpectralNormConv2d) EffectiveWeight() *tensor.Tensor {
	m, n := s.matDims()
	sigma := snSigmaTensor(s.Weight.Reshape(m, n), s.U.Data, s.V.Data)
	return s.Weight.Div(sigma)
}

// Forward updates u/v by power iteration (training mode only), then runs the
// wrapped conv's math with the normalized weight.
func (s *SpectralNormConv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if s.Training() {
		m, n := s.matDims()
		powerIterate(s.Weight.Data, m, n, s.U.Data, s.V.Data, s.NPowerIterations, s.Eps)
	}
	return convForwardWithWeight(&s.inner.convNd, x, s.EffectiveWeight(), s.Bias)
}

// RemoveSpectralNormConv2d bakes the current normalized weight into a fresh
// plain Conv2d with the same geometry.
func RemoveSpectralNormConv2d(s *SpectralNormConv2d) *Conv2d {
	c := newConv2dLike(s.inner, s.Bias != nil)
	sigma := s.EstimatedSigma()
	for i, w := range s.Weight.Data {
		c.Weight.Data[i] = w / sigma
	}
	if s.Bias != nil {
		copy(c.Bias.Data, s.Bias.Data)
	}
	return c
}
