package nn

import (
	"fmt"
	"math/rand"

	"gonn/tensor"
)

// Activation is the generic elementwise activation module. Fixed (parameter-
// free) kinds route through the tensor package's unary-op registry by name —
// the same single definition that powers the fluent tensor methods and GPU
// dispatch — while parameterized kinds call the corresponding fluent method
// with Alpha/Beta. Construct via the named constructors:
//
//	model := nn.NewSequential(
//	    nn.NewLinear(784, 256, true),
//	    nn.ReLU(),
//	    nn.NewLinear(256, 10, true),
//	)
type Activation struct {
	Base
	// Kind is the registry name for fixed activations ("relu", "gelu", ...)
	// or the fluent-method name for parameterized ones ("leakyrelu", ...).
	Kind string
	// Alpha carries the first parameter (slope/alpha/lambda/threshold).
	Alpha float64
	// Beta carries the second parameter (Threshold's replacement value).
	Beta float64

	parameterized bool
}

// Forward applies the activation.
func (a *Activation) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !a.parameterized {
		return x.Unary(a.Kind)
	}
	switch a.Kind {
	case "leakyrelu":
		return x.LeakyReLU(a.Alpha)
	case "elu":
		return x.ELU(a.Alpha)
	case "celu":
		return x.CELU(a.Alpha)
	case "hardshrink":
		return x.Hardshrink(a.Alpha)
	case "softshrink":
		return x.Softshrink(a.Alpha)
	case "threshold":
		return x.Threshold(a.Alpha, a.Beta)
	case "softplusbeta":
		return x.SoftplusBeta(a.Alpha, a.Beta)
	default:
		panic(fmt.Sprintf("nn: unknown parameterized activation %q", a.Kind))
	}
}

func fixedAct(kind string) *Activation { return &Activation{Kind: kind} }

// ReLU returns a ReLU activation module: max(0, x).
func ReLU() *Activation { return fixedAct("relu") }

// ReLU6 returns a ReLU6 activation module: min(max(0, x), 6).
func ReLU6() *Activation { return fixedAct("relu6") }

// Sigmoid returns a Sigmoid activation module.
func Sigmoid() *Activation { return fixedAct("sigmoid") }

// Tanh returns a Tanh activation module.
func Tanh() *Activation { return fixedAct("tanh") }

// GELU returns a GELU activation module (tanh approximation).
func GELU() *Activation { return fixedAct("gelu") }

// GELUExact returns the exact-erf GELU activation module:
// 0.5*x*(1+erf(x/sqrt(2))), matching torch.nn.GELU(approximate='none').
func GELUExact() *Activation { return fixedAct("geluexact") }

// GELUApprox mirrors torch.nn.GELU(approximate=...): "none" selects the
// exact-erf GELU, "tanh" the tanh approximation. Panics on any other value.
func GELUApprox(approximate string) *Activation {
	switch approximate {
	case "none":
		return fixedAct("geluexact")
	case "tanh":
		return fixedAct("gelu")
	default:
		panic(fmt.Sprintf("nn: GELUApprox: approximate must be %q or %q, got %q", "none", "tanh", approximate))
	}
}

// SiLU returns a SiLU (Swish) activation module: x * sigmoid(x).
func SiLU() *Activation { return fixedAct("silu") }

// SELU returns a SELU activation module.
func SELU() *Activation { return fixedAct("selu") }

// HardTanh returns a HardTanh activation module (clamp to [-1, 1]).
func HardTanh() *Activation { return fixedAct("hardtanh") }

// HardSigmoid returns a HardSigmoid activation module.
func HardSigmoid() *Activation { return fixedAct("hardsigmoid") }

// HardSwish returns a HardSwish activation module.
func HardSwish() *Activation { return fixedAct("hardswish") }

// Mish returns a Mish activation module: x * tanh(softplus(x)).
func Mish() *Activation { return fixedAct("mish") }

// Softplus returns a Softplus activation module: ln(1+e^x).
func Softplus() *Activation { return fixedAct("softplus") }

// Softsign returns a Softsign activation module: x/(1+|x|).
func Softsign() *Activation { return fixedAct("softsign") }

// LogSigmoid returns a LogSigmoid activation module.
func LogSigmoid() *Activation { return fixedAct("logsigmoid") }

// Tanhshrink returns a Tanhshrink activation module: x - tanh(x).
func Tanhshrink() *Activation { return fixedAct("tanhshrink") }

// LeakyReLU returns a LeakyReLU activation module with the given slope.
func LeakyReLU(alpha float64) *Activation {
	return &Activation{Kind: "leakyrelu", Alpha: alpha, parameterized: true}
}

// ELU returns an ELU activation module with the given alpha.
func ELU(alpha float64) *Activation {
	return &Activation{Kind: "elu", Alpha: alpha, parameterized: true}
}

// CELU returns a CELU activation module with the given alpha.
func CELU(alpha float64) *Activation {
	return &Activation{Kind: "celu", Alpha: alpha, parameterized: true}
}

// Hardshrink returns a Hardshrink activation module with threshold lambda.
func Hardshrink(lambda float64) *Activation {
	return &Activation{Kind: "hardshrink", Alpha: lambda, parameterized: true}
}

// Softshrink returns a Softshrink activation module with threshold lambda.
func Softshrink(lambda float64) *Activation {
	return &Activation{Kind: "softshrink", Alpha: lambda, parameterized: true}
}

// Threshold returns a Threshold activation module: x if x > thresh else value.
func Threshold(thresh, value float64) *Activation {
	return &Activation{Kind: "threshold", Alpha: thresh, Beta: value, parameterized: true}
}

// SoftplusWith returns a parameterized Softplus activation module,
// matching torch.nn.Softplus(beta, threshold):
// (1/beta)*log(1+exp(beta*x)), linear where beta*x > threshold.
// PyTorch defaults are beta=1, threshold=20 (== the zero-arg Softplus()).
func SoftplusWith(beta, threshold float64) *Activation {
	return &Activation{Kind: "softplusbeta", Alpha: beta, Beta: threshold, parameterized: true}
}

// ActivationByName returns an Activation for any registered fixed unary op
// name (see tensor.UnaryOpNames), e.g. nn.ActivationByName("gelu").
func ActivationByName(name string) *Activation {
	if _, ok := tensor.LookupUnary(name); !ok {
		panic(fmt.Sprintf("nn: unknown activation %q (registered: %v)", name, tensor.UnaryOpNames()))
	}
	return fixedAct(name)
}

// ---- Axis-reduction activations (not elementwise) ---------------------------

// Softmax applies softmax along Axis.
type Softmax struct {
	Base
	Axis int
}

// NewSoftmax constructs a Softmax module over the given axis.
func NewSoftmax(axis int) *Softmax { return &Softmax{Axis: axis} }

// Forward applies softmax along Axis.
func (s *Softmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softmax(s.Axis) }

// LogSoftmax applies log-softmax along Axis.
type LogSoftmax struct {
	Base
	Axis int
}

// NewLogSoftmax constructs a LogSoftmax module over the given axis.
func NewLogSoftmax(axis int) *LogSoftmax { return &LogSoftmax{Axis: axis} }

// Forward applies log-softmax along Axis.
func (s *LogSoftmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.LogSoftmax(s.Axis) }

// Softmin applies softmax(-x) along Axis, mirroring torch.nn.Softmin.
type Softmin struct {
	Base
	Axis int
}

// NewSoftmin constructs a Softmin module over the given axis.
func NewSoftmin(axis int) *Softmin { return &Softmin{Axis: axis} }

// Forward returns softmax(-x).
func (s *Softmin) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Neg().Softmax(s.Axis) }

// Softmax2d applies softmax over the channel dimension (axis 1) of an NCHW
// tensor, mirroring torch.nn.Softmax2d.
type Softmax2d struct{ Base }

// NewSoftmax2d constructs a Softmax2d module.
func NewSoftmax2d() *Softmax2d { return &Softmax2d{} }

// Forward applies channel-wise softmax over (N, C, H, W).
func (s *Softmax2d) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softmax(1) }

// ---- Learnable / structural activations -------------------------------------

// PReLU is the parametric ReLU: y = max(0, x) + weight * min(0, x).
// NumParams is either 1 (shared slope across all channels) or num_channels
// (per-channel slope, where the channel axis is treated as axis 1 of x).
// The weight is initialized to 0.25 (PyTorch default).
type PReLU struct {
	Base
	NumParams int
	Weight    *tensor.Tensor // shape (NumParams,)
}

// NewPReLU creates a PReLU module. Pass numParams = 1 for a shared slope, or
// numParams = num_channels for per-channel slopes.
func NewPReLU(numParams int) *PReLU {
	if numParams < 1 {
		panic("NewPReLU: numParams must be >= 1")
	}
	p := &PReLU{NumParams: numParams}
	p.Weight = p.reg("weight", tensor.Full(0.25, numParams).SetRequiresGrad(true))
	return p
}

// Forward computes y = max(0, x) + w * min(0, x).
// For NumParams == 1 the scalar weight broadcasts to all of x. For
// NumParams > 1 the weight is broadcast along axis 1 (the channel axis).
func (p *PReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	pos := x.ReLU()
	neg := x.Sub(pos) // negative part (or zero where x >= 0)
	var w *tensor.Tensor
	if p.NumParams == 1 {
		w = p.Weight
	} else {
		if len(x.Shape) < 2 {
			panic("PReLU.Forward: per-channel weight requires x with rank >= 2")
		}
		if x.Shape[1] != p.NumParams {
			panic("PReLU.Forward: channel dim does not match NumParams")
		}
		shape := make([]int, len(x.Shape))
		for i := range shape {
			shape[i] = 1
		}
		shape[1] = p.NumParams
		w = p.Weight.Reshape(shape...)
	}
	return pos.Add(w.Mul(neg))
}

// RReLULayer is the randomized leaky ReLU module, mirroring torch.nn.RReLU:
// y = x for x >= 0, y = slope*x for x < 0.
//
// Training mode: one slope is sampled uniformly from [Lower, Upper] per
// forward call (math/rand) and applied to the WHOLE tensor via LeakyReLU.
// DEVIATION from PyTorch: torch.nn.RReLU samples an independent slope per
// element; sampling a single per-forward slope keeps the op inside the
// framework's single fwd/bwd-closure unary shape. The slope distribution and
// its expectation match PyTorch's; only the per-element independence differs.
//
// Eval mode: the deterministic midpoint slope (Lower+Upper)/2 via
// tensor.RReLU, exactly PyTorch's eval-mode semantics.
type RReLULayer struct {
	Base
	Lower, Upper float64
	rng          *rand.Rand
}

// RReLU constructs a randomized leaky ReLU module sampling negative-region
// slopes from U(lower, upper). PyTorch defaults are lower=1/8, upper=1/3.
// The internal RNG is seeded from math/rand's global source; use Seed for
// deterministic runs.
func RReLU(lower, upper float64) *RReLULayer {
	if lower > upper {
		panic(fmt.Sprintf("nn: RReLU: lower (%v) must be <= upper (%v)", lower, upper))
	}
	return &RReLULayer{Lower: lower, Upper: upper, rng: rand.New(rand.NewSource(rand.Int63()))}
}

// Seed reseeds the module's slope RNG (for deterministic training or tests).
func (r *RReLULayer) Seed(seed int64) { r.rng = rand.New(rand.NewSource(seed)) }

// Forward applies the randomized (training) or midpoint (eval) leaky ReLU.
func (r *RReLULayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	if r.Training() {
		slope := r.Lower + r.rng.Float64()*(r.Upper-r.Lower)
		return x.LeakyReLU(slope)
	}
	return x.RReLU(r.Lower, r.Upper, nil)
}

// GLU is the gated linear unit: split input into halves a, b along Dim and
// return a * sigmoid(b). The size of x along Dim must be even.
type GLU struct {
	Base
	Dim int
}

// NewGLU constructs a GLU splitting along dim (negative counts from the end).
func NewGLU(dim int) *GLU { return &GLU{Dim: dim} }

// Forward implements GLU(x) = a * sigmoid(b).
func (g *GLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	dim := g.Dim
	if dim < 0 {
		dim += len(x.Shape)
	}
	if dim < 0 || dim >= len(x.Shape) {
		panic(fmt.Sprintf("GLU: dim %d out of range for shape %v", g.Dim, x.Shape))
	}
	if x.Shape[dim]%2 != 0 {
		panic(fmt.Sprintf("GLU: size along dim %d must be even, got %d", dim, x.Shape[dim]))
	}
	halves := x.Chunk(dim, 2)
	return halves[0].Mul(halves[1].Sigmoid())
}
