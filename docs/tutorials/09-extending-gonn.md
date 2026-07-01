# 09 — Extending GoNN

[← GPU Acceleration](08-gpu-acceleration.md) | [Index](README.md)

GoNN is built around three extension seams: the unary-op registry (new
activations), `tensor.MakeNode` (arbitrary custom autograd ops), and
`nn.Base` (custom modules). This tutorial walks all three, then sketches the
path for adding a GPU kernel.

## 1. Registering a custom activation

One `UnaryOpDef` gives your op the fluent ecosystem: autograd, name lookup,
and use inside `nn.ActivationByName`:

```go
package main

import (
	"fmt"
	"math"

	"gonn/nn"
	"gonn/tensor"
)

func init() {
	// Softsign-squared: f(x) = x·|x| / (1+x²), just as a demo.
	tensor.RegisterUnary(tensor.UnaryOpDef{
		Name: "sqsign",
		Kind: tensor.UnaryNone, // no GPU kernel; always runs the Go closure
		Fwd: func(x float64) float64 { return x * math.Abs(x) / (1 + x*x) },
		Bwd: func(g, x, y float64) float64 {
			// d/dx of x|x|/(1+x²): (2|x|(1+x²) − x|x|·2x) / (1+x²)²
			num := 2*math.Abs(x)*(1+x*x) - x*math.Abs(x)*2*x
			return g * num / ((1 + x*x) * (1 + x*x))
		},
	})
}

func main() {
	x := tensor.Randn(3, 4).SetRequiresGrad(true)
	y := x.Unary("sqsign") // full autograd support
	y.Sum().Backward()
	fmt.Println(x.Grad)

	// And as a layer:
	model := nn.NewSequential(
		nn.NewLinear(4, 4, true),
		nn.ActivationByName("sqsign"),
	)
	_ = model
}
```

Rules: `Fwd` and `Bwd` are per-element pure functions (`Bwd` receives the
upstream gradient `g`, the input `x`, and the forward output `y` — use
whichever makes the math cheapest). Registration panics on duplicate names.
Always verify with a finite-difference gradcheck (tutorial 02 §6).

## 2. Custom autograd ops with MakeNode

For ops that don't decompose into existing tensor ops — external kernels,
lookup tables, whole fused blocks — compute the forward however you like and
attach a backward closure:

```go
// SquaredDistanceMatrix: D[i,j] = ||a_i - b_j||² computed with plain loops,
// made differentiable w.r.t. both inputs via MakeNode.
func SquaredDistanceMatrix(a, b *tensor.Tensor) *tensor.Tensor {
	n, m, d := a.Shape[0], b.Shape[0], a.Shape[1]
	out := tensor.Zeros(n, m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			var s float64
			for k := 0; k < d; k++ {
				diff := a.Data[i*d+k] - b.Data[j*d+k]
				s += diff * diff
			}
			out.Data[i*m+j] = s
		}
	}

	tensor.MakeNode(out, "SqDist", []*tensor.Tensor{a, b},
		func(grad *tensor.Tensor) []*tensor.Tensor {
			ga := tensor.Zeros(a.Shape...)
			gb := tensor.Zeros(b.Shape...)
			for i := 0; i < n; i++ {
				for j := 0; j < m; j++ {
					g := grad.Data[i*m+j]
					for k := 0; k < d; k++ {
						diff := a.Data[i*d+k] - b.Data[j*d+k]
						ga.Data[i*d+k] += g * 2 * diff
						gb.Data[j*d+k] -= g * 2 * diff
					}
				}
			}
			return []*tensor.Tensor{ga, gb}
		})
	return out
}
```

`MakeNode` records the inputs and your closure; if no input requires
gradients it does nothing (zero overhead at inference). This is exactly how
the fused CUDA flash-attention integrates — its forward and backward are
single kernel launches wrapped in a `MakeNode` (see `nn/flashattn_cuda.go`
for the production template).

## 3. A custom module with its own parameters

`RegisterParam` / `RegisterBuffer` / `RegisterChild` wire your state into
`Parameters()`, `NamedParameters()`, `Buffers()`, and `Train()/Eval()`:

```go
// FiLM: feature-wise affine modulation, y = x·gamma + beta,
// with a running usage counter as a (non-trainable) buffer.
type FiLM struct {
	nn.Base
	Gamma, Beta *tensor.Tensor
	Calls       *tensor.Tensor
}

func NewFiLM(dim int) *FiLM {
	f := &FiLM{}
	f.Gamma = f.RegisterParam("gamma", tensor.Ones(dim).SetRequiresGrad(true))
	f.Beta = f.RegisterParam("beta", tensor.Zeros(dim).SetRequiresGrad(true))
	f.Calls = f.RegisterBuffer("calls", tensor.Zeros(1))
	return f
}

func (f *FiLM) Forward(x *tensor.Tensor) *tensor.Tensor {
	f.Calls.Data[0]++
	return x.Mul(f.Gamma).Add(f.Beta)
}
```

It now composes like any built-in:

```go
model := nn.NewSequential(nn.NewLinear(16, 32, true), NewFiLM(32), nn.GELU())
opt := optim.NewAdam(model.Parameters(), 1e-3) // gamma/beta included
for _, p := range model.NamedParameters() {
	fmt.Println(p.Name) // ..., "1.gamma", "1.beta", ...
}
```

Checklist for custom modules:

1. Embed `nn.Base`; use pointer receivers.
2. Register every grad-requiring tensor (`RegisterParam`) and every piece of
   persistent non-trainable state (`RegisterBuffer`) in the constructor —
   registration order defines `Parameters()` order.
3. Register sub-modules with `RegisterChild` so their parameters and
   train/eval mode propagate.
4. Build `Forward` from differentiable tensor ops (or `MakeNode`) only.
5. Gradcheck it (`nn/gradcheck_test.go` has the harness pattern).

## 4. Adding a GPU kernel

The elementwise kernel path is enum-dispatched end to end, so a new
accelerated op touches four small places:

1. **`backend/backend.go`** — append a constant to `UnaryKind` (append-only:
   the values are a C ABI).
2. **`backend/cuda/gonn_cuda.h`** — mirror the enum value.
3. **`backend/cuda/gonn_cuda.cu`** — one `DEF_UNOP(name, expr)` line plus a
   `case` in `launch_unary`.
4. **Your `tensor.UnaryOpDef`** — set `Kind` to the new constant instead of
   `tensor.UnaryNone`.

Dispatch, thresholds, CPU fallback, and autograd all come for free — the
registry runs your Go `Fwd` closure whenever the backend declines (small
tensors, CPU builds, kernel errors), and always uses your Go `Bwd` for the
backward.

Verify on real hardware with the Docker pipeline (tutorial 08 §2); it runs
GPU-vs-CPU parity for every registered kind at 1e-12 and executes the tensor
test suite with dispatch forced on.

## 5. Where to go next

- The `benchmark/` directory shows how to measure anything you add
  (CUDA-event timing on device, medians over iterations, JSON output).
- `tensor/registry_test.go`, `nn/module_test.go`, and `optim/groups_test.go`
  are the test patterns to copy for each extension seam.
- Open questions and deliberately deferred designs (tensor views,
  device-resident tensor storage) are documented in the README's Status
  section — contributions welcome.

---

[← GPU Acceleration](08-gpu-acceleration.md) | [Index](README.md)
