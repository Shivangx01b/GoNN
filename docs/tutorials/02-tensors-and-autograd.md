# 02 — Tensors & Autograd

[← Getting Started](01-getting-started.md) | [Index](README.md) | [Next: Building Models →](03-building-models.md)

The `tensor` package is GoNN's foundation. This tutorial covers construction,
the op families, broadcasting, batched matmul, dtypes, and how the autograd
engine actually works.

## 1. Construction

```go
tensor.New([]float64{1, 2, 3, 4}, 2, 2) // wrap data with a shape
tensor.Zeros(3, 4)                      // zeros
tensor.Ones(3, 4)                       // ones
tensor.Full(0.5, 3, 4)                  // constant fill
tensor.Randn(3, 4)                      // N(0, 1) samples
tensor.Uniform(-1, 1, 3, 4)             // U(low, high)
tensor.Arange(0, 10, 2)                 // [0 2 4 6 8]
tensor.Eye(3)                           // identity matrix
tensor.Scalar(3.14)                     // 0-d tensor
```

Useful accessors: `t.Shape`, `t.Numel()`, `t.Dim()`, `t.Item()` (scalar
tensors only), `t.Copy()`.

## 2. The op families

**Arithmetic (with NumPy-style broadcasting):**

```go
a.Add(b)  a.Sub(b)  a.Mul(b)  a.Div(b)  a.Neg()
a.AddScalar(2)  a.MulScalar(0.5) // …and Sub/Div variants
```

Shapes broadcast from the right: `(2, 3) + (3,)` works, `(2, 3) + (2, 1)`
works, and gradients are automatically sum-reduced back through the broadcast.

**Unary math:** `Exp, Log, Sqrt, Sin, Cos, Tan, Abs, Reciprocal, Pow(p),
Square, Clip(lo, hi)`.

**Activations:** `ReLU, LeakyReLU(α), ELU(α), SELU, CELU(α), Sigmoid, Tanh,
LogSigmoid, HardTanh, HardSigmoid, Softplus, Softsign, GELU, SiLU, HardSwish,
Mish, ReLU6, Hardshrink(λ), Softshrink(λ), Tanhshrink, Threshold(t, v),
Softmax(axis), LogSoftmax(axis)`.

Fixed activations are also reachable **by name** through the unary-op
registry — the same definitions that power the fluent methods and GPU
dispatch:

```go
y := x.Unary("gelu")            // == x.GELU()
fmt.Println(tensor.UnaryOpNames()) // every registered op, sorted
```

**Reductions:** `Sum, Mean, Max, Min` (to scalars) and the axis forms
`SumAxis(axis, keepDim)`, `MeanAxis`, `MaxAxis`, `MinAxis`, plus
`ArgMax(axis)` / `ArgMin(axis)`. Negative axes count from the end.

**Shape:** `Reshape(-1 allowed)`, `View`, `Flatten`, `Transpose` (swaps the
last two dims of any rank ≥ 2 tensor), `T`, `Permute`, `Squeeze`,
`Unsqueeze`, `Expand`, and the package functions `tensor.Concat(axis, ...)`
and `tensor.Stack(axis, ...)`.

**Indexing:** `Gather(axis, index)`, `IndexSelect(axis, index)`,
`Split(axis, size)`, `Chunk(axis, n)` — all differentiable with scatter-add
backwards. `Tril/Triu`, `Where`, `MaskedFill`, `Cumsum`, `Flip`, `Repeat`
round out the utility set.

## 3. Matrix multiplication — 2D and batched

`MatMul` is a subset of `torch.matmul`:

```go
// Plain 2D GEMM.
A := tensor.Randn(64, 128)
B := tensor.Randn(128, 32)
C := A.MatMul(B) // (64, 32)

// Batched: rank >= 3 multiplies the last two dims; leading dims broadcast.
Q := tensor.Randn(8, 12, 64, 32)  // (batch, heads, seq, dim)
K := tensor.Randn(8, 12, 64, 32)
S := Q.MatMul(K.Transpose())      // (8, 12, 64, 64) — one strided-batched GEMM

// A 2D right operand broadcasts across the batch (classic "apply weight"):
X := tensor.Randn(16, 10, 4)
W := tensor.Randn(4, 8)
Y := X.MatMul(W) // (16, 10, 8)

// Strict form when you want shape errors instead of broadcasting:
Z := tensor.Randn(5, 3, 4).BMM(tensor.Randn(5, 4, 2)) // (5, 3, 2)
```

Both forward *and* backward run through the active compute backend, so on a
CUDA build these are cuBLAS calls (tutorial 08).

## 4. Dtypes: Float64, Float32, Float16

Storage is always `[]float64`; `Dtype` controls the *numeric precision and
range* — results are rounded to the dtype's representable set at op
boundaries, giving you correct IEEE-754 float32/float16 semantics (overflow
to Inf, subnormals, the works) for testing mixed-precision numerics:

```go
h := tensor.NewTyped([]float64{1.0001, 65504, 70000}, tensor.Float16, 3)
fmt.Println(h) // [1.0, 65504 (max half), +Inf] — real binary16 behavior

x32 := x.AsType(tensor.Float32) // differentiable cast (straight-through grad)
```

Type promotion follows NumPy: `float16 + float32 → float32`, etc. For *true*
fp16 storage and tensor-core compute, use the GPU device-buffer path in
tutorial 08.

## 5. How autograd works

Every op that produces a tensor from grad-requiring inputs attaches a
`creator` node recording (a) its inputs and (b) a backward closure.
`Backward()` on a scalar:

1. Topologically sorts the graph (iteratively — a 100k-op chain is fine),
2. seeds the output gradient with 1,
3. walks the ops in reverse, calling each backward closure and accumulating
   into `input.Grad` (sum-reducing through any broadcasts).

Practical consequences:

```go
w := tensor.Randn(3, 3).SetRequiresGrad(true)

// Gradients ACCUMULATE — zero them between steps:
loss1 := w.Square().Sum(); loss1.Backward()
loss2 := w.Square().Sum(); loss2.Backward()
// w.Grad now holds the SUM of both backward passes.
w.ZeroGrad() // reset

// Backward requires a scalar — reduce first:
y := w.Mul(w)
y.Sum().Backward() // ok; y.Backward() would panic
```

Intermediate (non-leaf) tensors participate in the graph automatically; you
only mark the leaves you want gradients *for*.

## 6. Numerical gradient checking

When you build something new on top of tensor ops, verify it the way GoNN's
own test suite does — central finite differences:

```go
func numericalGrad(f func() *tensor.Tensor, p *tensor.Tensor, i int) float64 {
	const eps = 1e-6
	orig := p.Data[i]
	p.Data[i] = orig + eps
	fp := f().Item()
	p.Data[i] = orig - eps
	fm := f().Item()
	p.Data[i] = orig
	return (fp - fm) / (2 * eps)
}
```

Compare against `p.Grad.Data[i]` after an analytic `Backward()`; in float64
they should agree to ~1e-6 relative. See `nn/gradcheck_test.go` for the
production version with seeded sampling.

---

[← Getting Started](01-getting-started.md) | [Index](README.md) | [Next: Building Models →](03-building-models.md)
