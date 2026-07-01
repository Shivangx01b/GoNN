# 03 — Building Models

[← Tensors & Autograd](02-tensors-and-autograd.md) | [Index](README.md) | [Next: Training →](04-training.md)

The `nn` package turns raw tensors into composable layers. This tutorial
covers the module system: what a `Module` is, how parameters are tracked,
train/eval mode, and the layer catalogue.

## 1. Module and Sequential

A single-input layer satisfies:

```go
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	// plus (via the embedded Base):
	// Parameters() []*tensor.Tensor
	// NamedParameters() []nn.Param
	// Buffers() []nn.Param
	// SetTraining(bool)
}
```

`Sequential` chains modules and aggregates their parameters:

```go
model := nn.NewSequential(
	nn.NewLinear(784, 256, true), // in, out, bias
	nn.ReLU(),
	nn.NewDropout(0.2),
	nn.NewLinear(256, 10, true),
)

logits := model.Forward(x)                    // (N, 10)
fmt.Println(len(model.Parameters()))          // 4: two weights + two biases
```

## 2. Train / eval mode

Mode-dependent layers (`Dropout`, `BatchNorm*`) read their mode from the
module tree. `Train()` and `Eval()` propagate recursively through every
container — set it once at the top:

```go
model.Train() // dropout active, batchnorm uses batch stats (default state)
// ... training loop ...

model.Eval()  // dropout = identity, batchnorm uses running stats
pred := model.Forward(xTest)
```

## 3. Named parameters and filtering

Every parameter has a hierarchical dotted name — the hook for per-group
optimizer settings (tutorial 04) and future serialization:

```go
for _, p := range model.NamedParameters() {
	fmt.Println(p.Name, p.T.Shape) // "0.weight" [256 784], "0.bias" [256], "3.weight" ...
}

// Everything except biases (e.g. to exempt biases from weight decay):
weights := nn.FilterParams(model, func(name string) bool {
	return strings.HasSuffix(name, ".weight")
})
```

Non-trainable state (BatchNorm running mean/var) lives in `Buffers()`, not
`Parameters()` — it is never handed to an optimizer.

## 4. The layer catalogue

**Core:** `Linear(in, out, bias)`, `Embedding(vocab, dim)` (IndexSelect
lookup, scatter-add gradient), `Bilinear(in1, in2, out, bias)`, `Identity`,
`Flatten(start, end)`, `Unflatten(dim, sizes...)`.

**Activations** — one generic module, many constructors:

```go
nn.ReLU()  nn.GELU()  nn.SiLU()  nn.Tanh()  nn.Sigmoid()  nn.Mish()
nn.LeakyReLU(0.01)  nn.ELU(1.0)  nn.Threshold(0, -1)  // parameterized
nn.NewSoftmax(1)  nn.NewLogSoftmax(-1)                 // axis reductions
nn.NewPReLU(channels) // learnable slope (a real parameter)
nn.NewGLU(-1)         // gated linear unit, splits the given dim
nn.ActivationByName("hardswish") // anything in tensor.UnaryOpNames()
```

**Normalization** — shared defaults, overridable with options:

```go
nn.NewLayerNorm(512)                                // eps 1e-5
nn.NewLayerNorm(512, nn.WithEps(1e-6))
nn.NewRMSNorm(512)                                  // eps 1e-6
nn.NewBatchNorm1d(64, nn.WithMomentum(0.05))        // running-stat momentum
nn.NewBatchNorm2d(32)                               // (N, C, H, W)
nn.NewGroupNorm(8, 64)                              // groups, channels
nn.NewInstanceNorm2d(32, nn.WithAffine(true))       // affine off by default
```

**Convolution & pooling** — covered in depth in tutorial 05.

**Recurrent & attention** — covered in tutorial 06.

**Dropout:** `nn.NewDropout(p)` — mask-and-scale during training, identity in
eval mode.

## 5. Writing your own module

Embed `nn.Base`, register parameters and children in the constructor, and
implement `Forward`. Registration is what makes `Parameters()`,
`NamedParameters()`, and `Train()/Eval()` work — register in the order you
want parameters returned:

```go
// A residual MLP block: x + W2·act(W1·x)
type ResidualBlock struct {
	nn.Base
	FC1, FC2 *nn.Linear
	Act      *nn.Activation
}

func NewResidualBlock(dim, hidden int) *ResidualBlock {
	b := &ResidualBlock{
		FC1: nn.NewLinear(dim, hidden, true),
		FC2: nn.NewLinear(hidden, dim, true),
		Act: nn.GELU(),
	}
	// RegisterChild wires parameters + train/eval propagation. The names
	// become prefixes in NamedParameters (e.g. "fc1.weight").
	b.RegisterChild("fc1", b.FC1)
	b.RegisterChild("fc2", b.FC2)
	return b
}

func (b *ResidualBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Add(b.FC2.Forward(b.Act.Forward(b.FC1.Forward(x))))
}
```

Direct tensor parameters use `RegisterParam` / `RegisterBuffer` — tutorial 09
builds a module with its own weights from scratch.

Multi-input modules (e.g. `MultiHeadAttention.Forward(q, k, v, causal)`)
follow the same pattern but define their own `Forward` signature — they
satisfy the parameter/mode surface (`nn.Child`) without pretending to be
single-input.

## 6. Weight initialization

Constructors use sensible defaults (Kaiming-uniform for Linear/conv). To
re-initialize, the `nn` init helpers operate in place on any tensor:

```go
nn.XavierUniform(layer.Weight, fanIn, fanOut)
nn.KaimingNormal(layer.Weight, fanIn)
nn.Orthogonal(layer.Weight, 1.0)
nn.Constant(layer.Bias, 0)
```

---

[← Tensors & Autograd](02-tensors-and-autograd.md) | [Index](README.md) | [Next: Training →](04-training.md)
