<h1 align="center">
  <br>
  <a href=""><img src="https://github.com/Shivangx01b/GoNN/blob/main/static/logo.png" alt="" width="200px;"></a>
  <br>
  <img src="https://img.shields.io/github/languages/top/Shivangx01b/GoNN?style=flat-square">
  <a href="https://goreportcard.com/report/github.com/Shivangx01b/GoNN"><img src="https://goreportcard.com/badge/github.com/Shivangx01b/GoNN"></a>
  <a href="https://twitter.com/intent/follow?screen_name=shivangx01b"><img src="https://img.shields.io/twitter/follow/shivangx01b?style=flat-square"></a>
</h1>

# GoNN

A pure-Go deep learning and machine learning framework with PyTorch-style autograd, neural-network layers, optimizers, classical ML algorithms, and an optional CUDA backend.

## What is GoNN?

GoNN is a single-binary, dependency-light alternative to PyTorch / tinygrad / TensorFlow, written in Go. It provides:

- **Autograd Tensor** — flat-buffer `*tensor.Tensor` with shape + strides + automatic differentiation by reverse-mode graph traversal (iterative topological sort — arbitrarily deep graphs are safe). `MatMul` handles rank ≥ 3 batched matmul with NumPy-style broadcast of the leading dims (plus strict `BMM`), lowered to one strided-batched GEMM on the active backend.
- **Unary-op registry** — every elementwise activation/math op is defined exactly once (`tensor.UnaryOpDef`: forward closure + backward closure + GPU kernel kind); the fluent methods (`x.GELU()`), name-based lookup (`x.Unary("gelu")`, `nn.ActivationByName`), and GPU dispatch all derive from the same definition.
- **`nn` package** — modules embed a registration-based `Base`: `Parameters()`, dotted `NamedParameters()` (for optimizer param groups), `Buffers()`, and recursive `Train()`/`Eval()` come for free and propagate through containers.
  - **Linear & conv:** `Linear`, `Conv1d/2d/3d`, `ConvTranspose1d/2d/3d` — one shared per-channel gather + GEMM core with functional options (`WithStride`, `WithPad`, `WithDilation`, `WithKernel`, `WithNoBias`) and a cached gather matrix; `Embedding` (IndexSelect lookup, scatter-add backward), `Bilinear`.
  - **Pooling:** `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, `AdaptiveMaxPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d` — all on the same gather core.
  - **Normalization:** `BatchNorm1d/2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm1d/2d` — one `normalizeAxes` core, options `WithEps`/`WithMomentum`/`WithAffine`; running stats are registered buffers.
  - **Padding & upsample:** `ZeroPad2d`, `ConstantPad2d`, `ReflectionPad2d`, `ReplicationPad2d`, `Upsample`, `PixelShuffle`/`PixelUnshuffle`.
  - **Recurrent:** `RNN`, `LSTM`, `GRU` (multi-layer + bidirectional via `WithLayers(n)`, `WithBidirectional()`; default = 1 layer), `RNNCell`, `LSTMCell`, `GRUCell`, `Seq2Seq`.
  - **Attention/Transformer:** `MultiHeadAttention` (optional causal mask; batched-matmul core), `TransformerEncoderLayer`/`TransformerEncoder`, `TransformerDecoderLayer`/`TransformerDecoder`.
  - **Containers:** `Sequential`, `Dropout` (train/eval aware).
  - **Activations:** one generic `Activation` module with named constructors — `nn.ReLU()`, `nn.GELU()`, `nn.LeakyReLU(0.01)`, … — plus learnable `PReLU` and gated `GLU`.
- **`optim` package** — SGD (momentum/Nesterov), Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam, Adamax, RAdam, LBFGS (closure-style), Rprop, Adafactor, ASGD, LAMB, Lion, SparseAdam. All share a common base with **parameter groups** (per-group LR/weight-decay via `NewXXXGroups`), plus **gradient clipping** (`ClipGradNorm`, `ClipGradValue`) and group-aware LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, PolynomialLR, ChainedScheduler, SequentialLR, CyclicLR, ReduceLROnPlateau, OneCycleLR).
- **`ml` package** — classical algorithms:
  - **Linear:** LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LogisticRegression.
  - **Discriminant:** LinearDiscriminantAnalysis (LDA, with Fisher transform), QuadraticDiscriminantAnalysis (QDA).
  - **Trees & ensembles:** DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, ExtraTreesClassifier/Regressor, AdaBoostClassifier, GradientBoostingClassifier/Regressor, IsolationForest.
  - **Neighbors:** KNNClassifier, KNNRegressor.
  - **SVM:** LinearSVC.
  - **Naive Bayes:** GaussianNB, MultinomialNB, BernoulliNB.
  - **Clustering:** KMeans, DBSCAN, AgglomerativeClustering, MeanShift, GaussianMixture.
  - **Dim. reduction:** PCA, KernelPCA, FastICA, TSNE.
  - **Preprocessing:** StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures.
  - **Metrics + model selection:** Accuracy, Precision/Recall/F1, ConfusionMatrix, MSE/MAE/R²/SilhouetteScore/ROCAUC, TrainTestSplit, KFold, CrossValScore.
- **`data` package** — `Dataset`, `DataLoader`, transforms, MNIST/CSV loaders, synthetic dataset generators (`MakeRegression`, `MakeClassification`, `MakeBlobs`, `MakeMoons`).
- **`backend` package** — pluggable compute backend with a deliberately tiny contract: `Backend{Gemm (batched, trans flags), Synchronize}` plus an optional `Elementwiser` capability (enum-dispatched unary/binary kernels; a backend that declines falls back to the pure-Go path). Backend selection (`backend.Use`/`Current`) is goroutine-safe (atomic). Three implementations: **CPU** (gonum BLAS; the tensor package's Go closures are its elementwise path), **CUDA** (`-tags cuda`, CGO + cuBLAS strided-batched GEMM + macro-generated kernels with a grow-only device workspace cache; verified on an RTX 3060 and benchmarked vs PyTorch/TensorFlow/tinygrad — see [Benchmarks](#benchmarks)), and **OpenCL** (`-tags opencl`, CGO + fp64 kernels; numerically verified against the CPU backend).
- **Tensor dtypes** — `Float64` (default), `Float32`, `Float16` with correct IEEE-754 precision/range semantics (`x.To(tensor.Float16)`, numpy-style type promotion). Emulated on float64 storage for correct mixed-precision *numerics*; for real fp16 *storage + tensor-core compute*, use the GPU `DeviceBufferF16` path (fp16 GEMM at ~24 TFLOP/s on a 3060).
- **Fused CUDA flash-attention (forward + backward)** — a custom fp64 flash-attention kernel (online softmax, no S×S materialization) wired into `nn.MultiHeadAttention`. On **causal** fp64 attention it is **~1.4–1.5× faster than PyTorch's SDPA**, and its fused backward (gradcheck ≈ 5e-8) lets the differentiable `Forward` **train** on the kernel — not just run inference.

Everything compiles to a single static Go binary (no Python runtime).

## Install

```bash
go get github.com/Shivangx01b/GoNN
```

## Quickstart — Tensor + Autograd

```go
package main

import (
    "fmt"
    "gonn/tensor"
)

func main() {
    x := tensor.New([]float64{1, 2, 3}, 3, 1).SetRequiresGrad(true)
    W := tensor.New([]float64{2, -1, 0.5}, 1, 3).SetRequiresGrad(true)

    y := W.MatMul(x).Square().Sum()
    y.Backward()

    fmt.Println("y      =", y)
    fmt.Println("dy/dx  =", x.Grad)  // [6, -3, 1.5]
    fmt.Println("dy/dW  =", W.Grad)  // [3, 6, 9]
}
```

## Quickstart — Linear Regression with SGD

See [`examples/regression`](examples/regression):

```go
W := tensor.Randn(1, 1).SetRequiresGrad(true)
b := tensor.Zeros(1).SetRequiresGrad(true)
opt := optim.NewSGD([]*tensor.Tensor{W, b}, 0.01)

for epoch := 0; epoch < 200; epoch++ {
    opt.ZeroGrad()
    pred := X.MatMul(W).Add(b)
    loss := pred.Sub(Y).Square().Mean()
    loss.Backward()
    opt.Step()
}
```

## Quickstart — MLP Classifier (PyTorch-style)

```go
import (
    "gonn/nn"
    "gonn/optim"
    "gonn/tensor"
)

model := nn.NewSequential(
    nn.NewLinear(784, 256, true),
    nn.ReLU(),
    nn.NewLinear(256, 64, true),
    nn.ReLU(),
    nn.NewLinear(64, 10, true),
)

opt := optim.NewAdam(model.Parameters(), 1e-3)

model.Train() // enable Dropout/BatchNorm training behavior (recursive)
for epoch := 0; epoch < 10; epoch++ {
    for batch := range loader.Iter() {
        opt.ZeroGrad()
        logits := model.Forward(batch.X)
        loss := nn.CrossEntropyLoss(logits, batch.Y)
        loss.Backward()
        optim.ClipGradNorm(opt.Parameters(), 1.0) // optional
        opt.Step()
    }
}
model.Eval() // inference mode
```

## Quickstart — Parameter groups & gradient clipping

Give the head a higher learning rate than the backbone and exempt it from
weight decay; schedulers scale each group relative to its own base LR:

```go
opt := optim.NewAdamWGroups([]optim.Group{
    {Params: backbone.Parameters(), LR: 1e-4, WeightDecay: 0.01},
    {Params: head.Parameters(), LR: 1e-3},
})
sched := optim.NewCosineAnnealingLR(opt, 1000, 0)

loss.Backward()
optim.ClipGradNorm(opt.Parameters(), 1.0) // returns the pre-clip global norm
opt.Step()
sched.Step()
```

`nn.NamedParameters()` + `nn.FilterParams` build the groups by name
(e.g. exclude every `.bias` from decay).

## Quickstart — Classical ML

```go
import "gonn/ml"

// K-means clustering
km := ml.NewKMeans(3, 100, 1e-4)
km.Fit(X)
labels := km.Predict(X)

// Random forest classification
rf := ml.NewRandomForestClassifier(100, 10, 0)
rf.Fit(Xtr, ytr)
yhat := rf.Predict(Xte)

// Gradient boosting regression
gb := ml.NewGradientBoostingRegressor(100, 0.1, 3)
gb.Fit(Xtr, ytr)
```

## Tensor reference

| Category   | Methods |
|------------|---------|
| Construct  | `New`, `Zeros`, `Ones`, `Full`, `Randn`, `Uniform`, `Arange`, `Eye`, `Scalar` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `MatMul` (2D and batched rank ≥ 3 with broadcast batch dims), `BMM`, `Neg`, scalar variants `AddScalar`, … |
| Unary      | `Exp`, `Log`, `Sqrt`, `Sin`, `Cos`, `Tan`, `Abs`, `Reciprocal`, `Pow`, `Square`, `Clip` |
| Reduction  | `Sum`, `Mean`, `Max`, `Min`, `SumAxis`, `MeanAxis`, `MaxAxis`, `MinAxis`, `ArgMax`, `ArgMin` |
| Shape      | `Reshape`, `View`, `Flatten`, `Transpose` (swaps the last two dims, any rank), `T`, `Permute`, `Squeeze`, `Unsqueeze`, `Expand`, `Concat`, `Stack` |
| Activation | `ReLU`, `LeakyReLU`, `ELU`, `SELU`, `CELU`, `Sigmoid`, `Tanh`, `LogSigmoid`, `HardTanh`, `HardSigmoid`, `Softplus`, `Softsign`, `GELU`, `SiLU` (Swish), `HardSwish`, `Mish`, `ReLU6`, `Hardshrink`, `Softshrink`, `Tanhshrink`, `Threshold`, `RReLU`, `Softmax`, `LogSoftmax` — fixed activations are also reachable by name: `x.Unary("gelu")`, `tensor.UnaryOpNames()` |
| Autograd   | `SetRequiresGrad`, `Backward`, `ZeroGrad`, `.Grad`, `MakeNode` (custom-op escape hatch) |

## CUDA backend

The default build is pure-Go CPU. To compile against CUDA:

```bash
# 1. Build the native library
cd backend/cuda
nvcc -O3 -Xcompiler -fPIC -shared gonn_cuda.cu -o libgonn_cuda.so -lcublas

# 2. Build GoNN with the cuda tag
go build -tags cuda ./...
```

The CUDA implementation lives in `backend/cuda/gonn_cuda.cu`. The Go side calls
into it via CGO. The C ABI is three enum-dispatched entry points — `gonn_gemm`
(cuBLAS `DgemmStridedBatched`, trans flags, batch=1 for plain GEMM),
`gonn_unary(kind, …)`, and `gonn_binary(kind, …)` — with macro-generated
kernels, error-code returns, and a grow-only device workspace cache instead of
per-call `cudaMalloc`/`cudaFree`. Adding an accelerated op is one `DEF_UNOP`
line, one switch case, and one enum constant (mirrored in `backend`).

The GPU backend currently accelerates:

| Category    | Ops |
|-------------|-----|
| GEMM        | 2D and strided-batched matmul with trans flags — dispatched from `tensor.MatMul`/`BMM` (forward **and** backward) |
| Elementwise | `add`, `sub`, `mul`, `div` (`backend.BinaryKind`) |
| Activations | `relu`, `sigmoid`, `tanh`, `exp`, `log`, `gelu` (tanh approx.), `silu` (`backend.UnaryKind`) |
| Attention   | Fused fp64 flash-attention forward + backward (`flash_attn_f64_tiled`, online softmax) |

**Elementwise dispatch policy.** Tensor storage is host memory, so a
dispatched elementwise op pays a PCIe round trip. `tensor.SetDispatchPolicy`
controls when ops route to the GPU: transcendental unaries dispatch above
`UnaryMinElems` (default `1<<16`, tuned from the measured break-even on an
RTX 3060 — tanh at 64K elements is 3.2× faster dispatched), while
bandwidth-bound binaries are **disabled by default** (`BinaryMinElems =
MaxInt`) because the copy alone costs more than the compute. GEMMs always
dispatch. The device-resident benchmark path (`cuda.DeviceBuffer`,
`DeviceBufferF16`) keeps data on the GPU and remains the apples-to-apples
comparison against other frameworks.

The CUDA backend is verified for correctness against the CPU backend on the
GPU: batched/transposed GEMM and every elementwise kind at `1e-12`, plus
`go test -tags cuda` with dispatch forced on (see
`benchmark/docker/build_and_run.sh`).

### Reproducible GPU build (Docker, no local CUDA toolkit needed)

```bash
docker build -f benchmark/docker/Dockerfile.cuda -t gonn-cuda .
docker run --rm --gpus all -v "$PWD":/work -w /work gonn-cuda \
    bash benchmark/docker/build_and_run.sh
```

This compiles `gonn_cuda.cu` with `nvcc`, builds GoNN `-tags cuda`, verifies
correctness on the GPU, then runs the matmul, elementwise, flash-attention, and
`MultiHeadAttention.ForwardFused` benchmarks.

### OpenCL backend (`-tags opencl`)

The OpenCL backend (`backend/opencl`, fp64 kernels mirroring the CUDA ones) is
numerically verified against the CPU backend:

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work gonn-cuda \
    bash benchmark/docker/opencl_run.sh
```

It runs on **any** OpenCL device. The verification uses the portable `oclgrind`
fp64 runtime because this machine's Docker/WSL2 GPU passthrough does not inject
NVIDIA's OpenCL driver; the same binary runs on the GPU wherever a real GPU
OpenCL ICD is present (native Linux + NVIDIA driver, or a Windows host).

## Benchmarks

Run live on one machine (12-core CPU, RTX 3060) vs PyTorch 2.7.1+cu128,
TensorFlow 2.20, tinygrad 0.13, with matched CUDA-event timing. Full methodology
and tables: [`benchmark/REPORT.md`](benchmark/REPORT.md) and
[`benchmark/RESULTS.md`](benchmark/RESULTS.md). Honest highlights:

| op (N=2048 / shape) | GoNN | PyTorch | tinygrad | verdict |
|---|---|---|---|---|
| **causal attention f64** (GFLOP/s) | **~87–96** | 58–66 | — | **GoNN wins ~1.4–1.5×** (fused kernel) |
| matmul f32 GPU (GFLOP/s) | **~7,700–8,150** | ~7,600 | 1,104 (OpenCL) | GoNN ≈ PyTorch; ~7× tinygrad |
| matmul f64 GPU (GFLOP/s) | **~170–179** | ~174 | 176 | three-way tie (all cuBLAS) |
| fused attention in `nn.MultiHeadAttention` | trains (gradcheck ≈5e-8) + inference on GPU | — | — | fwd+bwd kernel |
| matmul f64 CPU (GFLOP/s) | 40 (gonum) | **166** (MKL) | — | PyTorch (MKL) |

**Honest summary:** GoNN's GPU matmul is on par with PyTorch (both lean on
cuBLAS), it beats PyTorch ~1.5× on causal fp64 attention via a custom fused
kernel, and its CPU matmul is 17× faster than the old naive loop but still behind
MKL (pure Go has no SIMD intrinsics). GoNN is **not** "faster than PyTorch
everywhere" — see the report for where it wins, ties, and loses.

## Project layout

```
GoNN/
├── tensor/        # Core Tensor + autograd
├── nn/            # Layers, losses, init, activations as modules
├── optim/         # Optimizers + LR schedulers
├── ml/            # Classical ML algorithms
├── data/          # Datasets, DataLoader, transforms
├── backend/       # CPU (gonum BLAS) / CUDA backend contract
│   └── cuda/      # CUDA kernels + fused flash-attention (build tag `cuda`)
├── benchmark/     # Cross-framework benchmarks + Docker GPU build + report
├── examples/      # Runnable demos
└── main.go        # Top-level smoke test
```

## Loss reference

`MSELoss`, `MAELoss` / `L1Loss`, `SmoothL1Loss`, `HuberLoss`, `CrossEntropyLoss`, `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `KLDivLoss`, `PoissonNLLLoss`, `GaussianNLLLoss`, `MarginRankingLoss`, `HingeEmbeddingLoss`, `CosineEmbeddingLoss`, `TripletMarginLoss`, `MultiMarginLoss`.

Every loss accepts a trailing reduction option:
`nn.MSELoss(pred, target, nn.WithReduction(nn.ReduceSum))` — `ReduceMean`
(default), `ReduceSum`, or `ReduceNone` (unreduced tensor).

## Examples

Runnable demos under `examples/`:

- `examples/regression` — linear regression with SGD.
- `examples/mlp` — 3-class MLP classifier with Adam (reaches 100%).
- `examples/cnn` — Conv2d + MaxPool2d + AdaptiveAvgPool2d image classifier.
- `examples/transformer` — small transformer encoder + classification head.
- `examples/ml_classical` — LinearRegression + KMeans + PCA.

## Migration notes (2026-07 refactor)

The framework-wide refactor preserved numerics bit-for-bit (verified by
seeded parity goldens and finite-difference gradchecks) but changed some
call shapes:

- `nn.ReLU{}` → `nn.ReLU()` (all activation modules are now constructors on
  one generic `Activation` type); `nn.Softmax{Axis: 1}` → `nn.NewSoftmax(1)`;
  `nn.GLU{Dim: d}` → `nn.NewGLU(d)`.
- `nn.NewConv2d(in, out, k, stride, pad, bias)` →
  `nn.NewConv2d(in, out, k, nn.WithStride(s), nn.WithPad(p))`
  (+ `nn.WithNoBias()`); `NewConv2dHW(...)` → `nn.WithKernel(kh, kw)` etc.
  Dilation is new: `nn.WithDilation(d)`.
- `nn.NewMaxPool2d(k, s)` → `nn.NewMaxPool2d(k)` (stride defaults to the
  kernel) or `nn.NewMaxPool2d(k, nn.WithPoolStride(s))`.
- `nn.NewMultiLayerLSTM(in, h, n, bidir)` →
  `nn.NewLSTM(in, h, nn.WithLayers(n), nn.WithBidirectional())` — the
  MultiLayer* types were folded into `RNN`/`LSTM`/`GRU`.
- `nn.NewInstanceNorm2d(c, affine)` → `nn.NewInstanceNorm2d(c)` /
  `nn.NewInstanceNorm2d(c, nn.WithAffine(true))`; norm eps/momentum are now
  options (`WithEps`, `WithMomentum`).
- Unchanged: the training loop, `nn.NewLinear(in, out, bias)`, all fluent
  tensor methods, every optimizer constructor and option, `backend.Use`/
  `Current`, `tensor.MakeNode`.
- New: `model.Train()`/`Eval()`, `NamedParameters`/`FilterParams`,
  `optim.Group`/`NewXXXGroups`/`Groups()`, `optim.ClipGradNorm`/
  `ClipGradValue`, batched `MatMul`/`BMM`, `tensor.SetDispatchPolicy`,
  the unary-op registry (`tensor.RegisterUnary`, `x.Unary(name)`).

## Status

The tensor + autograd core, the full NN layer catalogue (linear, conv 1/2/3-d, conv-transpose, pooling/adaptive pooling, normalization, padding, upsample, RNN/LSTM/GRU with cells and bidirectional/multi-layer variants, Seq2Seq, attention, transformer encoder/decoder), all common optimizers and schedulers, and the classical ML catalogue (linear, discriminant, trees, ensembles, boosting, isolation forest, SVM, NB, KNN, clustering, dimensionality reduction, preprocessing, metrics) are implemented and tested. The compute backend is pluggable: CPU (gonum BLAS) by default, CUDA (cuBLAS + custom kernels, incl. a fused fp64 flash-attention that trains) via `-tags cuda`, verified on GPU and benchmarked against PyTorch/TensorFlow/tinygrad.

Intentionally deferred (documented, single-seam designs make them tractable later): stride-based tensor views (`Permute`/`Expand` currently copy), device-resident tensor storage (every autograd backward runs on host slices; `tensor/dispatch.go` is the one place a `Device` field would land), GPU elementwise backward, sparse ops, distributed training, JIT, CTC.
