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

- **Autograd Tensor** — flat-buffer `*tensor.Tensor` with shape + strides + automatic differentiation by reverse-mode graph traversal.
- **`nn` package** —
  - **Linear & conv:** `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d/2d/3d`, `Embedding`.
  - **Pooling:** `MaxPool2d`, `AvgPool2d`, `AdaptiveMaxPool1d/2d/3d`, `AdaptiveAvgPool1d/2d/3d`.
  - **Normalization:** `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm1d/2d`.
  - **Padding & upsample:** `ZeroPad2d`, `ConstantPad2d`, `ReflectionPad2d`, `ReplicationPad2d`, `Upsample`, `PixelShuffle`/`PixelUnshuffle`.
  - **Recurrent:** `RNN`, `LSTM`, `GRU` (single layer), `RNNCell`, `LSTMCell`, `GRUCell`, `MultiLayerRNN`, `MultiLayerLSTM`, `MultiLayerGRU` (multi-layer + bidirectional), `Seq2Seq`.
  - **Attention/Transformer:** `MultiHeadAttention` (optional causal mask), `TransformerEncoderLayer`/`TransformerEncoder`, `TransformerDecoderLayer`/`TransformerDecoder`.
  - **Containers:** `Sequential`, `Dropout`.
  - **Parametric/gated activations as modules:** `PReLU` (learnable slope), `GLU`.
- **`optim` package** — SGD (momentum/Nesterov), Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam, Adamax, RAdam, LBFGS (closure-style), Rprop, plus LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, PolynomialLR, ChainedScheduler, SequentialLR, CyclicLR, ReduceLROnPlateau, OneCycleLR).
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
- **`backend` package** — pluggable compute backend. The CPU backend uses gonum's BLAS for matmul; the CUDA backend (`-tags cuda`, CGO + cuBLAS) is wired into `tensor.MatMul`, so real models run their GEMMs on the GPU. Verified on an RTX 3060 and benchmarked against PyTorch / TensorFlow / tinygrad (see [Benchmarks](#benchmarks)).
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

model := nn.Sequential(
    nn.NewLinear(784, 256, true),
    nn.ReLU{},
    nn.NewLinear(256, 64, true),
    nn.ReLU{},
    nn.NewLinear(64, 10, true),
)

opt := optim.NewAdam(model.Parameters(), 1e-3)

for epoch := 0; epoch < 10; epoch++ {
    for batch := range loader.Iter() {
        opt.ZeroGrad()
        logits := model.Forward(batch.X)
        loss := nn.CrossEntropyLoss(logits, batch.Y)
        loss.Backward()
        opt.Step()
    }
}
```

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
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `Neg`, scalar variants `AddScalar`, … |
| Unary      | `Exp`, `Log`, `Sqrt`, `Sin`, `Cos`, `Tan`, `Abs`, `Reciprocal`, `Pow`, `Square`, `Clip` |
| Reduction  | `Sum`, `Mean`, `Max`, `Min`, `SumAxis`, `MeanAxis`, `MaxAxis`, `MinAxis`, `ArgMax`, `ArgMin` |
| Shape      | `Reshape`, `View`, `Flatten`, `Transpose`, `T`, `Permute`, `Squeeze`, `Unsqueeze`, `Expand`, `Concat`, `Stack` |
| Activation | `ReLU`, `LeakyReLU`, `ELU`, `SELU`, `CELU`, `Sigmoid`, `Tanh`, `LogSigmoid`, `HardTanh`, `HardSigmoid`, `Softplus`, `Softsign`, `GELU`, `SiLU` (Swish), `HardSwish`, `Mish`, `ReLU6`, `Hardshrink`, `Softshrink`, `Tanhshrink`, `Threshold`, `RReLU`, `Softmax`, `LogSoftmax` |
| Autograd   | `SetRequiresGrad`, `Backward`, `ZeroGrad`, `.Grad` |

## CUDA backend

The default build is pure-Go CPU. To compile against CUDA:

```bash
# 1. Build the native library
cd backend/cuda
nvcc -O3 -Xcompiler -fPIC -shared gonn_cuda.cu -o libgonn_cuda.so -lcublas

# 2. Build GoNN with the cuda tag
go build -tags cuda ./...
```

The CUDA implementation lives in `backend/cuda/gonn_cuda.cu`. The Go side calls into it via CGO and uses cuBLAS for matmul. The CPU and CUDA backends share the same `backend.Backend` interface so callers do not need to change.

The GPU backend currently accelerates:

| Category    | Ops |
|-------------|-----|
| MatMul      | `MatMul` (cuBLAS `Dgemm`/`Sgemm`) — dispatched from `tensor.MatMul` |
| Elementwise | `AddElem`, `MulElem`, `Sub`, `Div`, `Scale`, `AxpyInto` (in-place `out += alpha*x`) |
| Reductions  | `Sum`, `Max` (single-block tree reduce in shared memory) |
| Activations | `ReLU`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `GELU` (tanh approx.), `SiLU` (Swish) |
| Attention   | Fused fp64 flash-attention forward (`flash_attn_f64_tiled`, online softmax) |

The CUDA backend is verified for correctness against the CPU backend on the GPU
(matmul `maxAbsDiff ≈ 7e-16`). The tensor-op path copies host↔device per call
(no device buffer caching yet); the device-resident benchmark path keeps inputs
on the GPU and is the apples-to-apples comparison against other frameworks.

### Reproducible GPU build (Docker, no local CUDA toolkit needed)

```bash
docker build -f benchmark/docker/Dockerfile.cuda -t gonn-cuda .
docker run --rm --gpus all -v "$PWD":/work -w /work gonn-cuda \
    bash benchmark/docker/build_and_run.sh
```

This compiles `gonn_cuda.cu` with `nvcc`, builds GoNN `-tags cuda`, verifies
correctness on the GPU, then runs the matmul, elementwise, flash-attention, and
`MultiHeadAttention.ForwardFused` benchmarks.

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

## Examples

Runnable demos under `examples/`:

- `examples/regression` — linear regression with SGD.
- `examples/mlp` — 3-class MLP classifier with Adam (reaches 100%).
- `examples/cnn` — Conv2d + MaxPool2d + AdaptiveAvgPool2d image classifier.
- `examples/transformer` — small transformer encoder + classification head.
- `examples/ml_classical` — LinearRegression + KMeans + PCA.

## Status

The tensor + autograd core, the full NN layer catalogue (linear, conv 1/2/3-d, conv-transpose, pooling/adaptive pooling, normalization, padding, upsample, RNN/LSTM/GRU with cells and bidirectional/multi-layer variants, Seq2Seq, attention, transformer encoder/decoder), all common optimizers and schedulers, and the classical ML catalogue (linear, discriminant, trees, ensembles, boosting, isolation forest, SVM, NB, KNN, clustering, dimensionality reduction, preprocessing, metrics) are implemented and tested. The compute backend is pluggable: CPU (gonum BLAS) by default, CUDA (cuBLAS + custom kernels, incl. a fused fp64 flash-attention) via `-tags cuda`, verified on GPU and benchmarked against PyTorch/TensorFlow/tinygrad. Coverage of more exotic corners (sparse ops, distributed training, JIT, CTC) is intentionally not pursued.

Recent correctness fixes: `Concat`/`Stack` are now autograd-aware (previously dropped gradients), `BCELoss` clamps to avoid `log(0)`, BatchNorm tracks the unbiased running variance (PyTorch parity), and the CUDA backend now compiles (a constant `-1.0/0.0` had blocked it).
