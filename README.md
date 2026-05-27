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
- **`nn` package** — `Linear`, `Conv2d`, `BatchNorm`, `LayerNorm`, `RMSNorm`, `Dropout`, `Embedding`, `RNN`, `LSTM`, `GRU`, `MultiHeadAttention`, `Transformer{Encoder,Decoder}`, `Sequential`, and the full activation/loss catalogue.
- **`optim` package** — SGD (with momentum / Nesterov), Adam, AdamW, RMSprop, Adagrad, Adadelta, NAdam, plus LR schedulers (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearLR, ReduceLROnPlateau, OneCycleLR).
- **`ml` package** — classical algorithms: LinearRegression, Ridge, LogisticRegression, KMeans, KNN, DecisionTree, RandomForest, GradientBoosting, NaiveBayes (Gaussian / Multinomial / Bernoulli), PCA, LinearSVC, DBSCAN, AgglomerativeClustering, plus preprocessing (StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures) and metrics.
- **`data` package** — `Dataset`, `DataLoader`, transforms, MNIST/CSV loaders, synthetic dataset generators (`MakeRegression`, `MakeClassification`, `MakeBlobs`, `MakeMoons`).
- **`backend` package** — CPU by default. CUDA via `-tags cuda` (CGO bindings to `backend/cuda/gonn_cuda.cu`, cuBLAS-backed matmul).

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
| Activation | `ReLU`, `LeakyReLU`, `ELU`, `SELU`, `Sigmoid`, `Tanh`, `HardTanh`, `HardSigmoid`, `Softplus`, `Softsign`, `GELU`, `SiLU` (Swish), `HardSwish`, `Mish`, `ReLU6`, `Softmax`, `LogSoftmax` |
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

## Project layout

```
GoNN/
├── tensor/        # Core Tensor + autograd
├── nn/            # Layers, losses, init, activations as modules
├── optim/         # Optimizers + LR schedulers
├── ml/            # Classical ML algorithms
├── data/          # Datasets, DataLoader, transforms
├── backend/       # CPU / CUDA backend contract
│   └── cuda/      # CUDA kernels (build tag `cuda`)
├── examples/      # Runnable demos
└── main.go        # Top-level smoke test
```

## Status

Active development. The tensor + autograd core, all major NN layers, all common optimizers, and the classical ML catalogue are implemented and tested. Coverage of more exotic PyTorch / sklearn corners (sparse ops, distributed training, JIT) is intentionally not pursued.
