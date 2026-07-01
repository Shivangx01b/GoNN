# GoNN Tutorials

A step-by-step walkthrough of the entire framework, from your first tensor to
custom CUDA kernels. Every code block targets the current API and the larger
programs are runnable as-is (`go run`) from the repository root.

## Learning path

| # | Tutorial | What you'll learn |
|---|----------|-------------------|
| 01 | [Getting Started](01-getting-started.md) | Install, first tensor, first gradient, first trained model — in 10 minutes |
| 02 | [Tensors & Autograd](02-tensors-and-autograd.md) | The tensor API end to end: construction, math, broadcasting, batched matmul, dtypes, how reverse-mode autograd works |
| 03 | [Building Models](03-building-models.md) | The `nn.Module` system: layers, `Sequential`, parameter registration, `Train()`/`Eval()`, named parameters |
| 04 | [Training](04-training.md) | Losses, optimizers, parameter groups, gradient clipping, LR schedulers, `DataLoader` — the full training loop |
| 05 | [Convolutional Networks](05-convolutional-networks.md) | Conv/pool/norm layers with functional options (stride, padding, dilation); a CNN classifier from scratch |
| 06 | [Sequence Models](06-sequence-models.md) | RNN/LSTM/GRU (multi-layer, bidirectional), Seq2Seq, multi-head attention, transformers |
| 07 | [Classical ML](07-classical-ml.md) | The `ml` package: regression, trees & ensembles, clustering, PCA, preprocessing, model selection |
| 08 | [GPU Acceleration](08-gpu-acceleration.md) | CPU vs CUDA vs OpenCL: build tags, the Docker workflow, dispatch policy, device-resident buffers, fused flash-attention |
| 09 | [Extending GoNN](09-extending-gonn.md) | Register custom activations, write custom autograd ops with `MakeNode`, build your own modules, add a CUDA kernel |

## How to read these

- **New to GoNN?** Read 01 → 04 in order; that covers the core workflow.
- **Coming from PyTorch?** Skim 01, then jump to 03 and 04 — the API is
  deliberately PyTorch-shaped (`Forward`/`Backward`/`Step`,
  `model.Train()`/`Eval()`, parameter groups, schedulers).
- **Here for the GPU?** 08 is self-contained; it assumes only tutorial 01.

## Prerequisites

- Go 1.21+ (`go version`)
- For the GPU tutorial: an NVIDIA GPU plus either the CUDA toolkit or Docker
  (the repo ships a ready-made image definition — no local toolkit needed).

## Runnable companions

The [`examples/`](../../examples) directory contains complete programs that
mirror these tutorials:

| Example | Tutorial |
|---------|----------|
| [`examples/regression`](../../examples/regression) | 01, 02 |
| [`examples/mlp`](../../examples/mlp) | 03, 04 |
| [`examples/cnn`](../../examples/cnn) | 05 |
| [`examples/transformer`](../../examples/transformer) | 06 |
| [`examples/ml_classical`](../../examples/ml_classical) | 07 |
| [`benchmark/`](../../benchmark) | 08 |
