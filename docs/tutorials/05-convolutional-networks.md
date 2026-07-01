# 05 — Convolutional Networks

[← Training](04-training.md) | [Index](README.md) | [Next: Sequence Models →](06-sequence-models.md)

Convolutions, pooling, padding, and spatial normalization — and a CNN image
classifier assembled from them. All spatial layers expect channels-first
layout: `(N, C, spatial...)`.

## 1. Convolutions

`Conv1d/2d/3d` share one constructor shape: `(inChannels, outChannels,
kernel, options...)`. Defaults are stride 1, no padding, dilation 1, bias on.

```go
nn.NewConv2d(3, 16, 3)                                   // 3x3, stride 1, "valid"
nn.NewConv2d(3, 16, 3, nn.WithPad(1))                    // "same" for 3x3
nn.NewConv2d(16, 32, 3, nn.WithStride(2), nn.WithPad(1)) // downsample x2
nn.NewConv2d(3, 8, 3, nn.WithDilation(2))                // dilated (à trous)
nn.NewConv1d(64, 64, 5, nn.WithNoBias())                 // no bias (e.g. before BN)

// Asymmetric geometry: pass one value per spatial dim.
nn.NewConv2d(3, 8, 3, nn.WithKernel(3, 5), nn.WithStride(1, 2), nn.WithPad(1, 2))
```

Output size per dim: `(in + 2·pad − dilation·(kernel−1) − 1)/stride + 1`.

**Transposed convolutions** ("deconvolution", for upsampling paths in
UNets/GANs) use the same options; output size is
`(in−1)·stride − 2·pad + dilation·(kernel−1) + 1`:

```go
up := nn.NewConvTranspose2d(64, 32, 4, nn.WithStride(2), nn.WithPad(1)) // ×2 upsample
```

> Under the hood every conv builds a per-channel gather matrix (cached per
> input shape) and runs a single GEMM — which means conv layers ride the GPU
> GEMM path automatically on CUDA builds (tutorial 08).

## 2. Pooling

Fixed-window pooling defaults its stride to the kernel (non-overlapping):

```go
nn.NewMaxPool2d(2)                          // 2x2, stride 2
nn.NewAvgPool2d(3, nn.WithPoolStride(1))    // overlapping average
nn.NewMaxPool3d(2)                          // (N, C, D, H, W)
```

Adaptive pooling fixes the *output* size instead — the standard trick to make
a classifier head independent of input resolution:

```go
gap := nn.NewAdaptiveAvgPool2d(1, 1) // global average pool -> (N, C, 1, 1)
nn.NewAdaptiveMaxPool1d(8)
```

## 3. Padding & resampling

```go
nn.NewZeroPad2d(1, 1, 1, 1)              // top, bottom, left, right
nn.NewConstantPad2d(1, 1, 1, 1, -1.0)    // pad with a value
nn.NewReflectionPad2d(2, 2, 2, 2)        // mirror without edge repeat
nn.NewReplicationPad2d(1, 1, 1, 1)       // clamp to edge
nn.NewUpsample(2, "nearest")             // or "bilinear"
nn.NewPixelShuffle(2)                    // (N, C·r², H, W) -> (N, C, H·r, W·r)
```

## 4. Spatial normalization

```go
nn.NewBatchNorm2d(32)                        // per channel over (N, H, W); tracks running stats
nn.NewGroupNorm(8, 32)                       // batch-size independent
nn.NewInstanceNorm2d(32, nn.WithAffine(true)) // per (sample, channel) — style transfer
```

Remember `model.Train()` / `model.Eval()`: BatchNorm normalizes with batch
statistics in training mode and with its running estimates at eval time.

## 5. A complete CNN classifier

Synthetic 1×8×8 images, three classes, the whole pipeline (this is
[`examples/cnn`](../../examples/cnn) restructured as a single `Sequential`):

```go
package main

import (
	"fmt"
	"math/rand"

	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

func main() {
	rand.Seed(3)
	X, Y := makeImgs(120, 3) // (120, 1, 8, 8), (120,)

	model := nn.NewSequential(
		nn.NewConv2d(1, 8, 3, nn.WithPad(1)), // (N, 8, 8, 8)
		nn.ReLU(),
		nn.NewMaxPool2d(2),                   // (N, 8, 4, 4)
		nn.NewConv2d(8, 16, 3, nn.WithPad(1)),
		nn.ReLU(),
		nn.NewAdaptiveAvgPool2d(1, 1),        // (N, 16, 1, 1)
		nn.NewFlatten(1, -1),                 // (N, 16)
		nn.NewLinear(16, 3, true),
	)

	opt := optim.NewAdam(model.Parameters(), 5e-3)

	model.Train()
	for epoch := 0; epoch < 40; epoch++ {
		opt.ZeroGrad()
		logits := model.Forward(X)
		loss := nn.CrossEntropyLoss(logits, Y)
		loss.Backward()
		opt.Step()
		if epoch%10 == 0 {
			fmt.Printf("epoch %2d  loss=%.4f  acc=%.1f%%\n", epoch, loss.Item(), accuracy(logits, Y))
		}
	}
	model.Eval()
	fmt.Printf("final accuracy: %.1f%%\n", accuracy(model.Forward(X), Y))
}

func makeImgs(n, classes int) (*tensor.Tensor, *tensor.Tensor) {
	x := make([]float64, n*64)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		c := i % classes
		for r := 0; r < 8; r++ {
			for col := 0; col < 8; col++ {
				idx := i*64 + r*8 + col
				x[idx] = rand.NormFloat64() * 0.1
				bright := (c == 0 && r < 3 && col < 3) ||
					(c == 1 && r >= 5 && col >= 5) ||
					(c == 2 && r < 3 && col >= 5)
				if bright {
					x[idx] += 2.0
				}
			}
		}
		y[i] = float64(c)
	}
	return tensor.New(x, n, 1, 8, 8), tensor.New(y, n)
}

func accuracy(logits, targets *tensor.Tensor) float64 {
	pred := logits.ArgMax(1)
	c := 0
	for i := range pred.Data {
		if int(pred.Data[i]) == int(targets.Data[i]) {
			c++
		}
	}
	return 100 * float64(c) / float64(len(targets.Data))
}
```

## 6. Performance notes (CPU vs GPU)

- Conv layers lower to im2col + GEMM; the GEMM is the hot spot and is exactly
  what the CUDA backend accelerates. Building with `-tags cuda` and calling
  `backend.Use(...)` moves conv training onto cuBLAS with zero model changes
  (tutorial 08).
- The gather matrix is cached per input shape — keep batch/spatial sizes
  fixed within a training loop (the normal case) and you pay its construction
  once.

---

[← Training](04-training.md) | [Index](README.md) | [Next: Sequence Models →](06-sequence-models.md)
