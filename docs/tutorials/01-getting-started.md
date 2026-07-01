# 01 — Getting Started

[Index](README.md) | [Next: Tensors & Autograd →](02-tensors-and-autograd.md)

GoNN is a pure-Go deep learning framework with PyTorch-style autograd. This
tutorial takes you from install to a trained model in about ten minutes,
entirely on CPU — no C toolchain, no Python, no GPU required.

## 1. Install

Inside an existing Go module:

```bash
go get github.com/Shivangx01b/GoNN
```

Or clone and run the examples directly:

```bash
git clone https://github.com/Shivangx01b/GoNN
cd GoNN
go test ./...        # everything should pass on a plain CPU build
go run ./examples/mlp
```

## 2. Your first tensor

A `*tensor.Tensor` is a flat `[]float64` plus a shape. Everything in GoNN —
model weights, activations, gradients — is one of these.

```go
package main

import (
	"fmt"

	"gonn/tensor"
)

func main() {
	a := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3) // shape (2, 3)
	b := tensor.Ones(2, 3)

	fmt.Println(a.Add(b))        // elementwise add
	fmt.Println(a.MulScalar(10)) // scalar broadcast
	fmt.Println(a.Transpose())   // (3, 2)
	fmt.Println(a.Sum().Item())  // 21
}
```

## 3. Your first gradient

Mark a tensor with `SetRequiresGrad(true)`, build any expression from it, and
call `Backward()` on a scalar result. Gradients accumulate into `.Grad`:

```go
x := tensor.New([]float64{1, 2, 3}, 3, 1).SetRequiresGrad(true)
W := tensor.New([]float64{2, -1, 0.5}, 1, 3).SetRequiresGrad(true)

y := W.MatMul(x).Square().Sum() // y = (W·x)² , a scalar
y.Backward()

fmt.Println("dy/dx =", x.Grad) // [6, -3, 1.5]
fmt.Println("dy/dW =", W.Grad) // [3, 6, 9]
```

That's reverse-mode automatic differentiation: each op records how to push
gradients back to its inputs, and `Backward()` walks the recorded graph in
reverse. Tutorial 02 opens the hood.

## 4. Your first trained model

Everything above composes into the canonical training loop. This is a
complete program — linear regression with stochastic gradient descent:

```go
package main

import (
	"fmt"

	"gonn/optim"
	"gonn/tensor"
)

func main() {
	// Synthetic data: y = 3x + 2 (+ the model has to figure that out).
	n := 100
	xs := make([]float64, n)
	ys := make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i)/float64(n)*4 - 2
		xs[i] = x
		ys[i] = 3*x + 2
	}
	X := tensor.New(xs, n, 1)
	Y := tensor.New(ys, n, 1)

	// Parameters.
	W := tensor.Randn(1, 1).SetRequiresGrad(true)
	b := tensor.Zeros(1).SetRequiresGrad(true)

	opt := optim.NewSGD([]*tensor.Tensor{W, b}, 0.1)

	for epoch := 0; epoch <= 200; epoch++ {
		opt.ZeroGrad()
		pred := X.MatMul(W).Add(b)
		loss := pred.Sub(Y).Square().Mean()
		loss.Backward()
		opt.Step()
		if epoch%50 == 0 {
			fmt.Printf("epoch %3d  loss=%.5f  W=%.3f  b=%.3f\n",
				epoch, loss.Item(), W.Data[0], b.Data[0])
		}
	}
}
```

The four-beat rhythm — `ZeroGrad` → forward → `Backward` → `Step` — is the
same for every model in GoNN, from this one-parameter regression to a
transformer.

## 5. The package map

| Package | Role |
|---------|------|
| `tensor` | N-d arrays + autograd + batched matmul + the unary-op registry |
| `nn` | Layers, containers, losses, weight init |
| `optim` | 16 optimizers, parameter groups, gradient clipping, 11 LR schedulers |
| `data` | `Dataset`, `DataLoader`, transforms, synthetic data, MNIST/CSV loaders |
| `ml` | Classical ML (trees, clustering, PCA, …) — standalone, no autograd |
| `backend` | Compute backends: CPU (default), CUDA (`-tags cuda`), OpenCL (`-tags opencl`) |

## 6. CPU and GPU in one sentence each

- **CPU** is the default: pure Go, one static binary, GEMMs via gonum BLAS.
  Everything in tutorials 01–07 runs this way with zero setup.
- **GPU** is opt-in: build with `-tags cuda`, call
  `backend.Use(...)` once, and GEMMs (plus large activations) run on the
  GPU — tutorial 08 covers it end to end, including a no-toolkit Docker path.

---

[Index](README.md) | [Next: Tensors & Autograd →](02-tensors-and-autograd.md)
