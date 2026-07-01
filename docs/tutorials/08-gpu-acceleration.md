# 08 — GPU Acceleration

[← Classical ML](07-classical-ml.md) | [Index](README.md) | [Next: Extending GoNN →](09-extending-gonn.md)

How GoNN's compute backends work, how to build and select the CUDA/OpenCL
backends, what actually gets accelerated (and what doesn't — honestly), and
the device-resident / fused-kernel paths for maximum throughput.

## 1. The backend model

The `backend` package defines a deliberately tiny contract:

```go
type Backend interface {
	Name() Device
	// Row-major batched C = op(A) @ op(B), with transpose flags.
	Gemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64
	Synchronize()
}

// Optional capability: enum-dispatched elementwise kernels.
type Elementwiser interface {
	Unary(kind UnaryKind, a, out []float64) bool  // false = decline -> CPU fallback
	Binary(kind BinaryKind, a, b, out []float64) bool
}
```

Selection is a single goroutine-safe switch — **no model code changes**:

```go
import (
	"gonn/backend"
	"gonn/backend/cuda"
)

if b, err := cuda.Backend(); err == nil {
	backend.Use(b) // every tensor GEMM (fwd + bwd) now runs on the GPU
} else {
	fmt.Println("CPU fallback:", err) // clear error on non-cuda builds
}
```

| Backend | Build | GEMM | Elementwise kernels |
|---------|-------|------|---------------------|
| CPU (default) | none | gonum BLAS (pure Go) | pure-Go closures (the reference path) |
| CUDA | `-tags cuda` | cuBLAS strided-batched | relu, sigmoid, tanh, exp, log, gelu, silu + add/sub/mul/div |
| OpenCL | `-tags opencl` | portable fp64 kernel | same enum set |

## 2. Building with CUDA

**Option A — local toolkit** (Linux, or anywhere `nvcc` lives):

```bash
cd backend/cuda
nvcc -O3 -arch=sm_86 -Xcompiler -fPIC -shared gonn_cuda.cu -o libgonn_cuda.so -lcublas
cd ../..
export CGO_CFLAGS="-I$PWD/backend/cuda"
export CGO_LDFLAGS="-L$PWD/backend/cuda -lgonn_cuda -lcudart -lcublas"
export LD_LIBRARY_PATH="$PWD/backend/cuda:$LD_LIBRARY_PATH"
go build -tags cuda ./...
```

**Option B — Docker, no local toolkit needed** (this is how the repo itself
is verified; works on Windows via WSL2 GPU passthrough):

```bash
docker build -f benchmark/docker/Dockerfile.cuda -t gonn-cuda .
docker run --rm --gpus all -v "$PWD":/work -w /work gonn-cuda \
    bash benchmark/docker/build_and_run.sh
```

That script compiles the kernels, builds `-tags cuda`, verifies GPU-vs-CPU
parity (GEMM incl. batched/transposed at 1e-12, every elementwise kind), runs
`go test -tags cuda` with GPU dispatch forced on, and finishes with the
benchmark suite. If it prints `VERIFY: ALL OK on cuda`, your setup is good.

> Git Bash on Windows: prefix `docker run` with `MSYS_NO_PATHCONV=1` so
> `-w /work` isn't rewritten to a Windows path.

**OpenCL** (`-tags opencl`) mirrors the CUDA backend with portable fp64
kernels; `bash benchmark/docker/opencl_run.sh` verifies it against a
software OpenCL runtime, and the same binary runs on any GPU with an OpenCL
ICD.

## 3. Training on the GPU — a complete program

The only GPU-specific lines are the import and `backend.Use`. Build tags do
the rest: on a CPU-only build, `cuda.Backend()` returns an error and the
program transparently trains on CPU.

```go
package main

import (
	"fmt"

	"gonn/backend"
	"gonn/backend/cuda"
	"gonn/data"
	"gonn/nn"
	"gonn/optim"
)

func main() {
	// Select the best available device. Everything below is device-agnostic.
	device := "cpu"
	if b, err := cuda.Backend(); err == nil {
		backend.Use(b)
		device = "cuda"
	}
	fmt.Println("training on:", device)

	X, Y := data.MakeClassification(2000, 128, 4, 1)
	loader := data.NewDataLoader(data.NewTensorDataset(X, Y), 128, true)

	model := nn.NewSequential(
		nn.NewLinear(128, 512, true), // these GEMMs run on the selected device
		nn.GELU(),
		nn.NewLinear(512, 512, true),
		nn.GELU(),
		nn.NewLinear(512, 4, true),
	)
	opt := optim.NewAdamW(model.Parameters(), 1e-3)

	model.Train()
	for epoch := 0; epoch < 5; epoch++ {
		var loss float64
		var n int
		for batch := range loader.Iter() {
			opt.ZeroGrad()
			l := nn.CrossEntropyLoss(model.Forward(batch.X), batch.Y)
			l.Backward()
			opt.Step()
			loss += l.Item()
			n++
		}
		fmt.Printf("epoch %d  loss=%.4f\n", epoch, loss/float64(n))
	}
}
```

Run it both ways:

```bash
go run .                 # CPU
go run -tags cuda .      # GPU (with the CUDA env from §2)
```

## 4. What is accelerated — and the honest cost model

Tensor storage lives in **host memory**; the eager GPU path copies inputs
up and results back per call. That shapes what's worth dispatching:

- **GEMMs always dispatch** — matmul is compute-dense enough that the PCIe
  copies are amortized. Linear/conv/attention layers (whose forward *and*
  backward are GEMMs) are where the GPU pays off.
- **Transcendental activations** (`exp`, `tanh`, `gelu`, …) dispatch above a
  size threshold. Default `UnaryMinElems = 1<<16` (64K elements), tuned from
  the measured break-even on an RTX 3060: dispatched tanh at 64K elements is
  3.2× faster than the host loop.
- **Bandwidth-bound binaries** (`add`, `mul`, …) are wired but **disabled by
  default** — with host-resident data the copy alone costs more than the
  compute. This is a measured engineering decision, not a limitation you
  should work around.

Tune it (or force it, for experiments) at runtime:

```go
tensor.SetDispatchPolicy(tensor.DispatchPolicy{
	UnaryMinElems:  1 << 15,     // dispatch smaller activations
	BinaryMinElems: math.MaxInt, // keep binaries on CPU (recommended)
})
```

Autograd backwards for elementwise ops intentionally run on the CPU
reference closures — gradcheck-ability is worth more than the marginal
speedup, and the GEMM backwards (the actual hot spots) do run on the GPU.

## 5. Device-resident buffers: skipping the copies

For inference pipelines where the eager copies dominate, keep data on the
GPU explicitly (CUDA builds only):

```go
import "gonn/backend/cuda"

// Upload weights once.
dW1 := cuda.DevUpload(w1Data) // []float64 -> device
dW2 := cuda.DevUpload(w2Data)
dH := cuda.DevAlloc(batch * hidden)
dOut := cuda.DevAlloc(batch * out)
defer func() { dW1.Free(); dW2.Free(); dH.Free(); dOut.Free() }()

// Per request: one upload, chained device ops, one download.
dX := cuda.DevUpload(x)
cuda.DevMatMul(dX, dW1, dH, batch, in, hidden)
cuda.DevReLU(dH, batch*hidden)
cuda.DevMatMul(dH, dW2, dOut, batch, hidden, out)
cuda.DevSync()
y := dOut.Download()
dX.Free()
```

`benchmark/resident` measures this MLP at a large multiple of the eager
path. There is also an **fp16 tensor-core** GEMM path
(`cuda.DevUploadF16`, `cuda.DevMatMulF16`) — ~24 TFLOP/s on an RTX 3060
versus ~173 GFLOP/s for fp64.

## 6. Fused flash-attention (trains on the GPU)

On CUDA builds, `nn.MultiHeadAttention.Forward` automatically routes its
scaled-dot-product core through a fused fp64 flash-attention kernel
(online softmax, no S×S materialization) whenever `headDim == 64` and the
query/key lengths match — **including the backward pass** (gradcheck ≈
5e-8), so ordinary training code gets the fused kernel with zero changes.
On causal fp64 attention it outperforms PyTorch's SDPA by ~1.4–1.5× on the
same GPU. `ForwardFused` is the inference-only variant.

```go
mha := nn.NewMultiHeadAttention(512, 8) // headDim = 512/8 = 64 -> fused path
out := mha.Forward(x, x, x, true)       // trains through the kernel on CUDA builds
```

## 7. Measured numbers (RTX 3060, from `benchmark/`)

| Op | GoNN | Reference |
|----|------|-----------|
| matmul f64 (N=2048, device-resident) | ~173 GFLOP/s | PyTorch ~174 (both cuBLAS) |
| matmul f32 | ~7.9 TFLOP/s | PyTorch ~7.6 |
| matmul f16 tensor-core | ~24.2 TFLOP/s | — |
| causal flash-attention f64 | ~89–94 GFLOP/s | PyTorch SDPA 58–66 |

Full methodology: [`benchmark/REPORT.md`](../../benchmark/REPORT.md).

---

[← Classical ML](07-classical-ml.md) | [Index](README.md) | [Next: Extending GoNN →](09-extending-gonn.md)
