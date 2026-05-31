#!/usr/bin/env bash
# Build the CUDA backend + GoNN with -tags cuda, verify correctness on GPU,
# then run the GoNN CUDA benchmark. Runs INSIDE the gonn-cuda docker image.
set -euo pipefail

echo "=== nvcc / go / gpu ==="
nvcc --version | tail -2
go version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== 1. Compile CUDA kernels -> libgonn_cuda.so ==="
cd backend/cuda
nvcc -O3 -Xcompiler -fPIC -shared gonn_cuda.cu -o libgonn_cuda.so -lcublas
ls -la libgonn_cuda.so
cd /work

# CGO needs to find the header (-I) and the shared libs (-L) + at runtime.
# CUDA runtime/cublas live in /usr/local/cuda/lib64 (a symlink) in the devel image.
CUDA_LIB="/usr/local/cuda/lib64"
[ -e "$CUDA_LIB/libcudart.so" ] || CUDA_LIB="$(dirname "$(find -L /usr/local -name libcudart.so 2>/dev/null | head -1)")"
echo "CUDA libs: $CUDA_LIB"
export CGO_CFLAGS="-I/work/backend/cuda"
export CGO_LDFLAGS="-L/work/backend/cuda -L${CUDA_LIB} -lgonn_cuda -lcudart -lcublas"
export LD_LIBRARY_PATH="/work/backend/cuda:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"

echo "=== 2. Build GoNN with -tags cuda ==="
go build -tags cuda ./...

echo "=== 3. Correctness: CUDA vs CPU on GPU ==="
go run -tags cuda ./benchmark/verify

echo "=== 4. GoNN CUDA benchmark (per-call H2D/D2H) ==="
go run -tags cuda ./benchmark/gonn

echo "=== 5. GoNN CUDA device-resident benchmark (f32 + f64) ==="
go run -tags cuda ./benchmark/gpuresident

echo "=== 6. GoNN fused flash-attention (f64): correctness + benchmark ==="
go run -tags cuda ./benchmark/flashattn

echo "=== 7. MultiHeadAttention.ForwardFused: correctness + speedup vs Forward ==="
go run -tags cuda ./benchmark/mha

echo "=== done ==="
