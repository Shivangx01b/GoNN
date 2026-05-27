// gonn_cuda.cu — CUDA kernels for GoNN's accelerated backend.
//
// Build (Linux/macOS):
//   nvcc -O3 -Xcompiler -fPIC -shared gonn_cuda.cu -o libgonn_cuda.so -lcublas
//
// Build (Windows):
//   nvcc -O3 gonn_cuda.cu -o gonn_cuda.dll --shared -Xcompiler "/MD" -lcublas
//
// Then run Go with: go build -tags cuda
//
// The C interface (declared in gonn_cuda.h) is consumed by cuda.go via CGO.

#include "gonn_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <stdio.h>

static cublasHandle_t g_handle = nullptr;
static void ensure_handle() {
    if (!g_handle) cublasCreate(&g_handle);
}

// ---------------------------------------------------------------------------
// Elementwise kernels
// ---------------------------------------------------------------------------

__global__ void add_kernel(const double* A, const double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void mul_kernel(const double* A, const double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * B[i];
}

__global__ void sub_kernel(const double* A, const double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] - B[i];
}

__global__ void div_kernel(const double* A, const double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] / B[i];
}

__global__ void scale_kernel(const double* A, double s, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * s;
}

__global__ void axpy_kernel(double* OUT, const double* X, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) OUT[i] += alpha * X[i];
}

// ---------------------------------------------------------------------------
// Reduction kernels (one-block tree reduce in shared memory).
//
// Each thread strides over the input loading + locally combining, then a
// shared-memory tree reduction collapses the block to a single value.
// Launched with a single block so the result lives in dOut[0].
// ---------------------------------------------------------------------------

#define REDUCE_THREADS 256

__global__ void sum_kernel(const double* A, double* out, int n) {
    __shared__ double sdata[REDUCE_THREADS];
    int tid = threadIdx.x;

    double acc = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
        acc += A[i];
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

__global__ void max_kernel(const double* A, double* out, int n) {
    __shared__ double sdata[REDUCE_THREADS];
    int tid = threadIdx.x;

    // Sentinel for "empty" lane.
    double acc = -CUDART_INF;
    for (int i = tid; i < n; i += blockDim.x) {
        double v = A[i];
        if (v > acc) acc = v;
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

// ---------------------------------------------------------------------------
// Activation kernels
// ---------------------------------------------------------------------------

__global__ void relu_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double v = A[i];
        C[i] = v > 0.0 ? v : 0.0;
    }
}

__global__ void sigmoid_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = 1.0 / (1.0 + exp(-A[i]));
}

__global__ void tanh_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = tanh(A[i]);
}

__global__ void exp_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = exp(A[i]);
}

__global__ void log_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = log(A[i]);
}

__global__ void gelu_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // sqrt(2/pi)
        const double k = 0.7978845608028654;
        double x = A[i];
        double inner = k * (x + 0.044715 * x * x * x);
        C[i] = 0.5 * x * (1.0 + tanh(inner));
    }
}

__global__ void silu_kernel(const double* A, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = A[i];
        C[i] = x / (1.0 + exp(-x));
    }
}

// ---------------------------------------------------------------------------
// C wrappers — allocate, copy in, launch, copy back, free.
// Follow the same simple pattern as the existing add/mul/matmul wrappers.
// ---------------------------------------------------------------------------

extern "C" void gonn_matmul(const double* A, const double* B, double* C, int m, int k, int n) {
    ensure_handle();
    double *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(double) * m * k);
    cudaMalloc(&dB, sizeof(double) * k * n);
    cudaMalloc(&dC, sizeof(double) * m * n);
    cudaMemcpy(dA, A, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    // cuBLAS is column-major. To compute C = A*B (row-major) we compute
    // C^T = B^T * A^T, then cuBLAS's output is C in row-major layout.
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha, dB, n, dA, k,
                &beta, dC, n);

    cudaMemcpy(C, dC, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

extern "C" void gonn_add(const double* A, const double* B, double* C, int n) {
    double *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dB, sizeof(double) * n);
    cudaMalloc(&dC, sizeof(double) * n);
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * n, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(dA, dB, dC, n);
    cudaMemcpy(C, dC, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

extern "C" void gonn_mul(const double* A, const double* B, double* C, int n) {
    double *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dB, sizeof(double) * n);
    cudaMalloc(&dC, sizeof(double) * n);
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * n, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(dA, dB, dC, n);
    cudaMemcpy(C, dC, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

extern "C" void gonn_sub(const double* A, const double* B, double* C, int n) {
    double *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dB, sizeof(double) * n);
    cudaMalloc(&dC, sizeof(double) * n);
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * n, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sub_kernel<<<blocks, threads>>>(dA, dB, dC, n);
    cudaMemcpy(C, dC, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

extern "C" void gonn_div(const double* A, const double* B, double* C, int n) {
    double *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dB, sizeof(double) * n);
    cudaMalloc(&dC, sizeof(double) * n);
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * n, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    div_kernel<<<blocks, threads>>>(dA, dB, dC, n);
    cudaMemcpy(C, dC, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

extern "C" void gonn_scale(const double* A, double s, double* C, int n) {
    double *dA, *dC;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dC, sizeof(double) * n);
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(dA, s, dC, n);
    cudaMemcpy(C, dC, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dC);
}

extern "C" void gonn_axpy(double* OUT, const double* X, double alpha, int n) {
    double *dOut, *dX;
    cudaMalloc(&dOut, sizeof(double) * n);
    cudaMalloc(&dX, sizeof(double) * n);
    cudaMemcpy(dOut, OUT, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, sizeof(double) * n, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    axpy_kernel<<<blocks, threads>>>(dOut, dX, alpha, n);
    cudaMemcpy(OUT, dOut, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dOut); cudaFree(dX);
}

extern "C" void gonn_sum(const double* A, double* out_scalar, int n) {
    double *dA, *dOut;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dOut, sizeof(double));
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    sum_kernel<<<1, REDUCE_THREADS>>>(dA, dOut, n);
    cudaMemcpy(out_scalar, dOut, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dOut);
}

extern "C" void gonn_max(const double* A, double* out_scalar, int n) {
    double *dA, *dOut;
    cudaMalloc(&dA, sizeof(double) * n);
    cudaMalloc(&dOut, sizeof(double));
    cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);
    max_kernel<<<1, REDUCE_THREADS>>>(dA, dOut, n);
    cudaMemcpy(out_scalar, dOut, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dOut);
}

// Generic unary launcher macro: copies A in, launches KERNEL, copies C out.
#define GONN_UNARY(NAME, KERNEL)                                           \
    extern "C" void NAME(const double* A, double* C, int n) {              \
        double *dA, *dC;                                                   \
        cudaMalloc(&dA, sizeof(double) * n);                               \
        cudaMalloc(&dC, sizeof(double) * n);                               \
        cudaMemcpy(dA, A, sizeof(double) * n, cudaMemcpyHostToDevice);     \
        int threads = 256;                                                 \
        int blocks = (n + threads - 1) / threads;                          \
        KERNEL<<<blocks, threads>>>(dA, dC, n);                            \
        cudaMemcpy(C, dC, sizeof(double) * n, cudaMemcpyDeviceToHost);     \
        cudaFree(dA); cudaFree(dC);                                        \
    }

GONN_UNARY(gonn_relu,    relu_kernel)
GONN_UNARY(gonn_sigmoid, sigmoid_kernel)
GONN_UNARY(gonn_tanh,    tanh_kernel)
GONN_UNARY(gonn_exp,     exp_kernel)
GONN_UNARY(gonn_log,     log_kernel)
GONN_UNARY(gonn_gelu,    gelu_kernel)
GONN_UNARY(gonn_silu,    silu_kernel)

extern "C" void gonn_sync() { cudaDeviceSynchronize(); }
