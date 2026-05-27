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
#include <stdio.h>

static cublasHandle_t g_handle = nullptr;
static void ensure_handle() {
    if (!g_handle) cublasCreate(&g_handle);
}

// Elementwise add kernel.
__global__ void add_kernel(const double* A, const double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Elementwise multiply kernel.
__global__ void mul_kernel(const double* A, const double* B, double* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * B[i];
}

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

extern "C" void gonn_sync() { cudaDeviceSynchronize(); }
