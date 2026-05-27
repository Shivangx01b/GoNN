// gonn_cuda.h — C interface to the CUDA kernels in gonn_cuda.cu.
// Linked from cuda.go via CGO when built with -tags cuda.
#ifndef GONN_CUDA_H
#define GONN_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Matrix multiply (row-major): C(m,n) = A(m,k) * B(k,n)
void gonn_matmul(const double* A, const double* B, double* C, int m, int k, int n);

// Elementwise add: C[i] = A[i] + B[i]
void gonn_add(const double* A, const double* B, double* C, int n);

// Elementwise multiply: C[i] = A[i] * B[i]
void gonn_mul(const double* A, const double* B, double* C, int n);

// Block until queued GPU work completes.
void gonn_sync();

#ifdef __cplusplus
}
#endif

#endif // GONN_CUDA_H
