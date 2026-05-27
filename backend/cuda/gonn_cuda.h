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

// Elementwise subtract: C[i] = A[i] - B[i]
void gonn_sub(const double* A, const double* B, double* C, int n);

// Elementwise divide: C[i] = A[i] / B[i]
void gonn_div(const double* A, const double* B, double* C, int n);

// Scalar multiply: C[i] = A[i] * s
void gonn_scale(const double* A, double s, double* C, int n);

// In-place axpy: OUT[i] += alpha * X[i]
// OUT is both an input (current value) and an output (updated value).
void gonn_axpy(double* OUT, const double* X, double alpha, int n);

// Reductions over n elements; result is a scalar written to *out_scalar.
void gonn_sum(const double* A, double* out_scalar, int n);
void gonn_max(const double* A, double* out_scalar, int n);

// Activations: C[i] = f(A[i])
void gonn_relu(const double* A, double* C, int n);
void gonn_sigmoid(const double* A, double* C, int n);
void gonn_tanh(const double* A, double* C, int n);
void gonn_exp(const double* A, double* C, int n);
void gonn_log(const double* A, double* C, int n);
// GELU: tanh approximation.
void gonn_gelu(const double* A, double* C, int n);
// SiLU / Swish: x * sigmoid(x).
void gonn_silu(const double* A, double* C, int n);

// Block until queued GPU work completes.
void gonn_sync();

#ifdef __cplusplus
}
#endif

#endif // GONN_CUDA_H
