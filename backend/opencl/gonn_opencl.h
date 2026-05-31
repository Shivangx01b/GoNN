// gonn_opencl.h — C interface to the OpenCL kernels in gonn_opencl.c.
// Mirrors the CUDA backend's C API (gonn_cuda.h) so the Go side is parallel.
// Linked from opencl.go via CGO when built with -tags opencl.
#ifndef GONN_OPENCL_H
#define GONN_OPENCL_H

// Matrix multiply (row-major): C(m,n) = A(m,k) * B(k,n)
void gonn_cl_matmul(const double* A, const double* B, double* C, int m, int k, int n);

// Elementwise: C[i] = A[i] (op) B[i]
void gonn_cl_add(const double* A, const double* B, double* C, int n);
void gonn_cl_mul(const double* A, const double* B, double* C, int n);
void gonn_cl_sub(const double* A, const double* B, double* C, int n);
void gonn_cl_div(const double* A, const double* B, double* C, int n);

// Scalar multiply: C[i] = A[i] * s
void gonn_cl_scale(const double* A, double s, double* C, int n);
// In-place axpy: OUT[i] += alpha * X[i]
void gonn_cl_axpy(double* OUT, const double* X, double alpha, int n);

// Reductions over n elements; scalar result written to *out_scalar.
void gonn_cl_sum(const double* A, double* out_scalar, int n);
void gonn_cl_max(const double* A, double* out_scalar, int n);

// Activations: C[i] = f(A[i])
void gonn_cl_relu(const double* A, double* C, int n);
void gonn_cl_sigmoid(const double* A, double* C, int n);
void gonn_cl_tanh(const double* A, double* C, int n);
void gonn_cl_exp(const double* A, double* C, int n);
void gonn_cl_log(const double* A, double* C, int n);
void gonn_cl_gelu(const double* A, double* C, int n);
void gonn_cl_silu(const double* A, double* C, int n);

// Block until the queue drains.
void gonn_cl_sync(void);
// Returns 1 if an OpenCL GPU device + fp64 was initialised, else 0.
int gonn_cl_available(void);

#endif // GONN_OPENCL_H
