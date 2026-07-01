// gonn_opencl.h — C interface to the OpenCL kernels in gonn_opencl.c.
// Mirrors the CUDA backend's C API (gonn_cuda.h) so the Go side is parallel.
// Linked from opencl.go via CGO when built with -tags opencl.
#ifndef GONN_OPENCL_H
#define GONN_OPENCL_H

// Elementwise op kinds. These mirror backend.UnaryKind / backend.BinaryKind
// in Go (backend/backend.go) and the CUDA enums (gonn_cuda.h) — the values
// are a shared ABI. Append only; never reorder or remove.
enum {
    GONN_CL_UN_RELU = 0,
    GONN_CL_UN_SIGMOID,
    GONN_CL_UN_TANH,
    GONN_CL_UN_EXP,
    GONN_CL_UN_LOG,
    GONN_CL_UN_GELU,
    GONN_CL_UN_SILU
};
enum {
    GONN_CL_BIN_ADD = 0,
    GONN_CL_BIN_SUB,
    GONN_CL_BIN_MUL,
    GONN_CL_BIN_DIV
};

// All return 0 on success, else a nonzero OpenCL error code.

// Batched row-major GEMM: for each of `batch` contiguous matrix pairs,
// C(m,n) = op(A) @ op(B) where op(A) is (m,k) (stored (k,m) when transA)
// and op(B) is (k,n) (stored (n,k) when transB). batch == 1 is a plain GEMM.
int gonn_cl_gemm(const double* A, const double* B, double* C,
                 int batch, int m, int k, int n, int transA, int transB);

// Elementwise unary: C[i] = f_kind(A[i]) for i in [0, n).
int gonn_cl_unary(int kind, const double* A, double* C, int n);

// Elementwise binary: C[i] = A[i] op_kind B[i] for i in [0, n).
int gonn_cl_binary(int kind, const double* A, const double* B, double* C, int n);

// Block until the queue drains.
void gonn_cl_sync(void);
// Returns 1 if an OpenCL device with fp64 was initialised, else 0.
int gonn_cl_available(void);

#endif // GONN_OPENCL_H
