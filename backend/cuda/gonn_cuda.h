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

// --- Device-resident benchmarks (no per-call H2D/D2H) -----------------------
// Allocate inputs once on the device, run `iters` GEMMs, and time them with
// CUDA events. Returns average ms per iter. f32 != 0 selects cublasSgemm
// (float), else cublasDgemm (double). Mirrors how PyTorch/tinygrad measure a
// device-resident op.
double gonn_bench_matmul_dev(int m, int k, int n, int iters, int f32);
// Device-resident elementwise add benchmark; returns average ms per iter.
double gonn_bench_add_dev(int n, int iters, int f32);

// --- Fused flash-attention forward (float64) --------------------------------
// Q,K,V,O are (BH, S, d) row-major; scale is usually 1/sqrt(d); causal!=0 masks
// future keys. One fused kernel, online softmax, no S*S materialization.
void gonn_flash_attn_f64(const double* Q, const double* K, const double* V,
                         double* O, int BH, int S, int d, double scale, int causal);
// Device-resident, CUDA-event-timed benchmark; returns average ms per iter.
double gonn_bench_flash_attn_f64(int BH, int S, int d, int iters, int causal);

// Training forward: also returns L (logsumexp per query row, length BH*S),
// needed by the backward pass.
void gonn_flash_attn_f64_fwd(const double* Q, const double* K, const double* V,
                             double* O, double* L, int BH, int S, int d,
                             double scale, int causal);
// Backward: from Q,K,V,O,L,dO compute dQ,dK,dV (each (BH,S,d)).
void gonn_flash_attn_f64_bwd(const double* Q, const double* K, const double* V,
                             const double* O, const double* L, const double* dO,
                             double* dQ, double* dK, double* dV,
                             int BH, int S, int d, double scale, int causal);

// --- Device-resident buffers (allocate once, chain ops, copy back once) -----
void* gonn_dev_alloc(long bytes);
void  gonn_dev_free(void* p);
void  gonn_dev_upload(void* dst, const double* src, long n);
void  gonn_dev_download(double* dst, const void* src, long n);
void  gonn_dev_sync(void);
void  gonn_dev_matmul_f64(const void* dA, const void* dB, void* dC, int m, int k, int n);
void  gonn_dev_add_f64(const void* dA, const void* dB, void* dC, int n);
void  gonn_dev_relu_f64(void* dA, int n);
// fp16 tensor-core path
void* gonn_dev_upload_f16(const double* src, int n);      // returns device __half buffer
void  gonn_dev_download_f16(double* dst, const void* dHalf, int n);
void  gonn_dev_matmul_f16(const void* dA, const void* dB, void* dC, int m, int k, int n);
double gonn_bench_matmul_f16_dev(int m, int k, int n, int iters);

#ifdef __cplusplus
}
#endif

#endif // GONN_CUDA_H
