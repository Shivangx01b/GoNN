// gonn_cuda.h — C interface to the CUDA kernels in gonn_cuda.cu.
// Linked from cuda.go via CGO when built with -tags cuda.
#ifndef GONN_CUDA_H
#define GONN_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Elementwise op kinds. These mirror backend.UnaryKind / backend.BinaryKind
// in Go (backend/backend.go) — the values are a shared ABI. Append only;
// never reorder or remove.
enum {
    GONN_UN_RELU = 0,
    GONN_UN_SIGMOID,
    GONN_UN_TANH,
    GONN_UN_EXP,
    GONN_UN_LOG,
    GONN_UN_GELU,       /* tanh approximation */
    GONN_UN_SILU,       /* x * sigmoid(x) */
    GONN_UN_GELU_EXACT  /* 0.5*x*(1+erf(x/sqrt(2))) — append only, never reorder */
};
enum {
    GONN_BIN_ADD = 0,
    GONN_BIN_SUB,
    GONN_BIN_MUL,
    GONN_BIN_DIV
};

// All return 0 on success, else a cudaError_t / cublasStatus_t value.

// Batched row-major GEMM: for each of `batch` contiguous matrix pairs,
// C(m,n) = op(A) @ op(B) where op(A) is (m,k) (stored (k,m) when transA)
// and op(B) is (k,n) (stored (n,k) when transB). batch == 1 is a plain GEMM.
// Uses cublasDgemmStridedBatched.
int gonn_gemm(const double* A, const double* B, double* C,
              int batch, int m, int k, int n, int transA, int transB);

// Elementwise unary: C[i] = f_kind(A[i]) for i in [0, n).
int gonn_unary(int kind, const double* A, double* C, long n);

// Elementwise binary: C[i] = A[i] op_kind B[i] for i in [0, n).
int gonn_binary(int kind, const double* A, const double* B, double* C, long n);

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
