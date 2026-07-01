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
//
// Layout:
//   1. Elementwise kernels, macro-generated (DEF_UNOP / DEF_BINOP). Adding an
//      op = one DEF_* line + one switch case + one enum value in the header
//      (mirrored in backend/backend.go).
//   2. gonn_gemm / gonn_unary / gonn_binary — the eager host-pointer entry
//      points used by the Go backend. They stage through a grow-only device
//      workspace cache (g_ws) instead of cudaMalloc/Free per call; the Go
//      side serializes entry with a mutex, so the cache needs no locking.
//   3. Fused flash-attention (fwd + bwd) and the device-resident buffer / f16
//      tensor-core / benchmark sections — unchanged from the verified
//      implementation (fwd/bwd gradcheck ~5e-8 on an RTX 3060).

#include "gonn_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdio.h>

static cublasHandle_t g_handle = nullptr;
static void ensure_handle() {
    if (!g_handle) cublasCreate(&g_handle);
}

// ---------------------------------------------------------------------------
// Error handling + launch configuration
// ---------------------------------------------------------------------------

#define GONN_THREADS 256

// Evaluate a CUDA runtime call; on failure print and return its error code.
#define GONN_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "gonn_cuda: %s: %s\n", #call,                    \
                    cudaGetErrorString(err_));                               \
            return (int)err_;                                                \
        }                                                                    \
    } while (0)

// Evaluate a cuBLAS call; on failure print and return its status code.
#define GONN_CHECK_CUBLAS(call)                                              \
    do {                                                                     \
        cublasStatus_t st_ = (call);                                         \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "gonn_cuda: %s: cublas status %d\n", #call,      \
                    (int)st_);                                               \
            return (int)st_;                                                 \
        }                                                                    \
    } while (0)

// Check the last kernel launch.
#define GONN_CHECK_LAUNCH(what)                                              \
    do {                                                                     \
        cudaError_t err_ = cudaGetLastError();                               \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "gonn_cuda: launch %s: %s\n", what,              \
                    cudaGetErrorString(err_));                               \
            return (int)err_;                                                \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Elementwise kernels (macro-generated). Long indices so >2^31-element
// tensors do not overflow. Expressions are kept identical to the historical
// hand-written kernels — numerics unchanged.
// ---------------------------------------------------------------------------

#define DEF_UNOP(NAME, EXPR)                                                 \
    __global__ void NAME##_kernel(const double* A, double* C, long n) {      \
        long i = blockIdx.x * (long)blockDim.x + threadIdx.x;                \
        if (i < n) { double x = A[i]; C[i] = (EXPR); }                       \
    }

#define DEF_BINOP(NAME, EXPR)                                                \
    __global__ void NAME##_kernel(const double* A, const double* B,          \
                                  double* C, long n) {                       \
        long i = blockIdx.x * (long)blockDim.x + threadIdx.x;                \
        if (i < n) { double a = A[i]; double b = B[i]; C[i] = (EXPR); }      \
    }

DEF_UNOP(relu,    x > 0.0 ? x : 0.0)
DEF_UNOP(sigmoid, 1.0 / (1.0 + exp(-x)))
DEF_UNOP(tanh,    tanh(x))
DEF_UNOP(exp,     exp(x))
DEF_UNOP(log,     log(x))
// GELU, tanh approximation; 0.7978845608028654 = sqrt(2/pi)
DEF_UNOP(gelu,    0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))))
DEF_UNOP(silu,    x / (1.0 + exp(-x)))

DEF_BINOP(add, a + b)
DEF_BINOP(sub, a - b)
DEF_BINOP(mul, a * b)
DEF_BINOP(div, a / b)

// ---------------------------------------------------------------------------
// Grow-only device workspace cache for the eager host-pointer entry points.
// Slot 0/1 stage inputs, slot 2 stages the output. NOT reentrant — the Go
// side (cuda.go) holds a mutex across every gonn_gemm/unary/binary call.
// ---------------------------------------------------------------------------

static struct { void* p; size_t cap; } g_ws[3];

static int ws_reserve(int slot, size_t bytes) {
    if (g_ws[slot].cap >= bytes) return 0;
    if (g_ws[slot].p) cudaFree(g_ws[slot].p);
    g_ws[slot].p = nullptr;
    g_ws[slot].cap = 0;
    cudaError_t err = cudaMalloc(&g_ws[slot].p, bytes);
    if (err != cudaSuccess) {
        g_ws[slot].p = nullptr;
        fprintf(stderr, "gonn_cuda: workspace alloc %zu bytes: %s\n", bytes,
                cudaGetErrorString(err));
        return (int)err;
    }
    g_ws[slot].cap = bytes;
    return 0;
}

// ---------------------------------------------------------------------------
// Eager entry points: gonn_gemm / gonn_unary / gonn_binary
// ---------------------------------------------------------------------------

// Batched row-major GEMM via cublasDgemmStridedBatched.
//
// cuBLAS is column-major, so we compute C^T = op(B)^T * op(A)^T: the
// column-major (n,m) result written to dC is exactly the row-major (m,n) C.
// A row-major stored matrix IS its transpose viewed column-major, hence:
//   transX == false -> the column-major view already provides op(X)^T -> OP_N
//   transX == true  -> take the column-major transpose               -> OP_T
// Leading dims are the stored row-major column counts.
extern "C" int gonn_gemm(const double* A, const double* B, double* C,
                         int batch, int m, int k, int n, int transA, int transB) {
    ensure_handle();
    size_t an = (size_t)batch * m * k, bn = (size_t)batch * k * n, cn = (size_t)batch * m * n;
    int rc;
    if ((rc = ws_reserve(0, an * sizeof(double))) != 0) return rc;
    if ((rc = ws_reserve(1, bn * sizeof(double))) != 0) return rc;
    if ((rc = ws_reserve(2, cn * sizeof(double))) != 0) return rc;
    double* dA = (double*)g_ws[0].p;
    double* dB = (double*)g_ws[1].p;
    double* dC = (double*)g_ws[2].p;
    GONN_CHECK(cudaMemcpy(dA, A, an * sizeof(double), cudaMemcpyHostToDevice));
    GONN_CHECK(cudaMemcpy(dB, B, bn * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0, beta = 0.0;
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = transA ? m : k; // stored row-major col count of A
    int ldb = transB ? k : n; // stored row-major col count of B
    GONN_CHECK_CUBLAS(cublasDgemmStridedBatched(
        g_handle, opB, opA, n, m, k,
        &alpha,
        dB, ldb, (long long)k * n,
        dA, lda, (long long)m * k,
        &beta,
        dC, n, (long long)m * n,
        batch));

    GONN_CHECK(cudaMemcpy(C, dC, cn * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

static int launch_unary(int kind, const double* dA, double* dC, long n) {
    int blocks = (int)((n + GONN_THREADS - 1) / GONN_THREADS);
    switch (kind) {
    case GONN_UN_RELU:    relu_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    case GONN_UN_SIGMOID: sigmoid_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    case GONN_UN_TANH:    tanh_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    case GONN_UN_EXP:     exp_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    case GONN_UN_LOG:     log_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    case GONN_UN_GELU:    gelu_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    case GONN_UN_SILU:    silu_kernel<<<blocks, GONN_THREADS>>>(dA, dC, n); break;
    default:
        fprintf(stderr, "gonn_cuda: unknown unary kind %d\n", kind);
        return -1;
    }
    GONN_CHECK_LAUNCH("unary");
    return 0;
}

static int launch_binary(int kind, const double* dA, const double* dB, double* dC, long n) {
    int blocks = (int)((n + GONN_THREADS - 1) / GONN_THREADS);
    switch (kind) {
    case GONN_BIN_ADD: add_kernel<<<blocks, GONN_THREADS>>>(dA, dB, dC, n); break;
    case GONN_BIN_SUB: sub_kernel<<<blocks, GONN_THREADS>>>(dA, dB, dC, n); break;
    case GONN_BIN_MUL: mul_kernel<<<blocks, GONN_THREADS>>>(dA, dB, dC, n); break;
    case GONN_BIN_DIV: div_kernel<<<blocks, GONN_THREADS>>>(dA, dB, dC, n); break;
    default:
        fprintf(stderr, "gonn_cuda: unknown binary kind %d\n", kind);
        return -1;
    }
    GONN_CHECK_LAUNCH("binary");
    return 0;
}

extern "C" int gonn_unary(int kind, const double* A, double* C, long n) {
    if (n <= 0) return 0;
    int rc;
    if ((rc = ws_reserve(0, (size_t)n * sizeof(double))) != 0) return rc;
    if ((rc = ws_reserve(2, (size_t)n * sizeof(double))) != 0) return rc;
    double* dA = (double*)g_ws[0].p;
    double* dC = (double*)g_ws[2].p;
    GONN_CHECK(cudaMemcpy(dA, A, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    if ((rc = launch_unary(kind, dA, dC, n)) != 0) return rc;
    GONN_CHECK(cudaMemcpy(C, dC, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int gonn_binary(int kind, const double* A, const double* B, double* C, long n) {
    if (n <= 0) return 0;
    int rc;
    if ((rc = ws_reserve(0, (size_t)n * sizeof(double))) != 0) return rc;
    if ((rc = ws_reserve(1, (size_t)n * sizeof(double))) != 0) return rc;
    if ((rc = ws_reserve(2, (size_t)n * sizeof(double))) != 0) return rc;
    double* dA = (double*)g_ws[0].p;
    double* dB = (double*)g_ws[1].p;
    double* dC = (double*)g_ws[2].p;
    GONN_CHECK(cudaMemcpy(dA, A, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    GONN_CHECK(cudaMemcpy(dB, B, (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    if ((rc = launch_binary(kind, dA, dB, dC, n)) != 0) return rc;
    GONN_CHECK(cudaMemcpy(C, dC, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" void gonn_sync() { cudaDeviceSynchronize(); }

// ---------------------------------------------------------------------------
// Device-resident benchmarks: allocate once, loop the kernel, time with events.
// This is the apples-to-apples comparison vs PyTorch/tinygrad (data already on
// device; only the kernel is timed; no H2D/D2H per call).
// ---------------------------------------------------------------------------

__global__ void addf_kernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

extern "C" double gonn_bench_matmul_dev(int m, int k, int n, int iters, int f32) {
    ensure_handle();
    size_t esz = f32 ? sizeof(float) : sizeof(double);
    void *dA, *dB, *dC;
    if (cudaMalloc(&dA, esz * (size_t)m * k) != cudaSuccess) return -1.0;
    if (cudaMalloc(&dB, esz * (size_t)k * n) != cudaSuccess) { cudaFree(dA); return -1.0; }
    if (cudaMalloc(&dC, esz * (size_t)m * n) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return -1.0; }
    cudaMemset(dA, 1, esz * (size_t)m * k);
    cudaMemset(dB, 1, esz * (size_t)k * n);

    const float  a32 = 1.0f, b32 = 0.0f;
    const double a64 = 1.0,  b64 = 0.0;
    // cuBLAS is column-major: compute C^T = B^T * A^T so the row-major result
    // lands in dC (same trick as gonn_gemm).
    #define GEMM_ONCE()                                                              \
        do {                                                                         \
            if (f32) cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,        \
                                 &a32, (const float*)dB, n, (const float*)dA, k,     \
                                 &b32, (float*)dC, n);                               \
            else     cublasDgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,        \
                                 &a64, (const double*)dB, n, (const double*)dA, k,   \
                                 &b64, (double*)dC, n);                              \
        } while (0)

    for (int i = 0; i < 5; i++) GEMM_ONCE();   // warmup
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) GEMM_ONCE();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    #undef GEMM_ONCE
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (double)ms / iters;
}

// ---------------------------------------------------------------------------
// Fused flash-attention-style forward, float64 (one launch, online softmax,
// NO S*S score matrix materialized). One thread per query row; keys streamed.
// Q,K,V,O are (BH, S, d) row-major. d <= 128. scale typically 1/sqrt(d).
// ---------------------------------------------------------------------------
// Generic fallback (one thread per query row) for d != 64.
__global__ void flash_attn_f64_kernel(const double* Q, const double* K, const double* V,
                                      double* O, double* L, int BH, int S, int d, double scale, int causal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BH * S) return;
    int bh = row / S, i = row % S;
    const double* q  = Q + (size_t)row * d;
    const double* Kb = K + (size_t)(bh * S) * d;
    const double* Vb = V + (size_t)(bh * S) * d;
    double* o = O + (size_t)row * d;
    double ql[128], acc[128];
    for (int t = 0; t < d; t++) { ql[t] = q[t]; acc[t] = 0.0; }
    double m = -CUDART_INF, l = 0.0;
    int jmax = causal ? (i + 1) : S;
    for (int j = 0; j < jmax; j++) {
        const double* k = Kb + (size_t)j * d;
        double s = 0.0;
        for (int t = 0; t < d; t++) s += ql[t] * k[t];
        s *= scale;
        double m_new = fmax(m, s);
        double corr = exp(m - m_new);
        double p = exp(s - m_new);
        const double* v = Vb + (size_t)j * d;
        for (int t = 0; t < d; t++) acc[t] = acc[t] * corr + p * v[t];
        l = l * corr + p;
        m = m_new;
    }
    double inv = 1.0 / l;
    for (int t = 0; t < d; t++) o[t] = acc[t] * inv;
    if (L) L[row] = m + log(l);
}

// Backward: given Q,K,V,O,L (from forward) and dO, accumulate dQ,dK,dV.
// One thread per query row; dQ is race-free per row, dK/dV use fp64 atomicAdd
// (sm_60+). Uses the standard FlashAttention backward identities:
//   p_ij   = exp(scale*q_i.k_j - L_i)
//   D_i    = dO_i . O_i
//   dV_j  += p_ij * dO_i
//   dS_ij  = p_ij * ((dO_i.v_j) - D_i) * scale
//   dQ_i  += dS_ij * k_j ;  dK_j += dS_ij * q_i
__global__ void flash_attn_bwd_f64_kernel(
        const double* Q, const double* K, const double* V,
        const double* O, const double* L, const double* dO,
        double* dQ, double* dK, double* dV,
        int BH, int S, int d, double scale, int causal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BH * S) return;
    int bh = row / S, i = row % S;
    const double* q   = Q  + (size_t)row * d;
    const double* dOi = dO + (size_t)row * d;
    const double* Oi  = O  + (size_t)row * d;
    const double* Kb  = K  + (size_t)(bh * S) * d;
    const double* Vb  = V  + (size_t)(bh * S) * d;
    double* dKb = dK + (size_t)(bh * S) * d;
    double* dVb = dV + (size_t)(bh * S) * d;
    double* dq  = dQ + (size_t)row * d;

    double Li = L[row];
    double ql[128], dol[128], dqloc[128];
    double Di = 0.0;
    for (int t = 0; t < d; t++) {
        ql[t] = q[t];
        dol[t] = dOi[t];
        dqloc[t] = 0.0;
        Di += dOi[t] * Oi[t];
    }

    int jmax = causal ? (i + 1) : S;
    for (int j = 0; j < jmax; j++) {
        const double* k = Kb + (size_t)j * d;
        const double* v = Vb + (size_t)j * d;
        double s = 0.0, dp = 0.0;
        for (int t = 0; t < d; t++) { s += ql[t] * k[t]; dp += dol[t] * v[t]; }
        double p = exp(s * scale - Li);
        double ds = p * (dp - Di) * scale;
        for (int t = 0; t < d; t++) {
            atomicAdd(&dVb[(size_t)j * d + t], p * dol[t]);
            dqloc[t] += ds * k[t];
            atomicAdd(&dKb[(size_t)j * d + t], ds * ql[t]);
        }
    }
    for (int t = 0; t < d; t++) dq[t] = dqloc[t];
}

// Tiled flash attention (d == 64): a block of TILE_Q queries from the same
// (bh) cooperatively streams K/V in tiles through shared memory, so each K[j]
// and V[j] is read from global once per block instead of once per query. Each
// thread owns one query row and keeps its d=64 accumulator in registers.
#define TILE_Q 64
#define TILE_K 16
__global__ void flash_attn_f64_tiled(const double* Q, const double* K, const double* V,
                                     double* O, double* L, int BH, int S, double scale, int causal) {
    const int d = 64;
    int bh = blockIdx.y;
    int q0 = blockIdx.x * TILE_Q;
    int tid = threadIdx.x;            // 0..TILE_Q-1, one query per thread
    int i = q0 + tid;
    bool active = i < S;

    const double* Kb = K + (size_t)(bh * S) * d;
    const double* Vb = V + (size_t)(bh * S) * d;

    __shared__ double sK[TILE_K][64];
    __shared__ double sV[TILE_K][64];

    double q[64];
    double acc[64];
    if (active) {
        const double* qp = Q + (size_t)(bh * S + i) * d;
        for (int t = 0; t < d; t++) { q[t] = qp[t]; acc[t] = 0.0; }
    }
    double m = -CUDART_INF, l = 0.0;
    int jmax = causal ? (q0 + TILE_Q) : S;   // upper bound any thread in block needs
    if (jmax > S) jmax = S;

    for (int j0 = 0; j0 < jmax; j0 += TILE_K) {
        int tk = TILE_K;
        if (j0 + tk > S) tk = S - j0;
        // cooperative load of K/V tile into shared (TILE_Q threads, TILE_K*64 doubles)
        for (int idx = tid; idx < tk * d; idx += TILE_Q) {
            sK[idx / d][idx % d] = Kb[(size_t)(j0) * d + idx];
            sV[idx / d][idx % d] = Vb[(size_t)(j0) * d + idx];
        }
        __syncthreads();
        if (active) {
            for (int jj = 0; jj < tk; jj++) {
                int j = j0 + jj;
                if (causal && j > i) break;
                double s = 0.0;
                #pragma unroll
                for (int t = 0; t < 64; t++) s += q[t] * sK[jj][t];
                s *= scale;
                double m_new = fmax(m, s);
                double corr = exp(m - m_new);
                double p = exp(s - m_new);
                #pragma unroll
                for (int t = 0; t < 64; t++) acc[t] = acc[t] * corr + p * sV[jj][t];
                l = l * corr + p;
                m = m_new;
            }
        }
        __syncthreads();
    }
    if (active) {
        double inv = 1.0 / l;
        double* o = O + (size_t)(bh * S + i) * d;
        for (int t = 0; t < d; t++) o[t] = acc[t] * inv;
        if (L) L[(size_t)bh * S + i] = m + log(l); // logsumexp, saved for backward
    }
}

// Launch helper: pick the tiled kernel for d == 64, else the generic fallback.
// L (logsumexp per query row, BH*S) may be null when not needed (inference).
static void launch_flash_f64(const double* Q, const double* K, const double* V, double* O,
                             double* L, int BH, int S, int d, double scale, int causal) {
    if (d == 64) {
        dim3 grid((S + TILE_Q - 1) / TILE_Q, BH);
        flash_attn_f64_tiled<<<grid, TILE_Q>>>(Q, K, V, O, L, BH, S, scale, causal);
    } else {
        int threads = 128, blocks = (BH * S + threads - 1) / threads;
        flash_attn_f64_kernel<<<blocks, threads>>>(Q, K, V, O, L, BH, S, d, scale, causal);
    }
}

// Run once on host pointers (H2D, launch, D2H) — for correctness checks.
extern "C" void gonn_flash_attn_f64(const double* Q, const double* K, const double* V,
                                    double* O, int BH, int S, int d, double scale, int causal) {
    size_t qn = (size_t)BH * S * d;
    double *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, sizeof(double) * qn);
    cudaMalloc(&dK, sizeof(double) * qn);
    cudaMalloc(&dV, sizeof(double) * qn);
    cudaMalloc(&dO, sizeof(double) * qn);
    cudaMemcpy(dQ, Q, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeof(double) * qn, cudaMemcpyHostToDevice);
    launch_flash_f64(dQ, dK, dV, dO, nullptr, BH, S, d, scale, causal);
    cudaMemcpy(O, dO, sizeof(double) * qn, cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

// Training forward: also returns L (logsumexp per query row, BH*S) for backward.
extern "C" void gonn_flash_attn_f64_fwd(const double* Q, const double* K, const double* V,
                                        double* O, double* L, int BH, int S, int d,
                                        double scale, int causal) {
    size_t qn = (size_t)BH * S * d, ln = (size_t)BH * S;
    double *dQ, *dK, *dV, *dO, *dL;
    cudaMalloc(&dQ, sizeof(double) * qn); cudaMalloc(&dK, sizeof(double) * qn);
    cudaMalloc(&dV, sizeof(double) * qn); cudaMalloc(&dO, sizeof(double) * qn);
    cudaMalloc(&dL, sizeof(double) * ln);
    cudaMemcpy(dQ, Q, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, sizeof(double) * qn, cudaMemcpyHostToDevice);
    launch_flash_f64(dQ, dK, dV, dO, dL, BH, S, d, scale, causal);
    cudaMemcpy(O, dO, sizeof(double) * qn, cudaMemcpyDeviceToHost);
    cudaMemcpy(L, dL, sizeof(double) * ln, cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dL);
}

// Backward: compute dQ,dK,dV from Q,K,V,O,L,dO.
extern "C" void gonn_flash_attn_f64_bwd(const double* Q, const double* K, const double* V,
                                        const double* O, const double* L, const double* dO,
                                        double* dQ, double* dK, double* dV,
                                        int BH, int S, int d, double scale, int causal) {
    size_t qn = (size_t)BH * S * d, ln = (size_t)BH * S;
    double *Qd, *Kd, *Vd, *Od, *Ld, *dOd, *dQd, *dKd, *dVd;
    cudaMalloc(&Qd, sizeof(double) * qn); cudaMalloc(&Kd, sizeof(double) * qn);
    cudaMalloc(&Vd, sizeof(double) * qn); cudaMalloc(&Od, sizeof(double) * qn);
    cudaMalloc(&Ld, sizeof(double) * ln); cudaMalloc(&dOd, sizeof(double) * qn);
    cudaMalloc(&dQd, sizeof(double) * qn); cudaMalloc(&dKd, sizeof(double) * qn);
    cudaMalloc(&dVd, sizeof(double) * qn);
    cudaMemcpy(Qd, Q, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(Kd, K, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(Vd, V, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(Od, O, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemcpy(Ld, L, sizeof(double) * ln, cudaMemcpyHostToDevice);
    cudaMemcpy(dOd, dO, sizeof(double) * qn, cudaMemcpyHostToDevice);
    cudaMemset(dQd, 0, sizeof(double) * qn);
    cudaMemset(dKd, 0, sizeof(double) * qn);
    cudaMemset(dVd, 0, sizeof(double) * qn);
    int threads = 128, blocks = (BH * S + threads - 1) / threads;
    flash_attn_bwd_f64_kernel<<<blocks, threads>>>(Qd, Kd, Vd, Od, Ld, dOd,
                                                   dQd, dKd, dVd, BH, S, d, scale, causal);
    cudaMemcpy(dQ, dQd, sizeof(double) * qn, cudaMemcpyDeviceToHost);
    cudaMemcpy(dK, dKd, sizeof(double) * qn, cudaMemcpyDeviceToHost);
    cudaMemcpy(dV, dVd, sizeof(double) * qn, cudaMemcpyDeviceToHost);
    cudaFree(Qd); cudaFree(Kd); cudaFree(Vd); cudaFree(Od); cudaFree(Ld);
    cudaFree(dOd); cudaFree(dQd); cudaFree(dKd); cudaFree(dVd);
}

// Device-resident, CUDA-event-timed benchmark. Returns avg ms/iter.
extern "C" double gonn_bench_flash_attn_f64(int BH, int S, int d, int iters, int causal) {
    size_t qn = (size_t)BH * S * d;
    double *dQ, *dK, *dV, *dO;
    if (cudaMalloc(&dQ, sizeof(double) * qn) != cudaSuccess) return -1.0;
    if (cudaMalloc(&dK, sizeof(double) * qn) != cudaSuccess) { cudaFree(dQ); return -1.0; }
    if (cudaMalloc(&dV, sizeof(double) * qn) != cudaSuccess) { cudaFree(dQ); cudaFree(dK); return -1.0; }
    if (cudaMalloc(&dO, sizeof(double) * qn) != cudaSuccess) { cudaFree(dQ); cudaFree(dK); cudaFree(dV); return -1.0; }
    cudaMemset(dQ, 0, sizeof(double) * qn);
    cudaMemset(dK, 0, sizeof(double) * qn);
    cudaMemset(dV, 0, sizeof(double) * qn);
    double scale = 1.0 / sqrt((double)d);

    for (int i = 0; i < 5; i++)
        launch_flash_f64(dQ, dK, dV, dO, nullptr, BH, S, d, scale, causal);
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++)
        launch_flash_f64(dQ, dK, dV, dO, nullptr, BH, S, d, scale, causal);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return (double)ms / iters;
}

extern "C" double gonn_bench_add_dev(int n, int iters, int f32) {
    size_t esz = f32 ? sizeof(float) : sizeof(double);
    void *dA, *dB, *dC;
    if (cudaMalloc(&dA, esz * (size_t)n) != cudaSuccess) return -1.0;
    if (cudaMalloc(&dB, esz * (size_t)n) != cudaSuccess) { cudaFree(dA); return -1.0; }
    if (cudaMalloc(&dC, esz * (size_t)n) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return -1.0; }
    cudaMemset(dA, 1, esz * (size_t)n);
    cudaMemset(dB, 1, esz * (size_t)n);
    int threads = 256, blocks = (n + threads - 1) / threads;

    for (int i = 0; i < 5; i++) {
        if (f32) addf_kernel<<<blocks, threads>>>((float*)dA, (float*)dB, (float*)dC, n);
        else     add_kernel<<<blocks, threads>>>((double*)dA, (double*)dB, (double*)dC, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        if (f32) addf_kernel<<<blocks, threads>>>((float*)dA, (float*)dB, (float*)dC, n);
        else     add_kernel<<<blocks, threads>>>((double*)dA, (double*)dB, (double*)dC, n);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (double)ms / iters;
}

// ===========================================================================
// Device-resident buffers: allocate once on the GPU, run a chain of ops, copy
// back once. Eliminates the per-call H2D/D2H that the eager backend pays.
// Pointers are returned to Go as void* (unsafe.Pointer). f64 path + an fp16
// tensor-core GEMM path (cublasGemmEx) for tensor-core throughput.
// ===========================================================================

extern "C" void* gonn_dev_alloc(long bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, (size_t)bytes) != cudaSuccess) return nullptr;
    return p;
}
extern "C" void gonn_dev_free(void* p) { if (p) cudaFree(p); }
extern "C" void gonn_dev_upload(void* dst, const double* src, long n) {
    cudaMemcpy(dst, src, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice);
}
extern "C" void gonn_dev_download(double* dst, const void* src, long n) {
    cudaMemcpy(dst, src, sizeof(double) * (size_t)n, cudaMemcpyDeviceToHost);
}
extern "C" void gonn_dev_sync(void) { cudaDeviceSynchronize(); }

// C(m,n) = A(m,k) * B(k,n), all device-resident f64. (cuBLAS column-major trick.)
extern "C" void gonn_dev_matmul_f64(const void* dA, const void* dB, void* dC, int m, int k, int n) {
    ensure_handle();
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                &alpha, (const double*)dB, n, (const double*)dA, k,
                &beta, (double*)dC, n);
}
extern "C" void gonn_dev_add_f64(const void* dA, const void* dB, void* dC, int n) {
    int t = 256, b = (n + t - 1) / t;
    add_kernel<<<b, t>>>((const double*)dA, (const double*)dB, (double*)dC, n);
}
extern "C" void gonn_dev_relu_f64(void* dA, int n) {
    int t = 256, b = (n + t - 1) / t;
    relu_kernel<<<b, t>>>((const double*)dA, (double*)dA, n); // in place
}

// ---- fp16 tensor-core path -------------------------------------------------
__global__ void d2h_kernel(const double* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __double2half(in[i]);
}
__global__ void h2d_kernel(const __half* in, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (double)__half2float(in[i]);
}

// Allocate n halfs and fill from host doubles (converted on device).
extern "C" void* gonn_dev_upload_f16(const double* src, int n) {
    void* dHalf = nullptr;
    if (cudaMalloc(&dHalf, sizeof(__half) * (size_t)n) != cudaSuccess) return nullptr;
    double* tmp = nullptr;
    cudaMalloc(&tmp, sizeof(double) * (size_t)n);
    cudaMemcpy(tmp, src, sizeof(double) * (size_t)n, cudaMemcpyHostToDevice);
    int t = 256, b = (n + t - 1) / t;
    d2h_kernel<<<b, t>>>(tmp, (__half*)dHalf, n);
    cudaFree(tmp);
    return dHalf;
}
extern "C" void gonn_dev_download_f16(double* dst, const void* dHalf, int n) {
    double* tmp = nullptr;
    cudaMalloc(&tmp, sizeof(double) * (size_t)n);
    int t = 256, b = (n + t - 1) / t;
    h2d_kernel<<<b, t>>>((const __half*)dHalf, tmp, n);
    cudaMemcpy(dst, tmp, sizeof(double) * (size_t)n, cudaMemcpyDeviceToHost);
    cudaFree(tmp);
}
// fp16 GEMM with f32 accumulate on tensor cores. half in/out, device-resident.
extern "C" void gonn_dev_matmul_f16(const void* dA, const void* dB, void* dC, int m, int k, int n) {
    ensure_handle();
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                 &alpha, dB, CUDA_R_16F, n, dA, CUDA_R_16F, k,
                 &beta, dC, CUDA_R_16F, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Device-resident fp16 tensor-core GEMM benchmark; returns avg ms/iter.
extern "C" double gonn_bench_matmul_f16_dev(int m, int k, int n, int iters) {
    void *dA, *dB, *dC;
    if (cudaMalloc(&dA, sizeof(__half) * (size_t)m * k) != cudaSuccess) return -1.0;
    if (cudaMalloc(&dB, sizeof(__half) * (size_t)k * n) != cudaSuccess) { cudaFree(dA); return -1.0; }
    if (cudaMalloc(&dC, sizeof(__half) * (size_t)m * n) != cudaSuccess) { cudaFree(dA); cudaFree(dB); return -1.0; }
    cudaMemset(dA, 0, sizeof(__half) * (size_t)m * k);
    cudaMemset(dB, 0, sizeof(__half) * (size_t)k * n);
    for (int i = 0; i < 5; i++) gonn_dev_matmul_f16(dA, dB, dC, m, k, n);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) gonn_dev_matmul_f16(dA, dB, dC, m, k, n);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return (double)ms / iters;
}
