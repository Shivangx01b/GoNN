// gonn_opencl.c — OpenCL backend kernels + host glue for GoNN.
//
// Build is driven by CGO (see opencl.go) with -tags opencl. The OpenCL ICD and
// headers ship with the CUDA toolkit (e.g. /usr/local/cuda/{include,lib64}).
// fp64 is enabled via cl_khr_fp64 (supported by NVIDIA OpenCL on the RTX 3060).
//
// Design mirrors the CUDA backend: enum-dispatched unary/binary elementwise
// kernels plus a batched GEMM with trans flags. Kernel expressions are kept
// identical to the CUDA ones (numerics parity). Host staging is per-call
// buffer create/release (correctness first; a workspace cache like the CUDA
// side's is possible later — OpenCL is the portability path, not the perf
// path). Kernel objects are created once per kind and cached.

#define CL_TARGET_OPENCL_VERSION 120
#include "gonn_opencl.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Elementwise kernel sources are string-pasted from one template per arity so
// each op's expression appears exactly once.
#define UN_K(NAME, EXPR) \
    "__kernel void " NAME "(__global const double* A,__global double* C,int n)" \
    "{int i=get_global_id(0); if(i<n){double x=A[i]; C[i]=" EXPR ";}}\n"
#define BIN_K(NAME, EXPR) \
    "__kernel void " NAME "(__global const double* A,__global const double* B,__global double* C,int n)" \
    "{int i=get_global_id(0); if(i<n){double a=A[i]; double b=B[i]; C[i]=" EXPR ";}}\n"

static const char* KERNEL_SRC =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
UN_K("reluk",    "x>0.0?x:0.0")
UN_K("sigmoidk", "1.0/(1.0+exp(-x))")
UN_K("tanhk",    "tanh(x)")
UN_K("expk",     "exp(x)")
UN_K("logk",     "log(x)")
UN_K("geluk",    "0.5*x*(1.0+tanh(0.7978845608028654*(x+0.044715*x*x*x)))")
UN_K("siluk",    "x/(1.0+exp(-x))")
BIN_K("addk", "a+b")
BIN_K("subk", "a-b")
BIN_K("mulk", "a*b")
BIN_K("divk", "a/b")
// Batched row-major GEMM with trans flags. NDRange: (n, m, batch).
// op(A) is (m,k): A[bi] stored (m,k), or (k,m) when transA (so op(A)[i,p] =
// A[p*m+i]); op(B) is (k,n): stored (k,n), or (n,k) when transB.
"__kernel void gemmk(__global const double* A,__global const double* B,__global double* C,\n"
"                    int m,int k,int n,int transA,int transB){\n"
"  int col=get_global_id(0); int row=get_global_id(1); int bi=get_global_id(2);\n"
"  if(row<m && col<n){\n"
"    __global const double* Ab=A+(long)bi*m*k;\n"
"    __global const double* Bb=B+(long)bi*k*n;\n"
"    double s=0.0;\n"
"    for(int p=0;p<k;p++){\n"
"      double av = transA ? Ab[p*m+row] : Ab[row*k+p];\n"
"      double bv = transB ? Bb[col*k+p] : Bb[p*n+col];\n"
"      s+=av*bv;\n"
"    }\n"
"    C[(long)bi*m*n+row*n+col]=s;\n"
"  }\n"
"}\n";

static cl_context       g_ctx;
static cl_command_queue g_queue;
static cl_program       g_prog;
static cl_device_id     g_dev;
static int              g_init = 0;   // 0=untried, 1=ok, -1=failed

static int init_cl(void) {
    if (g_init != 0) return g_init;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS || nplat == 0) { g_init = -1; return -1; }
    cl_platform_id* plats = (cl_platform_id*)malloc(sizeof(cl_platform_id) * nplat);
    clGetPlatformIDs(nplat, plats, NULL);

    // Prefer a GPU device; fall back to any OpenCL device (e.g. a CPU runtime
    // such as POCL) so the backend is portable and verifiable anywhere.
    cl_int err = CL_DEVICE_NOT_FOUND;
    for (cl_uint i = 0; i < nplat && err != CL_SUCCESS; i++) {
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &g_dev, NULL) == CL_SUCCESS)
            err = CL_SUCCESS;
    }
    for (cl_uint i = 0; i < nplat && err != CL_SUCCESS; i++) {
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 1, &g_dev, NULL) == CL_SUCCESS)
            err = CL_SUCCESS;
    }
    free(plats);
    if (err != CL_SUCCESS) { g_init = -1; return -1; }

    g_ctx = clCreateContext(NULL, 1, &g_dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) { g_init = -1; return -1; }
    g_queue = clCreateCommandQueue(g_ctx, g_dev, 0, &err);
    if (err != CL_SUCCESS) { g_init = -1; return -1; }

    const char* src = KERNEL_SRC;
    g_prog = clCreateProgramWithSource(g_ctx, 1, &src, NULL, &err);
    if (err != CL_SUCCESS) { g_init = -1; return -1; }
    err = clBuildProgram(g_prog, 1, &g_dev, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        char* log = (char*)malloc(logsz + 1);
        clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, logsz, log, NULL);
        log[logsz] = 0;
        fprintf(stderr, "gonn_opencl: program build failed:\n%s\n", log);
        free(log);
        g_init = -1;
        return -1;
    }
    g_init = 1;
    return 1;
}

int gonn_cl_available(void) { return init_cl() == 1; }

// Kind -> kernel name tables (indexes match the enums in gonn_opencl.h).
static const char* UNARY_NAMES[]  = {"reluk", "sigmoidk", "tanhk", "expk", "logk", "geluk", "siluk"};
static const char* BINARY_NAMES[] = {"addk", "subk", "mulk", "divk"};
#define N_UNARY  ((int)(sizeof(UNARY_NAMES) / sizeof(UNARY_NAMES[0])))
#define N_BINARY ((int)(sizeof(BINARY_NAMES) / sizeof(BINARY_NAMES[0])))

// Kernel objects, created lazily once per kind and cached. Not thread-safe —
// the Go side serializes entry with a mutex.
static cl_kernel g_unary_k[N_UNARY];
static cl_kernel g_binary_k[N_BINARY];
static cl_kernel g_gemm_k;

static cl_kernel get_kernel(cl_kernel* slot, const char* name, cl_int* err) {
    if (*slot) { *err = CL_SUCCESS; return *slot; }
    *slot = clCreateKernel(g_prog, name, err);
    return *slot;
}

static cl_mem buf_in(const double* p, size_t n, cl_int* err) {
    return clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(double) * n, (void*)p, err);
}
static cl_mem buf_out(size_t n, cl_int* err) {
    return clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, sizeof(double) * n, NULL, err);
}

int gonn_cl_unary(int kind, const double* A, double* C, int n) {
    if (init_cl() != 1) return -1;
    if (n <= 0) return 0;
    if (kind < 0 || kind >= N_UNARY) {
        fprintf(stderr, "gonn_opencl: unknown unary kind %d\n", kind);
        return -1;
    }
    cl_int err;
    cl_kernel k = get_kernel(&g_unary_k[kind], UNARY_NAMES[kind], &err);
    if (err != CL_SUCCESS) return (int)err;
    cl_mem bA = buf_in(A, n, &err);
    if (err != CL_SUCCESS) return (int)err;
    cl_mem bC = buf_out(n, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(bA); return (int)err; }
    clSetKernelArg(k, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(k, 1, sizeof(cl_mem), &bC);
    clSetKernelArg(k, 2, sizeof(int), &n);
    size_t lws = 256, gws = ((size_t)(n + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * n, C, 0, NULL, NULL);
    clReleaseMemObject(bA); clReleaseMemObject(bC);
    return (int)err;
}

int gonn_cl_binary(int kind, const double* A, const double* B, double* C, int n) {
    if (init_cl() != 1) return -1;
    if (n <= 0) return 0;
    if (kind < 0 || kind >= N_BINARY) {
        fprintf(stderr, "gonn_opencl: unknown binary kind %d\n", kind);
        return -1;
    }
    cl_int err;
    cl_kernel k = get_kernel(&g_binary_k[kind], BINARY_NAMES[kind], &err);
    if (err != CL_SUCCESS) return (int)err;
    cl_mem bA = buf_in(A, n, &err);
    if (err != CL_SUCCESS) return (int)err;
    cl_mem bB = buf_in(B, n, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(bA); return (int)err; }
    cl_mem bC = buf_out(n, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(bA); clReleaseMemObject(bB); return (int)err; }
    clSetKernelArg(k, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(k, 1, sizeof(cl_mem), &bB);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bC);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t lws = 256, gws = ((size_t)(n + 255) / 256) * 256;
    err = clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * n, C, 0, NULL, NULL);
    clReleaseMemObject(bA); clReleaseMemObject(bB); clReleaseMemObject(bC);
    return (int)err;
}

int gonn_cl_gemm(const double* A, const double* B, double* C,
                 int batch, int m, int k, int n, int transA, int transB) {
    if (init_cl() != 1) return -1;
    if (batch <= 0 || m <= 0 || n <= 0 || k <= 0) return 0;
    size_t an = (size_t)batch * m * k, bn = (size_t)batch * k * n, cn = (size_t)batch * m * n;
    cl_int err;
    cl_kernel ker = get_kernel(&g_gemm_k, "gemmk", &err);
    if (err != CL_SUCCESS) return (int)err;
    cl_mem bA = buf_in(A, an, &err);
    if (err != CL_SUCCESS) return (int)err;
    cl_mem bB = buf_in(B, bn, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(bA); return (int)err; }
    cl_mem bC = buf_out(cn, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(bA); clReleaseMemObject(bB); return (int)err; }
    clSetKernelArg(ker, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(ker, 1, sizeof(cl_mem), &bB);
    clSetKernelArg(ker, 2, sizeof(cl_mem), &bC);
    clSetKernelArg(ker, 3, sizeof(int), &m);
    clSetKernelArg(ker, 4, sizeof(int), &k);
    clSetKernelArg(ker, 5, sizeof(int), &n);
    clSetKernelArg(ker, 6, sizeof(int), &transA);
    clSetKernelArg(ker, 7, sizeof(int), &transB);
    size_t lws[3] = {16, 16, 1};
    size_t gws[3] = {((size_t)(n + 15) / 16) * 16, ((size_t)(m + 15) / 16) * 16, (size_t)batch};
    err = clEnqueueNDRangeKernel(g_queue, ker, 3, NULL, gws, lws, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * cn, C, 0, NULL, NULL);
    clReleaseMemObject(bA); clReleaseMemObject(bB); clReleaseMemObject(bC);
    return (int)err;
}

void gonn_cl_sync(void) { if (g_init == 1) clFinish(g_queue); }
