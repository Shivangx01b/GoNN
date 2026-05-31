// gonn_opencl.c — OpenCL backend kernels + host glue for GoNN.
//
// Build is driven by CGO (see opencl.go) with -tags opencl. The OpenCL ICD and
// headers ship with the CUDA toolkit (e.g. /usr/local/cuda/{include,lib64}).
// fp64 is enabled via cl_khr_fp64 (supported by NVIDIA OpenCL on the RTX 3060).
//
// Design mirrors the CUDA backend: per-call host->device copy, kernel launch,
// device->host copy (no buffer caching yet — correctness first).

#define CL_TARGET_OPENCL_VERSION 120
#include "gonn_opencl.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char* KERNEL_SRC =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void addk(__global const double* A,__global const double* B,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=A[i]+B[i];}\n"
"__kernel void mulk(__global const double* A,__global const double* B,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=A[i]*B[i];}\n"
"__kernel void subk(__global const double* A,__global const double* B,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=A[i]-B[i];}\n"
"__kernel void divk(__global const double* A,__global const double* B,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=A[i]/B[i];}\n"
"__kernel void scalek(__global const double* A,double s,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=A[i]*s;}\n"
"__kernel void axpyk(__global double* OUT,__global const double* X,double a,int n){int i=get_global_id(0); if(i<n) OUT[i]+=a*X[i];}\n"
"__kernel void reluk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n){double v=A[i]; C[i]=v>0.0?v:0.0;}}\n"
"__kernel void sigmoidk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=1.0/(1.0+exp(-A[i]));}\n"
"__kernel void tanhk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=tanh(A[i]);}\n"
"__kernel void expk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=exp(A[i]);}\n"
"__kernel void logk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n) C[i]=log(A[i]);}\n"
"__kernel void geluk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n){double x=A[i]; double in=0.7978845608028654*(x+0.044715*x*x*x); C[i]=0.5*x*(1.0+tanh(in));}}\n"
"__kernel void siluk(__global const double* A,__global double* C,int n){int i=get_global_id(0); if(i<n){double x=A[i]; C[i]=x/(1.0+exp(-x));}}\n"
"__kernel void matmulk(__global const double* A,__global const double* B,__global double* C,int m,int k,int n){\n"
"  int col=get_global_id(0); int row=get_global_id(1);\n"
"  if(row<m && col<n){ double s=0.0; for(int p=0;p<k;p++) s+=A[row*k+p]*B[p*n+col]; C[row*n+col]=s; }\n"
"}\n"
"__kernel void reduce_sum(__global const double* A,__global double* out,int n,__local double* sd){\n"
"  int t=get_local_id(0); int nt=get_local_size(0); double acc=0.0;\n"
"  for(int i=t;i<n;i+=nt) acc+=A[i]; sd[t]=acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int s=nt/2;s>0;s>>=1){ if(t<s) sd[t]+=sd[t+s]; barrier(CLK_LOCAL_MEM_FENCE);} if(t==0) out[0]=sd[0];\n"
"}\n"
"__kernel void reduce_max(__global const double* A,__global double* out,int n,__local double* sd){\n"
"  int t=get_local_id(0); int nt=get_local_size(0); double acc=-INFINITY;\n"
"  for(int i=t;i<n;i+=nt){ double v=A[i]; if(v>acc) acc=v; } sd[t]=acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int s=nt/2;s>0;s>>=1){ if(t<s){ double o=sd[t+s]; if(o>sd[t]) sd[t]=o; } barrier(CLK_LOCAL_MEM_FENCE);} if(t==0) out[0]=sd[0];\n"
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

static cl_mem buf_in(const double* p, int n) {
    return clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(double) * n, (void*)p, NULL);
}
static cl_mem buf_out(int n) {
    return clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, sizeof(double) * n, NULL, NULL);
}

static void run1d(const char* name, cl_mem* args, int nargs, int n) {
    cl_kernel k = clCreateKernel(g_prog, name, NULL);
    for (int i = 0; i < nargs; i++) clSetKernelArg(k, i, sizeof(cl_mem), &args[i]);
    clSetKernelArg(k, nargs, sizeof(int), &n);
    size_t lws = 256, gws = ((n + lws - 1) / lws) * lws;
    clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    clReleaseKernel(k);
}

static void binary(const char* name, const double* A, const double* B, double* C, int n) {
    if (init_cl() != 1 || n == 0) return;
    cl_mem bA = buf_in(A, n), bB = buf_in(B, n), bC = buf_out(n);
    cl_mem a[3] = {bA, bB, bC};
    run1d(name, a, 3, n);
    clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * n, C, 0, NULL, NULL);
    clReleaseMemObject(bA); clReleaseMemObject(bB); clReleaseMemObject(bC);
}

static void unary(const char* name, const double* A, double* C, int n) {
    if (init_cl() != 1 || n == 0) return;
    cl_mem bA = buf_in(A, n), bC = buf_out(n);
    cl_mem a[2] = {bA, bC};
    run1d(name, a, 2, n);
    clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * n, C, 0, NULL, NULL);
    clReleaseMemObject(bA); clReleaseMemObject(bC);
}

void gonn_cl_add(const double* A, const double* B, double* C, int n) { binary("addk", A, B, C, n); }
void gonn_cl_mul(const double* A, const double* B, double* C, int n) { binary("mulk", A, B, C, n); }
void gonn_cl_sub(const double* A, const double* B, double* C, int n) { binary("subk", A, B, C, n); }
void gonn_cl_div(const double* A, const double* B, double* C, int n) { binary("divk", A, B, C, n); }

void gonn_cl_relu(const double* A, double* C, int n)    { unary("reluk", A, C, n); }
void gonn_cl_sigmoid(const double* A, double* C, int n) { unary("sigmoidk", A, C, n); }
void gonn_cl_tanh(const double* A, double* C, int n)    { unary("tanhk", A, C, n); }
void gonn_cl_exp(const double* A, double* C, int n)     { unary("expk", A, C, n); }
void gonn_cl_log(const double* A, double* C, int n)     { unary("logk", A, C, n); }
void gonn_cl_gelu(const double* A, double* C, int n)    { unary("geluk", A, C, n); }
void gonn_cl_silu(const double* A, double* C, int n)    { unary("siluk", A, C, n); }

void gonn_cl_scale(const double* A, double s, double* C, int n) {
    if (init_cl() != 1 || n == 0) return;
    cl_mem bA = buf_in(A, n), bC = buf_out(n);
    cl_kernel k = clCreateKernel(g_prog, "scalek", NULL);
    clSetKernelArg(k, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(k, 1, sizeof(double), &s);
    clSetKernelArg(k, 2, sizeof(cl_mem), &bC);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t lws = 256, gws = ((n + lws - 1) / lws) * lws;
    clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * n, C, 0, NULL, NULL);
    clReleaseKernel(k); clReleaseMemObject(bA); clReleaseMemObject(bC);
}

void gonn_cl_axpy(double* OUT, const double* X, double alpha, int n) {
    if (init_cl() != 1 || n == 0) return;
    cl_mem bO = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(double) * n, OUT, NULL);
    cl_mem bX = buf_in(X, n);
    cl_kernel k = clCreateKernel(g_prog, "axpyk", NULL);
    clSetKernelArg(k, 0, sizeof(cl_mem), &bO);
    clSetKernelArg(k, 1, sizeof(cl_mem), &bX);
    clSetKernelArg(k, 2, sizeof(double), &alpha);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t lws = 256, gws = ((n + lws - 1) / lws) * lws;
    clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, bO, CL_TRUE, 0, sizeof(double) * n, OUT, 0, NULL, NULL);
    clReleaseKernel(k); clReleaseMemObject(bO); clReleaseMemObject(bX);
}

void gonn_cl_matmul(const double* A, const double* B, double* C, int m, int k, int n) {
    if (init_cl() != 1 || m == 0 || n == 0) return;
    cl_mem bA = buf_in(A, m * k), bB = buf_in(B, k * n), bC = buf_out(m * n);
    cl_kernel ker = clCreateKernel(g_prog, "matmulk", NULL);
    clSetKernelArg(ker, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(ker, 1, sizeof(cl_mem), &bB);
    clSetKernelArg(ker, 2, sizeof(cl_mem), &bC);
    clSetKernelArg(ker, 3, sizeof(int), &m);
    clSetKernelArg(ker, 4, sizeof(int), &k);
    clSetKernelArg(ker, 5, sizeof(int), &n);
    size_t lws[2] = {16, 16};
    size_t gws[2] = {((n + 15) / 16) * 16, ((m + 15) / 16) * 16};
    clEnqueueNDRangeKernel(g_queue, ker, 2, NULL, gws, lws, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, bC, CL_TRUE, 0, sizeof(double) * m * n, C, 0, NULL, NULL);
    clReleaseKernel(ker); clReleaseMemObject(bA); clReleaseMemObject(bB); clReleaseMemObject(bC);
}

static void reduce(const char* name, const double* A, double* out, int n, double empty) {
    if (init_cl() != 1) { *out = empty; return; }
    if (n == 0) { *out = empty; return; }
    cl_mem bA = buf_in(A, n);
    cl_mem bO = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, sizeof(double), NULL, NULL);
    cl_kernel k = clCreateKernel(g_prog, name, NULL);
    int nn = n;
    clSetKernelArg(k, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(k, 1, sizeof(cl_mem), &bO);
    clSetKernelArg(k, 2, sizeof(int), &nn);
    clSetKernelArg(k, 3, sizeof(double) * 256, NULL); // local scratch
    size_t lws = 256, gws = 256;
    clEnqueueNDRangeKernel(g_queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    clEnqueueReadBuffer(g_queue, bO, CL_TRUE, 0, sizeof(double), out, 0, NULL, NULL);
    clReleaseKernel(k); clReleaseMemObject(bA); clReleaseMemObject(bO);
}

void gonn_cl_sum(const double* A, double* out_scalar, int n) { reduce("reduce_sum", A, out_scalar, n, 0.0); }
void gonn_cl_max(const double* A, double* out_scalar, int n) { reduce("reduce_max", A, out_scalar, n, -INFINITY); }

void gonn_cl_sync(void) { if (g_init == 1) clFinish(g_queue); }
