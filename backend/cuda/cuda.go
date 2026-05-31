//go:build cuda
// +build cuda

// Package cuda is the CUDA backend, built with `-tags cuda`. It calls into
// gonn_cuda.cpp/cu via CGO. The native side is built on top of the CUDA
// runtime + cuBLAS for matmul.
package cuda

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lgonn_cuda -lcudart -lcublas

#include "gonn_cuda.h"
*/
import "C"

import (
	"math"
	"unsafe"

	"gonn/backend"
)

type cudaBackend struct{}

func (cudaBackend) Name() backend.Device { return backend.CUDA }

func (cudaBackend) MatMul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	C.gonn_matmul(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(m), C.int(k), C.int(n),
	)
	return out
}

func (cudaBackend) AddElem(a, b []float64) []float64 {
	out := make([]float64, len(a))
	C.gonn_add(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	)
	return out
}

func (cudaBackend) MulElem(a, b []float64) []float64 {
	out := make([]float64, len(a))
	C.gonn_mul(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	)
	return out
}

func (cudaBackend) Synchronize() { C.gonn_sync() }

// Elementwise arithmetic ----------------------------------------------------

func (cudaBackend) Sub(a, b []float64) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	C.gonn_sub(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	)
	return out
}

func (cudaBackend) Div(a, b []float64) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	C.gonn_div(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	)
	return out
}

func (cudaBackend) Scale(a []float64, s float64) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	C.gonn_scale(
		(*C.double)(unsafe.Pointer(&a[0])),
		C.double(s),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	)
	return out
}

func (cudaBackend) AxpyInto(out, x []float64, alpha float64) {
	if len(out) == 0 {
		return
	}
	C.gonn_axpy(
		(*C.double)(unsafe.Pointer(&out[0])),
		(*C.double)(unsafe.Pointer(&x[0])),
		C.double(alpha),
		C.int(len(out)),
	)
}

// Reductions ---------------------------------------------------------------

func (cudaBackend) Sum(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	var s C.double
	C.gonn_sum(
		(*C.double)(unsafe.Pointer(&a[0])),
		&s,
		C.int(len(a)),
	)
	return float64(s)
}

func (cudaBackend) Max(a []float64) float64 {
	if len(a) == 0 {
		// Match cpuBackend semantics (-Inf for empty).
		return math.Inf(-1)
	}
	var m C.double
	C.gonn_max(
		(*C.double)(unsafe.Pointer(&a[0])),
		&m,
		C.int(len(a)),
	)
	return float64(m)
}

// Activations / unary math -------------------------------------------------

func unary(a []float64, fn func(in, out *C.double, n C.int)) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	fn(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(len(a)),
	)
	return out
}

func (cudaBackend) ReLU(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_relu(in, out, n) })
}

func (cudaBackend) Sigmoid(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_sigmoid(in, out, n) })
}

func (cudaBackend) Tanh(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_tanh(in, out, n) })
}

func (cudaBackend) Exp(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_exp(in, out, n) })
}

func (cudaBackend) Log(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_log(in, out, n) })
}

func (cudaBackend) GELU(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_gelu(in, out, n) })
}

func (cudaBackend) SiLU(a []float64) []float64 {
	return unary(a, func(in, out *C.double, n C.int) { C.gonn_silu(in, out, n) })
}

// BenchMatMulDev runs a device-resident GEMM benchmark (inputs allocated once
// on the GPU, iters GEMMs, CUDA-event timed). Returns avg ms/iter. f32 selects
// cublasSgemm, else cublasDgemm. This is the apples-to-apples comparison vs
// PyTorch/tinygrad device-resident timing.
func BenchMatMulDev(m, k, n, iters int, f32 bool) float64 {
	var f C.int
	if f32 {
		f = 1
	}
	return float64(C.gonn_bench_matmul_dev(C.int(m), C.int(k), C.int(n), C.int(iters), f))
}

// BenchAddDev runs a device-resident elementwise-add benchmark. Returns avg ms/iter.
func BenchAddDev(n, iters int, f32 bool) float64 {
	var f C.int
	if f32 {
		f = 1
	}
	return float64(C.gonn_bench_add_dev(C.int(n), C.int(iters), f))
}

// FlashAttnF64 runs the fused flash-attention forward on the GPU for inputs
// Q,K,V of shape (BH, S, d) row-major, writing O. scale is usually 1/sqrt(d).
// Used for correctness checks against a CPU reference.
func FlashAttnF64(Q, K, V, O []float64, BH, S, d int, scale float64, causal bool) {
	var c C.int
	if causal {
		c = 1
	}
	C.gonn_flash_attn_f64(
		(*C.double)(unsafe.Pointer(&Q[0])),
		(*C.double)(unsafe.Pointer(&K[0])),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.double)(unsafe.Pointer(&O[0])),
		C.int(BH), C.int(S), C.int(d), C.double(scale), c,
	)
}

// FlashAttnF64Fwd runs the training forward, returning O and L (logsumexp per
// query row, length BH*S) for the backward pass. Q,K,V are flat (BH,S,d).
func FlashAttnF64Fwd(Q, K, V []float64, BH, S, d int, scale float64, causal bool) (O, L []float64) {
	O = make([]float64, len(Q))
	L = make([]float64, BH*S)
	var c C.int
	if causal {
		c = 1
	}
	C.gonn_flash_attn_f64_fwd(
		(*C.double)(unsafe.Pointer(&Q[0])),
		(*C.double)(unsafe.Pointer(&K[0])),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.double)(unsafe.Pointer(&O[0])),
		(*C.double)(unsafe.Pointer(&L[0])),
		C.int(BH), C.int(S), C.int(d), C.double(scale), c,
	)
	return O, L
}

// FlashAttnF64Bwd computes dQ,dK,dV from the saved forward tensors and dO.
func FlashAttnF64Bwd(Q, K, V, O, L, dO []float64, BH, S, d int, scale float64, causal bool) (dQ, dK, dV []float64) {
	dQ = make([]float64, len(Q))
	dK = make([]float64, len(K))
	dV = make([]float64, len(V))
	var c C.int
	if causal {
		c = 1
	}
	C.gonn_flash_attn_f64_bwd(
		(*C.double)(unsafe.Pointer(&Q[0])),
		(*C.double)(unsafe.Pointer(&K[0])),
		(*C.double)(unsafe.Pointer(&V[0])),
		(*C.double)(unsafe.Pointer(&O[0])),
		(*C.double)(unsafe.Pointer(&L[0])),
		(*C.double)(unsafe.Pointer(&dO[0])),
		(*C.double)(unsafe.Pointer(&dQ[0])),
		(*C.double)(unsafe.Pointer(&dK[0])),
		(*C.double)(unsafe.Pointer(&dV[0])),
		C.int(BH), C.int(S), C.int(d), C.double(scale), c,
	)
	return dQ, dK, dV
}

// BenchFlashAttnF64 runs the device-resident, CUDA-event-timed flash-attention
// benchmark and returns average ms/iter.
func BenchFlashAttnF64(BH, S, d, iters int, causal bool) float64 {
	var c C.int
	if causal {
		c = 1
	}
	return float64(C.gonn_bench_flash_attn_f64(C.int(BH), C.int(S), C.int(d), C.int(iters), c))
}

// DeviceBuffer is an opaque handle to a device (GPU) allocation of float64
// elements. Allocate once, run a chain of device ops, Download once — avoiding
// the per-call host<->device copies the eager backend pays. Free when done.
type DeviceBuffer struct {
	ptr unsafe.Pointer
	n   int
}

// DevAlloc allocates n float64 elements on the device (uninitialized).
func DevAlloc(n int) DeviceBuffer {
	return DeviceBuffer{ptr: unsafe.Pointer(C.gonn_dev_alloc(C.long(n) * 8)), n: n}
}

// DevUpload copies a host slice to a new device buffer.
func DevUpload(host []float64) DeviceBuffer {
	b := DevAlloc(len(host))
	if len(host) > 0 {
		C.gonn_dev_upload(b.ptr, (*C.double)(unsafe.Pointer(&host[0])), C.long(len(host)))
	}
	return b
}

// Download copies the device buffer back to a fresh host slice.
func (b DeviceBuffer) Download() []float64 {
	out := make([]float64, b.n)
	if b.n > 0 {
		C.gonn_dev_download((*C.double)(unsafe.Pointer(&out[0])), b.ptr, C.long(b.n))
	}
	return out
}

// Free releases the device allocation.
func (b DeviceBuffer) Free() { C.gonn_dev_free(b.ptr) }

// DevSync blocks until queued device work completes.
func DevSync() { C.gonn_dev_sync() }

// DevMatMul computes C(m,n) = A(m,k) * B(k,n) on resident buffers (no copies).
func DevMatMul(a, b, c DeviceBuffer, m, k, n int) {
	C.gonn_dev_matmul_f64(a.ptr, b.ptr, c.ptr, C.int(m), C.int(k), C.int(n))
}

// DevAdd computes C = A + B on resident buffers.
func DevAdd(a, b, c DeviceBuffer, n int) { C.gonn_dev_add_f64(a.ptr, b.ptr, c.ptr, C.int(n)) }

// DevReLU applies ReLU in place on a resident buffer.
func DevReLU(a DeviceBuffer, n int) { C.gonn_dev_relu_f64(a.ptr, C.int(n)) }

// DeviceBufferF16 is a device buffer of fp16 (__half) elements for the
// tensor-core GEMM path. Created from host float64 (converted on device).
type DeviceBufferF16 struct {
	ptr unsafe.Pointer
	n   int
}

// DevAllocF16 allocates n fp16 elements on the device.
func DevAllocF16(n int) DeviceBufferF16 {
	return DeviceBufferF16{ptr: unsafe.Pointer(C.gonn_dev_alloc(C.long(n) * 2)), n: n}
}

// DevUploadF16 uploads host float64 data as fp16 (converted on device).
func DevUploadF16(host []float64) DeviceBufferF16 {
	p := C.gonn_dev_upload_f16((*C.double)(unsafe.Pointer(&host[0])), C.int(len(host)))
	return DeviceBufferF16{ptr: unsafe.Pointer(p), n: len(host)}
}

// Download converts the fp16 buffer back to host float64.
func (b DeviceBufferF16) Download() []float64 {
	out := make([]float64, b.n)
	if b.n > 0 {
		C.gonn_dev_download_f16((*C.double)(unsafe.Pointer(&out[0])), b.ptr, C.int(b.n))
	}
	return out
}

// Free releases the fp16 device allocation.
func (b DeviceBufferF16) Free() { C.gonn_dev_free(b.ptr) }

// DevMatMulF16 computes C = A*B in fp16 on tensor cores (f32 accumulate).
func DevMatMulF16(a, b, c DeviceBufferF16, m, k, n int) {
	C.gonn_dev_matmul_f16(a.ptr, b.ptr, c.ptr, C.int(m), C.int(k), C.int(n))
}

// BenchMatMulF16Dev benchmarks the device-resident fp16 tensor-core GEMM.
func BenchMatMulF16Dev(m, k, n, iters int) float64 {
	return float64(C.gonn_bench_matmul_f16_dev(C.int(m), C.int(k), C.int(n), C.int(iters)))
}

// Backend returns the live CUDA backend.
func Backend() (backend.Backend, error) { return cudaBackend{}, nil }

// Available reports whether CUDA was compiled in.
func Available() bool { return true }
