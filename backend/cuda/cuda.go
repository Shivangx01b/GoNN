//go:build cuda
// +build cuda

// Package cuda is the CUDA backend, built with `-tags cuda`. It calls into
// gonn_cuda.cu via CGO. The native side is built on the CUDA runtime +
// cuBLAS (strided-batched GEMM) plus enum-dispatched elementwise kernels and
// a fused fp64 flash-attention (fwd+bwd).
package cuda

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lgonn_cuda -lcudart -lcublas

#include "gonn_cuda.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"gonn/backend"
)

// cudaBackend implements backend.Backend + backend.Elementwiser. The mutex
// serializes CGO entry: the native side stages host data through a grow-only
// workspace cache (g_ws in gonn_cuda.cu) that is not reentrant, and cuBLAS
// handles are not thread-safe across goroutines.
type cudaBackend struct {
	mu sync.Mutex
}

var theBackend = &cudaBackend{}

func (*cudaBackend) Name() backend.Device { return backend.CUDA }

// Gemm computes the row-major batched C = op(A) @ op(B) on the GPU via
// cublasDgemmStridedBatched. A GPU error here is fatal (panic): unlike the
// elementwise capability there is no cheap correct fallback at this layer,
// and silently degrading GEMM would invalidate benchmarks.
func (cb *cudaBackend) Gemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	out := make([]float64, batch*m*n)
	if batch == 0 || m == 0 || n == 0 || k == 0 {
		return out
	}
	var ta, tb C.int
	if transA {
		ta = 1
	}
	if transB {
		tb = 1
	}
	cb.mu.Lock()
	rc := C.gonn_gemm(
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.int(batch), C.int(m), C.int(k), C.int(n), ta, tb,
	)
	cb.mu.Unlock()
	if rc != 0 {
		panic(fmt.Sprintf("cuda: gonn_gemm failed with code %d", int(rc)))
	}
	return out
}

func (*cudaBackend) Synchronize() { C.gonn_sync() }

// Unary implements backend.Elementwiser. Returns false (declining the call,
// so the caller falls back to the CPU loop) on any GPU error.
func (cb *cudaBackend) Unary(kind backend.UnaryKind, a, out []float64) bool {
	if len(a) == 0 {
		return true
	}
	cb.mu.Lock()
	rc := C.gonn_unary(
		C.int(kind),
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.long(len(a)),
	)
	cb.mu.Unlock()
	return rc == 0
}

// Binary implements backend.Elementwiser.
func (cb *cudaBackend) Binary(kind backend.BinaryKind, a, b, out []float64) bool {
	if len(a) == 0 {
		return true
	}
	cb.mu.Lock()
	rc := C.gonn_binary(
		C.int(kind),
		(*C.double)(unsafe.Pointer(&a[0])),
		(*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&out[0])),
		C.long(len(a)),
	)
	cb.mu.Unlock()
	return rc == 0
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
func Backend() (backend.Backend, error) { return theBackend, nil }

// Available reports whether CUDA was compiled in.
func Available() bool { return true }
