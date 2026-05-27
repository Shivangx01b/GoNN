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
		return -1.0 / 0.0
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

// Backend returns the live CUDA backend.
func Backend() (backend.Backend, error) { return cudaBackend{}, nil }

// Available reports whether CUDA was compiled in.
func Available() bool { return true }
