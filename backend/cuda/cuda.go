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

// Backend returns the live CUDA backend.
func Backend() (backend.Backend, error) { return cudaBackend{}, nil }

// Available reports whether CUDA was compiled in.
func Available() bool { return true }
