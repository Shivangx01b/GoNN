//go:build opencl
// +build opencl

// Package opencl is the OpenCL compute backend, built with `-tags opencl`. It
// calls into gonn_opencl.c via CGO (OpenCL ICD + fp64). The OpenCL headers and
// loader ship with the CUDA toolkit; set CGO flags to point at them, e.g.:
//
//	CGO_CFLAGS=-I/usr/local/cuda/include
//	CGO_LDFLAGS=-L/usr/local/cuda/lib64
//	go build -tags opencl ./...
package opencl

/*
#cgo CFLAGS: -I${SRCDIR} -DCL_TARGET_OPENCL_VERSION=120
#cgo LDFLAGS: -lOpenCL

#include "gonn_opencl.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"gonn/backend"
)

// openclBackend implements backend.Backend + backend.Elementwiser. The mutex
// serializes CGO entry: the native side caches kernel objects per op kind,
// which is not thread-safe.
type openclBackend struct {
	mu sync.Mutex
}

var theBackend = &openclBackend{}

func (*openclBackend) Name() backend.Device { return backend.OpenCL }

func cptr(s []float64) *C.double { return (*C.double)(unsafe.Pointer(&s[0])) }

// Gemm computes the row-major batched C = op(A) @ op(B) via the gemmk kernel
// (naive 3D NDRange — correctness-first; OpenCL is the portability path).
// A GPU error here is fatal, matching the CUDA backend's contract.
func (ob *openclBackend) Gemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
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
	ob.mu.Lock()
	rc := C.gonn_cl_gemm(cptr(a), cptr(b), cptr(out),
		C.int(batch), C.int(m), C.int(k), C.int(n), ta, tb)
	ob.mu.Unlock()
	if rc != 0 {
		panic(fmt.Sprintf("opencl: gonn_cl_gemm failed with code %d", int(rc)))
	}
	return out
}

func (*openclBackend) Synchronize() { C.gonn_cl_sync() }

// Unary implements backend.Elementwiser. Returns false (declining the call,
// so the caller falls back to the CPU loop) on any device error.
func (ob *openclBackend) Unary(kind backend.UnaryKind, a, out []float64) bool {
	if len(a) == 0 {
		return true
	}
	ob.mu.Lock()
	rc := C.gonn_cl_unary(C.int(kind), cptr(a), cptr(out), C.int(len(a)))
	ob.mu.Unlock()
	return rc == 0
}

// Binary implements backend.Elementwiser.
func (ob *openclBackend) Binary(kind backend.BinaryKind, a, b, out []float64) bool {
	if len(a) == 0 {
		return true
	}
	ob.mu.Lock()
	rc := C.gonn_cl_binary(C.int(kind), cptr(a), cptr(b), cptr(out), C.int(len(a)))
	ob.mu.Unlock()
	return rc == 0
}

// Backend returns the OpenCL backend if a device with fp64 is available.
func Backend() (backend.Backend, error) {
	if C.gonn_cl_available() == 0 {
		return nil, errors.New("opencl: no usable OpenCL device (fp64) found")
	}
	return theBackend, nil
}

// Available reports whether the OpenCL backend is compiled in and usable.
func Available() bool { return C.gonn_cl_available() != 0 }
