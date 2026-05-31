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
	"math"
	"unsafe"

	"gonn/backend"
)

type openclBackend struct{}

func (openclBackend) Name() backend.Device { return backend.OpenCL }

func cptr(s []float64) *C.double { return (*C.double)(unsafe.Pointer(&s[0])) }

func (openclBackend) MatMul(a, b []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	if m == 0 || n == 0 {
		return out
	}
	C.gonn_cl_matmul(cptr(a), cptr(b), cptr(out), C.int(m), C.int(k), C.int(n))
	return out
}

func binary(a, b []float64, fn func(a, b, c *C.double, n C.int)) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	fn(cptr(a), cptr(b), cptr(out), C.int(len(a)))
	return out
}

func (openclBackend) AddElem(a, b []float64) []float64 {
	return binary(a, b, func(a, b, c *C.double, n C.int) { C.gonn_cl_add(a, b, c, n) })
}
func (openclBackend) MulElem(a, b []float64) []float64 {
	return binary(a, b, func(a, b, c *C.double, n C.int) { C.gonn_cl_mul(a, b, c, n) })
}
func (openclBackend) Sub(a, b []float64) []float64 {
	return binary(a, b, func(a, b, c *C.double, n C.int) { C.gonn_cl_sub(a, b, c, n) })
}
func (openclBackend) Div(a, b []float64) []float64 {
	return binary(a, b, func(a, b, c *C.double, n C.int) { C.gonn_cl_div(a, b, c, n) })
}

func (openclBackend) Scale(a []float64, s float64) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	C.gonn_cl_scale(cptr(a), C.double(s), cptr(out), C.int(len(a)))
	return out
}

func (openclBackend) AxpyInto(out, x []float64, alpha float64) {
	if len(out) == 0 {
		return
	}
	C.gonn_cl_axpy(cptr(out), cptr(x), C.double(alpha), C.int(len(out)))
}

func (openclBackend) Sum(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	var s C.double
	C.gonn_cl_sum(cptr(a), &s, C.int(len(a)))
	return float64(s)
}

func (openclBackend) Max(a []float64) float64 {
	if len(a) == 0 {
		return math.Inf(-1)
	}
	var m C.double
	C.gonn_cl_max(cptr(a), &m, C.int(len(a)))
	return float64(m)
}

func unary(a []float64, fn func(a, c *C.double, n C.int)) []float64 {
	out := make([]float64, len(a))
	if len(a) == 0 {
		return out
	}
	fn(cptr(a), cptr(out), C.int(len(a)))
	return out
}

func (openclBackend) ReLU(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_relu(a, c, n) })
}
func (openclBackend) Sigmoid(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_sigmoid(a, c, n) })
}
func (openclBackend) Tanh(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_tanh(a, c, n) })
}
func (openclBackend) Exp(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_exp(a, c, n) })
}
func (openclBackend) Log(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_log(a, c, n) })
}
func (openclBackend) GELU(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_gelu(a, c, n) })
}
func (openclBackend) SiLU(a []float64) []float64 {
	return unary(a, func(a, c *C.double, n C.int) { C.gonn_cl_silu(a, c, n) })
}

func (openclBackend) Synchronize() { C.gonn_cl_sync() }

// Backend returns the OpenCL backend if a GPU device with fp64 is available.
func Backend() (backend.Backend, error) {
	if C.gonn_cl_available() == 0 {
		return nil, errors.New("opencl: no usable OpenCL GPU device (fp64) found")
	}
	return openclBackend{}, nil
}

// Available reports whether the OpenCL backend is compiled in and usable.
func Available() bool { return C.gonn_cl_available() != 0 }
