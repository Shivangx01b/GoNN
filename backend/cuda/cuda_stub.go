//go:build !cuda
// +build !cuda

// Package cuda is the CUDA backend. The default (non-cuda) build provides
// a stub that returns a clear error when the backend is requested without
// the `cuda` build tag.
package cuda

import (
	"errors"

	"gonn/backend"
)

// Backend reports the CUDA backend, or an error if CUDA was not compiled in.
func Backend() (backend.Backend, error) {
	return nil, errors.New("cuda: this binary was not built with the `cuda` build tag; rebuild with `go build -tags cuda` and ensure nvcc is on PATH")
}

// Available reports whether the CUDA backend is compiled in.
func Available() bool { return false }
