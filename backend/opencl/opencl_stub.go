//go:build !opencl
// +build !opencl

// Package opencl is the OpenCL backend. The default (non-opencl) build provides
// a stub that returns a clear error when the backend is requested without the
// `opencl` build tag.
package opencl

import (
	"errors"

	"gonn/backend"
)

// Backend reports the OpenCL backend, or an error if it was not compiled in.
func Backend() (backend.Backend, error) {
	return nil, errors.New("opencl: this binary was not built with the `opencl` build tag; rebuild with `go build -tags opencl` (needs the OpenCL ICD + headers)")
}

// Available reports whether the OpenCL backend is compiled in.
func Available() bool { return false }
