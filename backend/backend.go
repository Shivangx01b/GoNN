// Package backend defines the compute-backend contract for GoNN.
// The default backend is pure-Go CPU. CUDA is available via build tag `cuda`
// and CGO; see backend/cuda/*.
package backend

// Device names the compute target.
type Device string

const (
	CPU  Device = "cpu"
	CUDA Device = "cuda"
)

// Backend is the minimal contract a compute target must implement. For v1
// we keep it tiny — full op coverage lives in the tensor package and only
// the hot kernels (matmul, elementwise add/mul) get accelerated.
type Backend interface {
	Name() Device
	MatMul(a, b []float64, m, k, n int) []float64
	AddElem(a, b []float64) []float64
	MulElem(a, b []float64) []float64
	Synchronize()
}

// current is the active backend. Default to the CPU implementation.
var current Backend = cpuBackend{}

// Use switches the active backend. Returns the previous backend so callers
// can restore on defer.
func Use(b Backend) Backend {
	prev := current
	current = b
	return prev
}

// Current returns the active backend.
func Current() Backend { return current }
