// Package backend defines the compute-backend contract for GoNN.
// The default backend is pure-Go CPU. CUDA is available via build tag `cuda`
// and CGO; see backend/cuda/*.
package backend

// Device names the compute target.
type Device string

const (
	CPU    Device = "cpu"
	CUDA   Device = "cuda"
	OpenCL Device = "opencl"
)

// Backend is the minimal contract a compute target must implement. For v1
// we keep it tiny — full op coverage lives in the tensor package and only
// the hot kernels (matmul, elementwise add/mul, common activations, simple
// reductions) get accelerated. New methods are appended to remain
// backward-compatible with older implementations that embedded an earlier
// version of the interface.
type Backend interface {
	Name() Device
	MatMul(a, b []float64, m, k, n int) []float64
	AddElem(a, b []float64) []float64
	MulElem(a, b []float64) []float64
	Synchronize()

	// Elementwise arithmetic.
	Sub(a, b []float64) []float64
	Div(a, b []float64) []float64
	Scale(a []float64, s float64) []float64
	// AxpyInto computes out += alpha*x in place.
	AxpyInto(out, x []float64, alpha float64)

	// Reductions.
	Sum(a []float64) float64
	Max(a []float64) float64

	// Activations / unary math.
	ReLU(a []float64) []float64
	Sigmoid(a []float64) []float64
	Tanh(a []float64) []float64
	Exp(a []float64) []float64
	Log(a []float64) []float64
	// GELU uses the tanh approximation:
	//   0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
	GELU(a []float64) []float64
	// SiLU (a.k.a. Swish): x * sigmoid(x)
	SiLU(a []float64) []float64
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
