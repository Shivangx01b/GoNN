// Package backend defines the compute-backend contract for GoNN.
//
// The required surface is deliberately tiny: a batched GEMM plus a
// synchronization point. Everything else is an optional capability that a
// backend may advertise by implementing the corresponding interface
// (currently Elementwiser); callers feature-detect with a type assertion and
// fall back to the pure-Go path when the capability is absent or an
// implementation declines a call. Adding an accelerated op therefore never
// changes the Backend interface again — it adds an enum value and a kernel.
//
// The default backend is pure-Go CPU (gonum BLAS GEMM). CUDA is available via
// build tag `cuda`, OpenCL via `opencl`; see backend/cuda and backend/opencl.
package backend

import "sync/atomic"

// Device names the compute target.
type Device string

const (
	CPU    Device = "cpu"
	CUDA   Device = "cuda"
	OpenCL Device = "opencl"
)

// UnaryKind identifies an accelerated elementwise unary op. The numeric
// values are part of the C ABI — they are mirrored as enums in
// backend/cuda/gonn_cuda.h and backend/opencl/gonn_opencl.h. Append only;
// never reorder or remove.
type UnaryKind int32

const (
	// UnaryNone marks an op with no accelerated kernel; dispatch always
	// declines it and the caller runs the pure-Go implementation.
	UnaryNone UnaryKind = -1
)

const (
	UnaryReLU UnaryKind = iota
	UnarySigmoid
	UnaryTanh
	UnaryExp
	UnaryLog
	UnaryGELU // tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
	UnarySiLU // x * sigmoid(x)
)

// BinaryKind identifies an accelerated elementwise binary op. Same append-only
// C ABI contract as UnaryKind.
type BinaryKind int32

const (
	BinaryAdd BinaryKind = iota
	BinarySub
	BinaryMul
	BinaryDiv
)

// Backend is the minimal required contract for a compute target.
type Backend interface {
	Name() Device

	// Gemm computes the row-major batched product C = op(A) @ op(B) for
	// `batch` independent matrix pairs stored contiguously:
	//
	//   op(A) is (m,k) — stored (m,k), or (k,m) when transA;
	//   op(B) is (k,n) — stored (k,n), or (n,k) when transB;
	//   C     is (m,n).
	//
	// A has batch*m*k elements, B batch*k*n, and the result batch*m*n.
	// batch == 1 is the plain 2D GEMM. Implementations may assume the slice
	// lengths match; callers (the tensor package) validate shapes first.
	Gemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64

	// Synchronize blocks until queued device work completes (no-op on CPU).
	Synchronize()
}

// Elementwiser is an optional capability for accelerated elementwise ops.
// Implementations write the result into out and return true, or return false
// to decline the call (unsupported kind, device error, ...), in which case
// the caller falls back to the pure-Go loop. len(a) == len(out) (and len(b)
// for Binary) is the caller's responsibility.
//
// The CPU backend intentionally does NOT implement Elementwiser: the tensor
// package's Go closures are the canonical CPU implementation, keeping every
// op defined exactly once per device.
type Elementwiser interface {
	Unary(kind UnaryKind, a, out []float64) bool
	Binary(kind BinaryKind, a, b, out []float64) bool
}

// holder boxes the Backend so atomic.Pointer works across distinct concrete
// backend types (atomic.Value would panic on inconsistent types).
type holder struct{ b Backend }

var current atomic.Pointer[holder]

func init() {
	current.Store(&holder{cpuBackend{}})
}

// Use switches the active backend and returns the previous one so callers can
// restore it on defer. Safe for concurrent use.
func Use(b Backend) Backend {
	return current.Swap(&holder{b}).b
}

// Current returns the active backend. Safe for concurrent use (lock-free).
func Current() Backend {
	return current.Load().b
}
