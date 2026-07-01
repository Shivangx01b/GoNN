package tensor

// This file is the ONLY place the tensor package touches the backend: the
// single seam through which GEMMs and (optionally) elementwise ops reach an
// accelerator. A future device-resident tensor type would change dispatch
// here and nowhere else.
//
// Cost model for the elementwise path (documented so the defaults make
// sense): tensor storage is host memory, so a dispatched elementwise op pays
// a host->device copy of every input and a device->host copy of the result —
// 16-24 bytes per element over PCIe (~1.5-2 ns/elem) plus tens of µs of
// launch latency. Transcendental unaries (exp/tanh/sigmoid/gelu/silu: ~10-25
// ns/elem on CPU) can win above roughly 10^5-10^6 elements; bandwidth-bound
// binaries (add/mul: <1 ns/elem on CPU) essentially never win, because the
// copy alone costs more than the compute. Hence the defaults below: unary
// dispatch on for large tensors, binary dispatch off. Tune with
// SetDispatchPolicy after measuring on your hardware (see benchmark/verify).

import (
	"math"
	"sync/atomic"

	"gonn/backend"
)

// UnaryKind aliases backend.UnaryKind (with the constants re-exported below)
// so registry definitions and custom-op authors never import the backend
// package directly — this file stays the package's only backend dependency.
type UnaryKind = backend.UnaryKind

// Re-exported unary kernel kinds (see backend.UnaryKind for the ABI contract).
const (
	UnaryNone    = backend.UnaryNone
	UnaryReLU    = backend.UnaryReLU
	UnarySigmoid = backend.UnarySigmoid
	UnaryTanh    = backend.UnaryTanh
	UnaryExp     = backend.UnaryExp
	UnaryLog     = backend.UnaryLog
	UnaryGELU    = backend.UnaryGELU
	UnarySiLU    = backend.UnarySiLU
)

// DispatchPolicy controls when elementwise tensor ops are routed to a backend
// that implements backend.Elementwiser. An op is dispatched when the tensor
// has at least MinElems elements AND the active backend advertises the
// capability AND accepts the call; otherwise the pure-Go loop runs.
type DispatchPolicy struct {
	// UnaryMinElems is the minimum element count for dispatching unary ops
	// (activations, exp, log, ...). Default 1<<16, tuned from the measured
	// break-even on an RTX 3060 (benchmark/gonn dispatch table: tanh at 64K
	// elems runs 3.2x faster dispatched — 0.22ms vs 0.70ms host — and is
	// already at parity by 16K thanks to the kernel-side workspace cache).
	UnaryMinElems int
	// BinaryMinElems is the minimum element count for dispatching binary ops
	// (add/sub/mul/div). Default math.MaxInt (disabled): with host-resident
	// storage the PCIe copies always dominate these bandwidth-bound ops.
	BinaryMinElems int
}

var dispatchPolicy atomic.Pointer[DispatchPolicy]

func init() {
	p := DispatchPolicy{UnaryMinElems: 1 << 16, BinaryMinElems: math.MaxInt}
	dispatchPolicy.Store(&p)
}

// SetDispatchPolicy replaces the dispatch thresholds. Safe for concurrent use.
func SetDispatchPolicy(p DispatchPolicy) { dispatchPolicy.Store(&p) }

// GetDispatchPolicy returns the active dispatch thresholds.
func GetDispatchPolicy() DispatchPolicy { return *dispatchPolicy.Load() }

// dispatchUnary tries to run out[i] = f(a[i]) on the active backend.
// Returns false (caller must run the CPU loop) when the op has no kernel,
// the tensor is below the threshold, the backend lacks the capability, or
// the backend declines.
func dispatchUnary(kind backend.UnaryKind, a, out []float64) bool {
	if kind == backend.UnaryNone || len(a) < GetDispatchPolicy().UnaryMinElems {
		return false
	}
	ew, ok := backend.Current().(backend.Elementwiser)
	return ok && ew.Unary(kind, a, out)
}

// binaryKindOf maps the op name used by binOp to the backend dispatch enum.
func binaryKindOf(op string) backend.BinaryKind {
	switch op {
	case "Add":
		return backend.BinaryAdd
	case "Sub":
		return backend.BinarySub
	case "Mul":
		return backend.BinaryMul
	default:
		return backend.BinaryDiv
	}
}

// dispatchBinary tries to run out[i] = a[i] op b[i] on the active backend.
func dispatchBinary(kind backend.BinaryKind, a, b, out []float64) bool {
	if len(a) < GetDispatchPolicy().BinaryMinElems {
		return false
	}
	ew, ok := backend.Current().(backend.Elementwiser)
	return ok && ew.Binary(kind, a, b, out)
}

// dispatchGemm runs the batched GEMM on the active backend, unconditionally:
// GEMM is compute-dense enough that acceleration pays for the copies.
func dispatchGemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	return backend.Current().Gemm(a, b, batch, m, k, n, transA, transB)
}
