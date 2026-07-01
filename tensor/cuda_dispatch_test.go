//go:build cuda
// +build cuda

package tensor

import (
	"math"
	"math/rand"
	"testing"

	"gonn/backend"
	"gonn/backend/cuda"
)

// GPU-gated dispatch tests: run inside the CUDA Docker image via
//
//	go test -tags cuda ./tensor/
//
// They force DispatchPolicy{0,0} so every dispatchable op actually executes
// on the GPU, then compare against the pure-Go reference path at tight
// tolerance (different libm implementations may differ by ~1 ULP).

func withCUDA(t *testing.T) {
	t.Helper()
	b, err := cuda.Backend()
	if err != nil {
		t.Skipf("CUDA backend unavailable: %v", err)
	}
	prevB := backend.Use(b)
	prevP := GetDispatchPolicy()
	SetDispatchPolicy(DispatchPolicy{UnaryMinElems: 0, BinaryMinElems: 0})
	t.Cleanup(func() {
		backend.Use(prevB)
		SetDispatchPolicy(prevP)
	})
}

func maxAbsDiff(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		if v := math.Abs(a[i] - b[i]); v > d {
			d = v
		}
	}
	return d
}

func TestCUDAUnaryDispatchParity(t *testing.T) {
	withCUDA(t)
	rng := rand.New(rand.NewSource(11))
	x := Zeros(4096)
	for i := range x.Data {
		x.Data[i] = rng.NormFloat64()
	}
	ops := []struct {
		name string
		f    func(*Tensor) *Tensor
	}{
		{"relu", (*Tensor).ReLU},
		{"sigmoid", (*Tensor).Sigmoid},
		{"tanh", (*Tensor).Tanh},
		{"exp", (*Tensor).Exp},
		{"gelu", (*Tensor).GELU},
		{"silu", (*Tensor).SiLU},
		{"geluexact", (*Tensor).GELUExact},
	}
	for _, op := range ops {
		gpu := op.f(x)
		// Reference: force the Go loop by dropping below the threshold.
		SetDispatchPolicy(DispatchPolicy{UnaryMinElems: math.MaxInt, BinaryMinElems: math.MaxInt})
		ref := op.f(x)
		SetDispatchPolicy(DispatchPolicy{UnaryMinElems: 0, BinaryMinElems: 0})
		if d := maxAbsDiff(gpu.Data, ref.Data); d > 1e-12 {
			t.Fatalf("%s: GPU vs CPU maxAbsDiff = %g", op.name, d)
		}
	}
}

func TestCUDABinaryDispatchParity(t *testing.T) {
	withCUDA(t)
	rng := rand.New(rand.NewSource(12))
	a := Zeros(4096)
	b := Zeros(4096)
	for i := range a.Data {
		a.Data[i] = rng.NormFloat64()
		b.Data[i] = rng.NormFloat64() + 3 // keep away from 0 for Div
	}
	ops := []struct {
		name string
		f    func(x, y *Tensor) *Tensor
	}{
		{"add", (*Tensor).Add},
		{"sub", (*Tensor).Sub},
		{"mul", (*Tensor).Mul},
		{"div", (*Tensor).Div},
	}
	for _, op := range ops {
		gpu := op.f(a, b)
		SetDispatchPolicy(DispatchPolicy{UnaryMinElems: math.MaxInt, BinaryMinElems: math.MaxInt})
		ref := op.f(a, b)
		SetDispatchPolicy(DispatchPolicy{UnaryMinElems: 0, BinaryMinElems: 0})
		if d := maxAbsDiff(gpu.Data, ref.Data); d > 1e-12 {
			t.Fatalf("%s: GPU vs CPU maxAbsDiff = %g", op.name, d)
		}
	}
}

func TestCUDAGemmParity(t *testing.T) {
	withCUDA(t)
	rng := rand.New(rand.NewSource(13))
	mk := func(shape ...int) *Tensor {
		x := Zeros(shape...)
		for i := range x.Data {
			x.Data[i] = rng.NormFloat64()
		}
		return x
	}
	a2, b2 := mk(31, 17), mk(17, 23)
	a4, b4 := mk(2, 3, 8, 5), mk(2, 3, 5, 7)

	gpu2 := a2.MatMul(b2)
	gpu4 := a4.MatMul(b4)
	// CPU reference through the default backend.
	backend.Use(backend.NewCPU())
	ref2 := a2.MatMul(b2)
	ref4 := a4.MatMul(b4)
	if d := maxAbsDiff(gpu2.Data, ref2.Data); d > 1e-10 {
		t.Fatalf("2D gemm GPU vs CPU maxAbsDiff = %g", d)
	}
	if d := maxAbsDiff(gpu4.Data, ref4.Data); d > 1e-10 {
		t.Fatalf("batched gemm GPU vs CPU maxAbsDiff = %g", d)
	}

	// Gradients flow through the GPU GEMM path.
	b, _ := cuda.Backend()
	backend.Use(b)
	w := mk(4, 3).SetRequiresGrad(true)
	x := mk(5, 4)
	x.MatMul(w).Square().Mean().Backward()
	if w.Grad == nil {
		t.Fatal("no gradient through GPU GEMM")
	}
}
