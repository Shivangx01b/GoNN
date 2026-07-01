package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

func TestParametersToVectorRoundTrip(t *testing.T) {
	a := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := tensor.New([]float64{7, 8}, 2)
	params := []*tensor.Tensor{a, b}

	vec := ParametersToVector(params)
	if vec.Numel() != 8 || len(vec.Shape) != 1 {
		t.Fatalf("vector shape %v, want [8]", vec.Shape)
	}
	for i := 0; i < 8; i++ {
		if vec.Data[i] != float64(i+1) {
			t.Fatalf("vec[%d] = %v, want %v", i, vec.Data[i], i+1)
		}
	}

	// The vector is a copy: mutating it must not touch the parameters.
	vec.Data[0] = 100
	if a.Data[0] != 1 {
		t.Fatal("ParametersToVector aliased parameter storage")
	}

	// Round trip: scramble the parameters, restore from the saved vector.
	save := ParametersToVector(params)
	for i := range a.Data {
		a.Data[i] = -1
	}
	for i := range b.Data {
		b.Data[i] = -1
	}
	VectorToParameters(save, params)
	for i := 0; i < 6; i++ {
		if a.Data[i] != float64(i+1) {
			t.Fatalf("a restored wrong at %d: %v", i, a.Data[i])
		}
	}
	if b.Data[0] != 7 || b.Data[1] != 8 {
		t.Fatalf("b restored wrong: %v", b.Data)
	}
}

func TestVectorToParametersLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	VectorToParameters(tensor.Zeros(3), []*tensor.Tensor{tensor.Zeros(2, 2)})
}

func TestParametersToVectorNilParamPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on nil parameter")
		}
	}()
	ParametersToVector([]*tensor.Tensor{tensor.Zeros(2), nil})
}

func TestGradsToVectorZerosForNilGrad(t *testing.T) {
	a := gradTensor(1, 2) // grad = [1 2]
	b := tensor.Zeros(3)  // nil grad -> zeros segment
	v := GradsToVector([]*tensor.Tensor{a, b})
	want := []float64{1, 2, 0, 0, 0}
	if v.Numel() != len(want) {
		t.Fatalf("grads vector length %d, want %d", v.Numel(), len(want))
	}
	for i, w := range want {
		if v.Data[i] != w {
			t.Fatalf("grads vec[%d] = %v, want %v", i, v.Data[i], w)
		}
	}
}

func TestTotalGradNormMatchesClipGradNormReturn(t *testing.T) {
	a := gradTensor(3)
	b := gradTensor(4)
	params := []*tensor.Tensor{a, b, nil}

	total := TotalGradNorm(params)
	if math.Abs(total-5) > 1e-12 {
		t.Fatalf("TotalGradNorm = %v, want 5", total)
	}
	// maxNorm above the total: ClipGradNorm is a no-op and returns the same norm.
	pre := ClipGradNorm(params, 10)
	if pre != total {
		t.Fatalf("ClipGradNorm pre-clip norm %v != TotalGradNorm %v", pre, total)
	}
	if a.Grad.Data[0] != 3 || b.Grad.Data[0] != 4 {
		t.Fatal("grads modified below the limit")
	}
}

func TestClipGradsWithNormMatchesClipGradNormEndState(t *testing.T) {
	a1 := gradTensor(3, 4)
	a2 := gradTensor(3, 4)

	ClipGradNorm([]*tensor.Tensor{a1}, 1.0)
	ClipGradsWithNorm([]*tensor.Tensor{a2}, 1.0, TotalGradNorm([]*tensor.Tensor{a2}))
	for i := range a1.Grad.Data {
		if a1.Grad.Data[i] != a2.Grad.Data[i] {
			t.Fatalf("grad[%d]: ClipGradNorm %v != ClipGradsWithNorm %v", i, a1.Grad.Data[i], a2.Grad.Data[i])
		}
	}
}

func TestClipGradsWithNormNeverAmplifies(t *testing.T) {
	p := gradTensor(0.3, 0.4) // norm 0.5
	ClipGradsWithNorm([]*tensor.Tensor{p}, 10, 0.5)
	if p.Grad.Data[0] != 0.3 || p.Grad.Data[1] != 0.4 {
		t.Fatalf("scale > 1 must be clamped to 1 (no amplification), got %v", p.Grad.Data)
	}
}
