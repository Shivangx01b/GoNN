package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

func gradTensor(vals ...float64) *tensor.Tensor {
	p := tensor.New(append([]float64(nil), vals...), len(vals)).SetRequiresGrad(true)
	p.Grad = tensor.New(append([]float64(nil), vals...), len(vals))
	return p
}

func TestClipGradNormScales(t *testing.T) {
	p := gradTensor(3, 4) // grad norm 5
	got := ClipGradNorm([]*tensor.Tensor{p}, 1.0)
	if math.Abs(got-5) > 1e-12 {
		t.Fatalf("returned pre-clip norm %v, want 5", got)
	}
	var sq float64
	for _, g := range p.Grad.Data {
		sq += g * g
	}
	if post := math.Sqrt(sq); math.Abs(post-1) > 1e-5 {
		t.Fatalf("post-clip norm %v, want ~1", post)
	}
}

func TestClipGradNormNoopUnderLimit(t *testing.T) {
	p := gradTensor(0.3, 0.4) // norm 0.5
	got := ClipGradNorm([]*tensor.Tensor{p}, 1.0)
	if math.Abs(got-0.5) > 1e-12 {
		t.Fatalf("returned %v, want 0.5", got)
	}
	if p.Grad.Data[0] != 0.3 || p.Grad.Data[1] != 0.4 {
		t.Fatalf("grads modified below the limit: %v", p.Grad.Data)
	}
}

func TestClipGradNormGlobalAcrossParams(t *testing.T) {
	a := gradTensor(3)
	b := gradTensor(4)
	total := ClipGradNorm([]*tensor.Tensor{a, b, nil}, 2.5) // global norm 5
	if math.Abs(total-5) > 1e-12 {
		t.Fatalf("total norm %v, want 5", total)
	}
	// Both scaled by the same factor 2.5/(5+1e-6) ~ 0.5.
	if math.Abs(a.Grad.Data[0]-1.5) > 1e-5 || math.Abs(b.Grad.Data[0]-2.0) > 1e-5 {
		t.Fatalf("grads not globally scaled: %v %v", a.Grad.Data[0], b.Grad.Data[0])
	}
}

func TestClipGradValue(t *testing.T) {
	p := gradTensor(-3, -0.5, 0.5, 3)
	ClipGradValue([]*tensor.Tensor{p}, 1.0)
	want := []float64{-1, -0.5, 0.5, 1}
	for i, w := range want {
		if p.Grad.Data[i] != w {
			t.Fatalf("clip value [%d] = %v, want %v", i, p.Grad.Data[i], w)
		}
	}
}
