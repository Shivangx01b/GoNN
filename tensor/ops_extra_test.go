package tensor

import (
	"math"
	"testing"
)

// gradCheck verifies the analytic gradient (from Backward) of a scalar loss
// against central finite differences. f builds a fresh graph from the current
// values of x and returns the scalar loss tensor; x.Data is perturbed in place.
// The analytic gradient is read from x.Grad after f(x).Backward().
func gradCheck(t *testing.T, x *Tensor, f func(*Tensor) *Tensor) {
	t.Helper()
	x.RequiresGrad = true
	x.Grad = nil
	loss := f(x)
	if len(loss.Data) != 1 {
		t.Fatalf("gradCheck: loss must be scalar, got shape %v", loss.Shape)
	}
	loss.Backward()
	if x.Grad == nil {
		t.Fatalf("gradCheck: no gradient flowed to x")
	}
	analytic := append([]float64(nil), x.Grad.Data...)

	const eps = 1e-6
	num := make([]float64, len(x.Data))
	for i := range x.Data {
		orig := x.Data[i]
		x.Data[i] = orig + eps
		yp := f(x).Data[0]
		x.Data[i] = orig - eps
		ym := f(x).Data[0]
		x.Data[i] = orig
		num[i] = (yp - ym) / (2 * eps)
	}

	var maxRel float64
	for i := range analytic {
		denom := math.Max(1e-8, math.Max(math.Abs(analytic[i]), math.Abs(num[i])))
		rel := math.Abs(analytic[i]-num[i]) / denom
		if rel > maxRel {
			maxRel = rel
		}
	}
	if maxRel >= 1e-5 {
		t.Fatalf("gradient check failed: maxRelErr=%g\nanalytic=%v\nnumeric =%v", maxRel, analytic, num)
	}
}

// fixedWeights returns a deterministic non-trivial weight tensor of the same
// numel as shape, so the scalar loss is sum(out*w) rather than plain sum(out)
// (this catches backward bugs that a uniform upstream grad would hide).
func fixedWeights(shape ...int) *Tensor {
	n := numel(shape)
	d := make([]float64, n)
	for i := range d {
		d[i] = math.Sin(float64(i)*0.7+1.3) + 0.5*float64(i%3)
	}
	return New(d, shape...)
}

func TestTrilTriuForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	lo := a.Tril(0)
	approxEq(t, lo.Data, []float64{1, 0, 0, 4, 5, 0, 7, 8, 9}, 1e-9)
	up := a.Triu(0)
	approxEq(t, up.Data, []float64{1, 2, 3, 0, 5, 6, 0, 0, 9}, 1e-9)
	// k=1 tril keeps one super-diagonal
	lo1 := a.Tril(1)
	approxEq(t, lo1.Data, []float64{1, 2, 0, 4, 5, 6, 7, 8, 9}, 1e-9)
	// batched: shape (2,2,2)
	b := New([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2)
	bl := b.Tril(0)
	approxEq(t, bl.Data, []float64{1, 0, 3, 4, 5, 0, 7, 8}, 1e-9)
}

func TestTrilTriuGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	w := fixedWeights(3, 3)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Tril(0).Mul(w).Sum() })
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Triu(-1).Mul(w).Sum() })
}

func TestWhereForward(t *testing.T) {
	cond := New([]float64{1, 0, 1, 0}, 2, 2)
	a := New([]float64{10, 20, 30, 40}, 2, 2)
	b := New([]float64{-1, -2, -3, -4}, 2, 2)
	o := Where(cond, a, b)
	approxEq(t, o.Data, []float64{10, -2, 30, -4}, 1e-9)
}

func TestWhereGrad(t *testing.T) {
	cond := New([]float64{1, 0, 1, 0, 1, 0}, 2, 3)
	b := New([]float64{9, 8, 7, 6, 5, 4}, 2, 3)
	wa := fixedWeights(2, 3)
	// grad wrt a
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	gradCheck(t, a, func(a *Tensor) *Tensor {
		return Where(cond, a, b.Copy()).Mul(wa).Sum()
	})
	// grad wrt b
	a2 := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	gradCheck(t, b, func(b *Tensor) *Tensor {
		return Where(cond, a2.Copy(), b).Mul(wa).Sum()
	})
}

func TestMaskedFillForward(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, 2, 2)
	m := New([]float64{0, 1, 0, 1}, 2, 2)
	o := x.MaskedFill(m, -9)
	approxEq(t, o.Data, []float64{1, -9, 3, -9}, 1e-9)
}

func TestMaskedFillGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	m := New([]float64{0, 1, 0, 1, 0, 0}, 2, 3)
	w := fixedWeights(2, 3)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.MaskedFill(m, 3.14).Mul(w).Sum() })
}

func TestCumsumForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	c1 := a.Cumsum(1)
	approxEq(t, c1.Data, []float64{1, 3, 6, 4, 9, 15}, 1e-9)
	c0 := a.Cumsum(0)
	approxEq(t, c0.Data, []float64{1, 2, 3, 5, 7, 9}, 1e-9)
}

func TestCumsumGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	w := fixedWeights(2, 3)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Cumsum(1).Mul(w).Sum() })
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Cumsum(0).Mul(w).Sum() })
}

func TestGatherForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	idx := New([]float64{0, 2, 1, 1, 0, 2}, 2, 3)
	o := a.Gather(1, idx)
	// row0: a[0,0],a[0,2],a[0,1] = 1,3,2 ; row1: a[1,1],a[1,0],a[1,2]=5,4,6
	approxEq(t, o.Data, []float64{1, 3, 2, 5, 4, 6}, 1e-9)
	// axis 0 gather
	idx0 := New([]float64{1, 0, 1}, 1, 3)
	o0 := a.Gather(0, idx0)
	approxEq(t, o0.Data, []float64{4, 2, 6}, 1e-9)
}

func TestGatherGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	idx := New([]float64{0, 2, 2, 1, 0, 1}, 2, 3) // includes duplicate indices
	w := fixedWeights(2, 3)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Gather(1, idx).Mul(w).Sum() })
}

func TestIndexSelectForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	o := a.IndexSelect(1, New([]float64{2, 0}, 2))
	approxEq(t, o.Data, []float64{3, 1, 6, 4}, 1e-9)
	o2 := a.IndexSelect(0, New([]float64{1, 1, 0}, 3))
	approxEq(t, o2.Data, []float64{4, 5, 6, 4, 5, 6, 1, 2, 3}, 1e-9)
}

func TestIndexSelectGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	idx := New([]float64{2, 0, 2}, 3) // duplicate -> scatter-add
	w := fixedWeights(2, 3)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.IndexSelect(1, idx).Mul(w).Sum() })
}

func TestFlipForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	f1 := a.Flip(1)
	approxEq(t, f1.Data, []float64{3, 2, 1, 6, 5, 4}, 1e-9)
	f0 := a.Flip(0)
	approxEq(t, f0.Data, []float64{4, 5, 6, 1, 2, 3}, 1e-9)
	fb := a.Flip(0, 1)
	approxEq(t, fb.Data, []float64{6, 5, 4, 3, 2, 1}, 1e-9)
}

func TestFlipGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	w := fixedWeights(2, 3)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Flip(0, 1).Mul(w).Sum() })
}

func TestRepeatForward(t *testing.T) {
	a := New([]float64{1, 2, 3}, 1, 3)
	r := a.Repeat(2, 2)
	// shape (2,6): [[1,2,3,1,2,3],[1,2,3,1,2,3]]
	if !shapesEqual(r.Shape, []int{2, 6}) {
		t.Fatalf("Repeat shape: got %v", r.Shape)
	}
	approxEq(t, r.Data, []float64{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, 1e-9)
}

func TestRepeatGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, 2, 2)
	w := fixedWeights(4, 6)
	gradCheck(t, x, func(x *Tensor) *Tensor { return x.Repeat(2, 3).Mul(w).Sum() })
}

func TestSplitForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 2, 5)
	parts := a.Split(1, 2)
	if len(parts) != 3 {
		t.Fatalf("Split: got %d parts want 3", len(parts))
	}
	approxEq(t, parts[0].Data, []float64{1, 2, 6, 7}, 1e-9)
	approxEq(t, parts[1].Data, []float64{3, 4, 8, 9}, 1e-9)
	approxEq(t, parts[2].Data, []float64{5, 10}, 1e-9) // last chunk smaller
}

func TestChunkForward(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	parts := a.Chunk(1, 2)
	if len(parts) != 2 {
		t.Fatalf("Chunk: got %d parts want 2", len(parts))
	}
	approxEq(t, parts[0].Data, []float64{1, 2, 5, 6}, 1e-9)
	approxEq(t, parts[1].Data, []float64{3, 4, 7, 8}, 1e-9)
}

func TestSplitGrad(t *testing.T) {
	x := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 2, 5)
	// loss combines all pieces with distinct weights so each region's grad matters
	w0 := fixedWeights(2, 2)
	w1 := fixedWeights(2, 2)
	w2 := fixedWeights(2, 1)
	gradCheck(t, x, func(x *Tensor) *Tensor {
		p := x.Split(1, 2)
		l0 := p[0].Mul(w0).Sum()
		l1 := p[1].MulScalar(2).Mul(w1).Sum()
		l2 := p[2].MulScalar(3).Mul(w2).Sum()
		return l0.Add(l1).Add(l2)
	})
}

func TestArgWhere(t *testing.T) {
	a := New([]float64{0, 5, 0, 3, 0, 0, 9}, 7)
	idx := a.ArgWhere()
	approxEq(t, idx.Data, []float64{1, 3, 6}, 1e-9)
}
