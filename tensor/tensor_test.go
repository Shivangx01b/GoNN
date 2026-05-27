package tensor

import (
	"math"
	"testing"
)

func approxEq(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestAdd(t *testing.T) {
	a := New([]float64{1, 2, 3}, 3)
	b := New([]float64{4, 5, 6}, 3)
	c := a.Add(b)
	approxEq(t, c.Data, []float64{5, 7, 9}, 1e-9)
}

func TestMul(t *testing.T) {
	a := New([]float64{1, 2, 3}, 3)
	b := New([]float64{4, 5, 6}, 3)
	c := a.Mul(b)
	approxEq(t, c.Data, []float64{4, 10, 18}, 1e-9)
}

func TestMatMul(t *testing.T) {
	a := New([]float64{1, 2, 3, 4}, 2, 2) // [[1,2],[3,4]]
	b := New([]float64{5, 6, 7, 8}, 2, 2) // [[5,6],[7,8]]
	c := a.MatMul(b)
	// [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
	approxEq(t, c.Data, []float64{19, 22, 43, 50}, 1e-9)
}

func TestBroadcastAdd(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := New([]float64{10, 20, 30}, 3) // broadcasts to (2,3)
	c := a.Add(b)
	approxEq(t, c.Data, []float64{11, 22, 33, 14, 25, 36}, 1e-9)
}

func TestSumReduce(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if s := a.Sum().Data[0]; s != 21 {
		t.Fatalf("sum: got %v want 21", s)
	}
	row := a.SumAxis(1, false) // sum each row
	approxEq(t, row.Data, []float64{6, 15}, 1e-9)
	col := a.SumAxis(0, false) // sum each col
	approxEq(t, col.Data, []float64{5, 7, 9}, 1e-9)
}

func TestReshape(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Reshape(3, 2)
	if b.Shape[0] != 3 || b.Shape[1] != 2 {
		t.Fatalf("shape: got %v want [3 2]", b.Shape)
	}
}

func TestTranspose(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := a.Transpose()
	// [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
	approxEq(t, b.Data, []float64{1, 4, 2, 5, 3, 6}, 1e-9)
}

func TestAutogradAdd(t *testing.T) {
	x := New([]float64{2, 3}, 2).SetRequiresGrad(true)
	y := New([]float64{5, 7}, 2).SetRequiresGrad(true)
	z := x.Add(y).Sum()
	z.Backward()
	approxEq(t, x.Grad.Data, []float64{1, 1}, 1e-9)
	approxEq(t, y.Grad.Data, []float64{1, 1}, 1e-9)
}

func TestAutogradMul(t *testing.T) {
	x := New([]float64{2, 3}, 2).SetRequiresGrad(true)
	y := New([]float64{5, 7}, 2).SetRequiresGrad(true)
	z := x.Mul(y).Sum()
	z.Backward()
	approxEq(t, x.Grad.Data, []float64{5, 7}, 1e-9)
	approxEq(t, y.Grad.Data, []float64{2, 3}, 1e-9)
}

func TestAutogradMatMul(t *testing.T) {
	// y = sum(A @ x), dA = x^T broadcast, dx = sum(A) column-wise
	A := New([]float64{1, 2, 3, 4}, 2, 2).SetRequiresGrad(true)
	x := New([]float64{1, 1}, 2, 1).SetRequiresGrad(true)
	loss := A.MatMul(x).Sum()
	loss.Backward()
	// dA[i,j] = dout[i] * x[j] = 1 * x[j]
	approxEq(t, A.Grad.Data, []float64{1, 1, 1, 1}, 1e-9)
	// dx[j] = sum_i A[i,j] * dout[i] = sum_i A[i,j]
	approxEq(t, x.Grad.Data, []float64{1 + 3, 2 + 4}, 1e-9)
}

func TestAutogradChainRule(t *testing.T) {
	// y = (Wx)^2 sum; with W=[2,-1,0.5], x=[1,2,3]
	x := New([]float64{1, 2, 3}, 3, 1).SetRequiresGrad(true)
	W := New([]float64{2, -1, 0.5}, 1, 3).SetRequiresGrad(true)
	y := W.MatMul(x).Square().Sum()
	y.Backward()
	if math.Abs(y.Data[0]-2.25) > 1e-9 {
		t.Fatalf("forward: got %v want 2.25", y.Data[0])
	}
	approxEq(t, x.Grad.Data, []float64{6, -3, 1.5}, 1e-9)
	approxEq(t, W.Grad.Data, []float64{3, 6, 9}, 1e-9)
}

func TestActivations(t *testing.T) {
	x := New([]float64{-2, -1, 0, 1, 2}, 5)
	relu := x.ReLU()
	approxEq(t, relu.Data, []float64{0, 0, 0, 1, 2}, 1e-9)

	sig := x.Sigmoid()
	approxEq(t, sig.Data, []float64{
		1 / (1 + math.Exp(2)),
		1 / (1 + math.Exp(1)),
		0.5,
		1 / (1 + math.Exp(-1)),
		1 / (1 + math.Exp(-2)),
	}, 1e-9)
}

func TestSoftmaxSumsToOne(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, 4)
	s := x.Softmax(0)
	var sum float64
	for _, v := range s.Data {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Fatalf("softmax sum: got %v want 1.0", sum)
	}
}

func TestLogSoftmaxConsistency(t *testing.T) {
	x := New([]float64{1, 2, 3, 4}, 4)
	logS := x.LogSoftmax(0)
	soft := x.Softmax(0)
	for i := range logS.Data {
		if math.Abs(logS.Data[i]-math.Log(soft.Data[i])) > 1e-9 {
			t.Fatalf("log_softmax[%d]: got %v want %v", i, logS.Data[i], math.Log(soft.Data[i]))
		}
	}
}

func TestGradientCheckSigmoid(t *testing.T) {
	// numerical vs analytic gradient for f(x) = sum(sigmoid(x))
	x := New([]float64{-1.5, -0.5, 0.5, 1.5}, 4).SetRequiresGrad(true)
	y := x.Sigmoid().Sum()
	y.Backward()
	analytic := append([]float64(nil), x.Grad.Data...)
	// numeric gradient
	const eps = 1e-5
	num := make([]float64, len(x.Data))
	for i := range x.Data {
		x.Data[i] += eps
		yp := New(append([]float64(nil), x.Data...), x.Shape...).Sigmoid().Sum().Data[0]
		x.Data[i] -= 2 * eps
		ym := New(append([]float64(nil), x.Data...), x.Shape...).Sigmoid().Sum().Data[0]
		x.Data[i] += eps
		num[i] = (yp - ym) / (2 * eps)
	}
	approxEq(t, analytic, num, 1e-5)
}
