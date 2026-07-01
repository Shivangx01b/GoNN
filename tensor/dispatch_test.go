package tensor

import (
	"math"
	"testing"

	"gonn/backend"
)

// fakeEW is a Backend+Elementwiser that records dispatched calls and computes
// correct results, so dispatched ops remain checkable. decline forces the
// capability to refuse, exercising the CPU fallback path.
type fakeEW struct {
	unaryCalls, binaryCalls int
	decline                 bool
}

func (f *fakeEW) Name() backend.Device { return backend.Device("fake") }
func (f *fakeEW) Synchronize()         {}

func (f *fakeEW) Gemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	// Delegate to the real CPU backend via a temporary swap-free construction:
	// naive loop (tests only use tiny sizes).
	out := make([]float64, batch*m*n)
	at := func(bi, i, j int) float64 {
		if transA {
			return a[bi*m*k+j*m+i]
		}
		return a[bi*m*k+i*k+j]
	}
	bt := func(bi, i, j int) float64 {
		if transB {
			return b[bi*k*n+j*k+i]
		}
		return b[bi*k*n+i*n+j]
	}
	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float64
				for p := 0; p < k; p++ {
					s += at(bi, i, p) * bt(bi, p, j)
				}
				out[bi*m*n+i*n+j] = s
			}
		}
	}
	return out
}

func (f *fakeEW) Unary(kind backend.UnaryKind, a, out []float64) bool {
	if f.decline {
		return false
	}
	f.unaryCalls++
	for i, v := range a {
		switch kind {
		case backend.UnaryReLU:
			out[i] = math.Max(v, 0)
		case backend.UnaryExp:
			out[i] = math.Exp(v)
		case backend.UnaryTanh:
			out[i] = math.Tanh(v)
		case backend.UnarySigmoid:
			out[i] = 1 / (1 + math.Exp(-v))
		default:
			return false
		}
	}
	return true
}

func (f *fakeEW) Binary(kind backend.BinaryKind, a, b, out []float64) bool {
	if f.decline {
		return false
	}
	f.binaryCalls++
	for i := range a {
		switch kind {
		case backend.BinaryAdd:
			out[i] = a[i] + b[i]
		case backend.BinarySub:
			out[i] = a[i] - b[i]
		case backend.BinaryMul:
			out[i] = a[i] * b[i]
		case backend.BinaryDiv:
			out[i] = a[i] / b[i]
		}
	}
	return true
}

// withFakeBackend swaps in the fake backend and a permissive dispatch policy,
// restoring both on cleanup.
func withFakeBackend(t *testing.T, f *fakeEW, p DispatchPolicy) {
	t.Helper()
	prevB := backend.Use(f)
	prevP := GetDispatchPolicy()
	SetDispatchPolicy(p)
	t.Cleanup(func() {
		backend.Use(prevB)
		SetDispatchPolicy(prevP)
	})
}

func TestBinaryDispatchThreshold(t *testing.T) {
	f := &fakeEW{}
	withFakeBackend(t, f, DispatchPolicy{UnaryMinElems: 0, BinaryMinElems: 4})

	small := New([]float64{1, 2}, 2)
	_ = small.Add(New([]float64{3, 4}, 2))
	if f.binaryCalls != 0 {
		t.Fatalf("below-threshold binary op was dispatched (%d calls)", f.binaryCalls)
	}

	big := New([]float64{1, 2, 3, 4, 5, 6}, 6)
	got := big.Add(New([]float64{10, 20, 30, 40, 50, 60}, 6))
	if f.binaryCalls != 1 {
		t.Fatalf("above-threshold binary op not dispatched (%d calls)", f.binaryCalls)
	}
	want := []float64{11, 22, 33, 44, 55, 66}
	for i, v := range got.Data {
		if v != want[i] {
			t.Fatalf("dispatched Add[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestDispatchDeclineFallsBack(t *testing.T) {
	f := &fakeEW{decline: true}
	withFakeBackend(t, f, DispatchPolicy{UnaryMinElems: 0, BinaryMinElems: 0})

	got := New([]float64{1, 2, 3}, 3).Mul(New([]float64{2, 3, 4}, 3))
	want := []float64{2, 6, 12}
	for i, v := range got.Data {
		if v != want[i] {
			t.Fatalf("fallback Mul[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestDispatchedMatMulThroughBackend(t *testing.T) {
	f := &fakeEW{}
	withFakeBackend(t, f, GetDispatchPolicy())

	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := New([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	got := a.MatMul(b)
	want := []float64{58, 64, 139, 154}
	for i, v := range got.Data {
		if v != want[i] {
			t.Fatalf("MatMul[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestDefaultPolicyBinaryDisabled(t *testing.T) {
	p := GetDispatchPolicy()
	if p.BinaryMinElems != math.MaxInt {
		t.Fatalf("default BinaryMinElems = %d, want MaxInt (disabled)", p.BinaryMinElems)
	}
	if p.UnaryMinElems != 1<<18 {
		t.Fatalf("default UnaryMinElems = %d, want %d", p.UnaryMinElems, 1<<18)
	}
}
