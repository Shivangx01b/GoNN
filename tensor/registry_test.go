package tensor

import (
	"math"
	"sort"
	"testing"
)

func TestUnaryByNameMatchesFluent(t *testing.T) {
	x := New([]float64{-2, -0.5, 0, 0.5, 1, 3}, 6)
	cases := []struct {
		name   string
		fluent func(*Tensor) *Tensor
	}{
		{"relu", (*Tensor).ReLU},
		{"gelu", (*Tensor).GELU},
		{"silu", (*Tensor).SiLU},
		{"sigmoid", (*Tensor).Sigmoid},
		{"tanh", (*Tensor).Tanh},
		{"mish", (*Tensor).Mish},
		{"softplus", (*Tensor).Softplus},
		{"exp", (*Tensor).Exp},
		{"square", (*Tensor).Square},
	}
	for _, c := range cases {
		byName := x.Unary(c.name)
		fluent := c.fluent(x)
		for i := range byName.Data {
			if byName.Data[i] != fluent.Data[i] {
				t.Fatalf("%s: Unary(name)[%d]=%v != fluent %v", c.name, i, byName.Data[i], fluent.Data[i])
			}
		}
	}
}

func TestUnaryByNameGrad(t *testing.T) {
	x := New([]float64{-1, 0.5, 2}, 3).SetRequiresGrad(true)
	y := x.Unary("GELU").Sum() // case-insensitive
	y.Backward()
	if x.Grad == nil {
		t.Fatal("no grad through Unary(name)")
	}
	// Compare against fluent-path gradient.
	x2 := New([]float64{-1, 0.5, 2}, 3).SetRequiresGrad(true)
	x2.GELU().Sum().Backward()
	for i := range x.Grad.Data {
		if x.Grad.Data[i] != x2.Grad.Data[i] {
			t.Fatalf("grad[%d]: %v != %v", i, x.Grad.Data[i], x2.Grad.Data[i])
		}
	}
}

func TestUnaryOpNamesSortedAndComplete(t *testing.T) {
	names := UnaryOpNames()
	if !sort.StringsAreSorted(names) {
		t.Fatal("UnaryOpNames not sorted")
	}
	for _, want := range []string{"relu", "relu6", "sigmoid", "tanh", "gelu", "silu",
		"selu", "hardtanh", "hardsigmoid", "hardswish", "softplus", "softsign",
		"mish", "logsigmoid", "tanhshrink", "exp", "log", "sqrt", "abs", "square"} {
		if _, ok := LookupUnary(want); !ok {
			t.Fatalf("expected %q in registry; have %v", want, names)
		}
	}
}

func TestUnaryUnknownNamePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for unknown op name")
		}
	}()
	Zeros(3).Unary("definitely-not-an-op")
}

func TestRegisterUnaryDuplicatePanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on duplicate registration")
		}
	}()
	RegisterUnary(UnaryOpDef{Name: "ReLU", Kind: UnaryNone,
		Fwd: func(x float64) float64 { return x },
		Bwd: func(g, x, y float64) float64 { return g }})
}

// TestUnaryDispatchThreshold verifies registered ops route through the
// backend capability above the threshold, with correct values and grads.
func TestUnaryDispatchThreshold(t *testing.T) {
	f := &fakeEW{}
	withFakeBackend(t, f, DispatchPolicy{UnaryMinElems: 4, BinaryMinElems: math.MaxInt})

	small := New([]float64{-1, 2}, 2)
	_ = small.Tanh()
	if f.unaryCalls != 0 {
		t.Fatalf("below-threshold unary op was dispatched (%d calls)", f.unaryCalls)
	}

	big := New([]float64{-2, -1, 0, 1, 2, 3}, 6).SetRequiresGrad(true)
	got := big.Tanh()
	if f.unaryCalls != 1 {
		t.Fatalf("above-threshold unary op not dispatched (%d calls)", f.unaryCalls)
	}
	for i, v := range big.Data {
		if want := math.Tanh(v); got.Data[i] != want {
			t.Fatalf("dispatched Tanh[%d] = %v, want %v", i, got.Data[i], want)
		}
	}
	// Backward runs the host closure even when forward was dispatched.
	got.Sum().Backward()
	for i, v := range big.Data {
		th := math.Tanh(v)
		if want := 1 - th*th; math.Abs(big.Grad.Data[i]-want) > 1e-15 {
			t.Fatalf("grad[%d] = %v, want %v", i, big.Grad.Data[i], want)
		}
	}

	// Ops without a kernel (UnaryNone) never dispatch.
	calls := f.unaryCalls
	_ = big.Mish()
	if f.unaryCalls != calls {
		t.Fatal("UnaryNone op was dispatched")
	}
}
