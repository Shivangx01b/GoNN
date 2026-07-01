package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

func assertData(t *testing.T, what string, got *tensor.Tensor, want []float64) {
	t.Helper()
	if got.Numel() != len(want) {
		t.Fatalf("%s: got %d elements, want %d", what, got.Numel(), len(want))
	}
	for i, w := range want {
		if math.Abs(got.Data[i]-w) > 1e-12 {
			t.Fatalf("%s: data[%d] = %v, want %v (full: %v)", what, i, got.Data[i], w, got.Data)
		}
	}
}

func TestForwardPreHookReplacesInput(t *testing.T) {
	m := NewIdentity()
	id := m.RegisterForwardPreHook(func(c Child, x *tensor.Tensor) *tensor.Tensor {
		if c != Child(m) {
			t.Errorf("pre-hook received wrong module")
		}
		return x.MulScalar(2)
	})
	defer m.RemoveHook(id)

	assertData(t, "Call with pre-hook", Call(m, tensor.New([]float64{1, 2, 3}, 3)), []float64{2, 4, 6})

	// Bare Forward bypasses hooks entirely.
	assertData(t, "bare Forward", m.Forward(tensor.New([]float64{1, 2, 3}, 3)), []float64{1, 2, 3})
}

func TestForwardPreHookNilReturnLeavesInputUnchanged(t *testing.T) {
	m := NewIdentity()
	fired := false
	id := m.RegisterForwardPreHook(func(c Child, x *tensor.Tensor) *tensor.Tensor {
		fired = true
		return nil
	})
	defer m.RemoveHook(id)

	assertData(t, "nil-return pre-hook", Call(m, tensor.New([]float64{5, 6}, 2)), []float64{5, 6})
	if !fired {
		t.Fatal("pre-hook did not fire through Call")
	}
}

func TestForwardHookReplacesOutput(t *testing.T) {
	m := NewIdentity()
	id := m.RegisterForwardHook(func(c Child, x, y *tensor.Tensor) *tensor.Tensor {
		// x is the (possibly pre-hook-replaced) input; y the raw output.
		if x.Data[0] != y.Data[0] {
			t.Errorf("identity forward hook saw x[0]=%v y[0]=%v", x.Data[0], y.Data[0])
		}
		return y.AddScalar(3)
	})
	defer m.RemoveHook(id)

	assertData(t, "Call with forward hook", Call(m, tensor.New([]float64{1, 2}, 2)), []float64{4, 5})
}

func TestBackwardHookObservesAndScalesGradient(t *testing.T) {
	l := NewLinear(3, 2, true)
	x := seededRandn(60, 4, 3)

	// Baseline gradients without hooks.
	Call(l, x).Sum().Backward()
	baseW := append([]float64(nil), l.Weight.Grad.Data...)
	baseB := append([]float64(nil), l.Bias.Grad.Data...)
	for _, p := range l.Parameters() {
		p.ZeroGrad()
	}

	observed := false
	id := l.RegisterFullBackwardHook(func(c Child, g *tensor.Tensor) *tensor.Tensor {
		observed = true
		// The gradient of Sum() w.r.t. the output is all ones.
		for _, v := range g.Data {
			if math.Abs(v-1) > 1e-12 {
				t.Errorf("backward hook saw grad element %v, want 1", v)
			}
		}
		return g.MulScalar(2)
	})
	defer l.RemoveHook(id)

	Call(l, x).Sum().Backward()
	if !observed {
		t.Fatal("backward hook never fired")
	}
	for i, w := range baseW {
		if math.Abs(l.Weight.Grad.Data[i]-2*w) > 1e-10 {
			t.Fatalf("weight grad[%d] = %v, want doubled %v", i, l.Weight.Grad.Data[i], 2*w)
		}
	}
	for i, b := range baseB {
		if math.Abs(l.Bias.Grad.Data[i]-2*b) > 1e-10 {
			t.Fatalf("bias grad[%d] = %v, want doubled %v", i, l.Bias.Grad.Data[i], 2*b)
		}
	}
}

func TestBackwardHookSkippedWhenOutputHasNoGrad(t *testing.T) {
	m := NewIdentity()
	id := m.RegisterFullBackwardHook(func(c Child, g *tensor.Tensor) *tensor.Tensor {
		t.Error("backward hook fired for a grad-free graph")
		return nil
	})
	defer m.RemoveHook(id)

	y := Call(m, tensor.New([]float64{1, 2}, 2)) // input does not require grad
	if y.RequiresGrad {
		t.Fatal("output of a grad-free Call should not require grad")
	}
	assertData(t, "grad-free Call", y, []float64{1, 2})
}

func TestGlobalHooksFireForEveryChildInSequential(t *testing.T) {
	forwardCount := 0
	fid := RegisterModuleForwardHook(func(m Child, x, y *tensor.Tensor) *tensor.Tensor {
		forwardCount++
		return nil
	})
	defer RemoveModuleHook(fid)
	preCount := 0
	pid := RegisterModuleForwardPreHook(func(m Child, x *tensor.Tensor) *tensor.Tensor {
		preCount++
		return nil
	})
	defer RemoveModuleHook(pid)
	backwardCount := 0
	bid := RegisterModuleFullBackwardHook(func(m Child, g *tensor.Tensor) *tensor.Tensor {
		backwardCount++
		return nil
	})
	defer RemoveModuleHook(bid)

	seq := NewSequential(NewLinear(2, 4, true), ReLU(), NewLinear(4, 1, true))
	x := seededRandn(61, 3, 2)

	seq.Forward(x).Sum().Backward()
	if forwardCount != 3 || preCount != 3 {
		t.Fatalf("seq.Forward: forward hooks fired %d, pre hooks %d, want 3 each", forwardCount, preCount)
	}
	if backwardCount != 3 {
		t.Fatalf("global backward hook fired %d times, want 3 (once per child)", backwardCount)
	}

	// Call on the Sequential itself adds one more pipeline pass for seq.
	forwardCount, preCount = 0, 0
	Call(seq, x)
	if forwardCount != 4 || preCount != 4 {
		t.Fatalf("Call(seq): forward hooks fired %d, pre hooks %d, want 4 each (3 children + seq)", forwardCount, preCount)
	}

	// Removal stops them.
	if !RemoveModuleHook(fid) || !RemoveModuleHook(pid) || !RemoveModuleHook(bid) {
		t.Fatal("RemoveModuleHook did not find registered hooks")
	}
	forwardCount, preCount, backwardCount = 0, 0, 0
	seq.Forward(x).Sum().Backward()
	if forwardCount != 0 || preCount != 0 || backwardCount != 0 {
		t.Fatalf("hooks fired after removal: fwd=%d pre=%d bwd=%d", forwardCount, preCount, backwardCount)
	}
}

func TestGlobalHooksRunBeforePerModuleHooks(t *testing.T) {
	var order []string
	gid := RegisterModuleForwardPreHook(func(m Child, x *tensor.Tensor) *tensor.Tensor {
		order = append(order, "global")
		return nil
	})
	defer RemoveModuleHook(gid)

	m := NewIdentity()
	id := m.RegisterForwardPreHook(func(c Child, x *tensor.Tensor) *tensor.Tensor {
		order = append(order, "module")
		return nil
	})
	defer m.RemoveHook(id)

	Call(m, tensor.New([]float64{1}, 1))
	if len(order) != 2 || order[0] != "global" || order[1] != "module" {
		t.Fatalf("hook order = %v, want [global module]", order)
	}
}

func TestRemoveHookPerModule(t *testing.T) {
	m := NewIdentity()
	id := m.RegisterForwardPreHook(func(c Child, x *tensor.Tensor) *tensor.Tensor {
		return x.MulScalar(10)
	})
	if !m.RemoveHook(id) {
		t.Fatal("RemoveHook did not find the hook")
	}
	if m.RemoveHook(id) {
		t.Fatal("RemoveHook found an already-removed hook")
	}
	assertData(t, "after removal", Call(m, tensor.New([]float64{1, 2}, 2)), []float64{1, 2})
}

func TestHookPipelineChainsInsideSequential(t *testing.T) {
	// A per-module forward hook on an inner layer is honored by
	// Sequential.Forward (the container routes children through Call).
	inner := NewIdentity()
	id := inner.RegisterForwardHook(func(c Child, x, y *tensor.Tensor) *tensor.Tensor {
		return y.MulScalar(3)
	})
	defer inner.RemoveHook(id)

	seq := NewSequential(inner, NewIdentity())
	assertData(t, "sequential with hooked child", seq.Forward(tensor.New([]float64{1, 2}, 2)), []float64{3, 6})
}
