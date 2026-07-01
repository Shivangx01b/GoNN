package nn

import (
	"testing"

	"gonn/tensor"
)

func TestModuleListAggregatesParametersInOrder(t *testing.T) {
	l0 := NewLinear(2, 3, true)
	l1 := NewLinear(3, 1, false)
	ml := NewModuleList(l0).Append(l1)

	if ml.Len() != 2 {
		t.Fatalf("Len = %d, want 2", ml.Len())
	}
	if ml.Get(0) != Module(l0) || ml.Get(1) != Module(l1) {
		t.Fatal("Get returned wrong modules")
	}

	ps := ml.Parameters()
	want := []*tensor.Tensor{l0.Weight, l0.Bias, l1.Weight}
	if len(ps) != len(want) {
		t.Fatalf("Parameters len = %d, want %d", len(ps), len(want))
	}
	for i := range want {
		if ps[i] != want[i] {
			t.Fatalf("Parameters[%d] is not the expected tensor", i)
		}
	}

	names := []string{"0.weight", "0.bias", "1.weight"}
	np := ml.NamedParameters()
	for i, n := range names {
		if np[i].Name != n {
			t.Fatalf("NamedParameters[%d].Name = %q, want %q", i, np[i].Name, n)
		}
	}
}

func TestModuleListTrainEvalPropagation(t *testing.T) {
	l0 := NewLinear(2, 2, true)
	ml := NewModuleList(l0)
	ml.Eval()
	if l0.Training() {
		t.Fatal("Eval did not propagate to list entry")
	}
	ml.Train()
	if !l0.Training() {
		t.Fatal("Train did not propagate to list entry")
	}

	// Registered on a parent, the list wires into the tree.
	parent := &struct{ Base }{}
	parent.RegisterChild("blocks", ml)
	parent.Eval()
	if l0.Training() {
		t.Fatal("parent Eval did not reach module inside ModuleList")
	}
	np := parent.NamedParameters()
	if np[0].Name != "blocks.0.weight" {
		t.Fatalf("parent NamedParameters[0].Name = %q, want blocks.0.weight", np[0].Name)
	}
}

func TestModuleDictSortedDeterministicOrder(t *testing.T) {
	lz := NewLinear(2, 2, false)
	la := NewLinear(2, 3, false)
	d := NewModuleDict().Set("z", lz).Set("a", la)

	keys := d.Keys()
	if len(keys) != 2 || keys[0] != "a" || keys[1] != "z" {
		t.Fatalf("Keys = %v, want [a z]", keys)
	}
	ps := d.Parameters()
	if len(ps) != 2 || ps[0] != la.Weight || ps[1] != lz.Weight {
		t.Fatal("Parameters not in sorted-key order")
	}
	np := d.NamedParameters()
	if np[0].Name != "a.weight" || np[1].Name != "z.weight" {
		t.Fatalf("NamedParameters names = [%q %q], want [a.weight z.weight]", np[0].Name, np[1].Name)
	}
	if d.Get("a") != Module(la) || d.Get("missing") != nil {
		t.Fatal("Get lookup wrong")
	}

	// Set replaces an existing key.
	la2 := NewLinear(2, 3, false)
	d.Set("a", la2)
	if d.Len() != 2 || d.Get("a") != Module(la2) {
		t.Fatal("Set did not replace existing entry")
	}
}

func TestModuleDictTrainEvalPropagation(t *testing.T) {
	la := NewLinear(2, 2, false)
	d := NewModuleDict().Set("a", la)
	d.Eval()
	if la.Training() {
		t.Fatal("Eval did not propagate to dict entry")
	}
	// Entries added while the dict is in eval mode start in eval mode.
	lb := NewLinear(2, 2, false)
	d.Set("b", lb)
	if lb.Training() {
		t.Fatal("entry added to an eval-mode dict should be in eval mode")
	}

	// Registered on a parent, mode and names flow through.
	parent := &struct{ Base }{}
	parent.RegisterChild("dict", d)
	parent.Train()
	if !la.Training() || !lb.Training() {
		t.Fatal("parent Train did not reach dict entries")
	}
	np := parent.NamedParameters()
	if np[0].Name != "dict.a.weight" {
		t.Fatalf("parent NamedParameters[0].Name = %q, want dict.a.weight", np[0].Name)
	}

	// Buffers flow through too (BatchNorm has running stats).
	d.Set("bn", NewBatchNorm1d(3))
	bufs := d.Buffers()
	if len(bufs) != 2 || bufs[0].Name != "bn.running_mean" || bufs[1].Name != "bn.running_var" {
		t.Fatalf("dict Buffers = %v, want bn running stats", bufs)
	}
}

func TestParameterList(t *testing.T) {
	a := tensor.Randn(2, 2)
	b := tensor.Randn(3)
	pl := NewParameterList(a).Append(b)

	if pl.Len() != 2 || pl.Get(0) != a || pl.Get(1) != b {
		t.Fatal("ParameterList Get/Len wrong")
	}
	if !a.RequiresGrad || !b.RequiresGrad {
		t.Fatal("Append should mark tensors as requiring grad")
	}
	ps := pl.Parameters()
	if len(ps) != 2 || ps[0] != a || ps[1] != b {
		t.Fatal("Parameters not in append order")
	}
	np := pl.NamedParameters()
	if np[0].Name != "0" || np[1].Name != "1" {
		t.Fatalf("NamedParameters names = [%q %q], want [0 1]", np[0].Name, np[1].Name)
	}
}

func TestParameterDictSortedDeterministicOrder(t *testing.T) {
	w1 := tensor.Randn(2)
	w2 := tensor.Randn(3)
	pd := NewParameterDict().Set("w2", w2).Set("w1", w1)

	keys := pd.Keys()
	if len(keys) != 2 || keys[0] != "w1" || keys[1] != "w2" {
		t.Fatalf("Keys = %v, want [w1 w2]", keys)
	}
	if !w1.RequiresGrad || !w2.RequiresGrad {
		t.Fatal("Set should mark tensors as requiring grad")
	}
	ps := pd.Parameters()
	if len(ps) != 2 || ps[0] != w1 || ps[1] != w2 {
		t.Fatal("Parameters not in sorted-key order")
	}
	np := pd.NamedParameters()
	if np[0].Name != "w1" || np[1].Name != "w2" {
		t.Fatalf("NamedParameters names = [%q %q], want [w1 w2]", np[0].Name, np[1].Name)
	}
	if pd.Get("w1") != w1 || pd.Get("nope") != nil {
		t.Fatal("Get lookup wrong")
	}

	// Registered on a parent, names flow through.
	parent := &struct{ Base }{}
	parent.RegisterChild("extras", pd)
	pnp := parent.NamedParameters()
	if pnp[0].Name != "extras.w1" || pnp[1].Name != "extras.w2" {
		t.Fatalf("parent names = [%q %q], want [extras.w1 extras.w2]", pnp[0].Name, pnp[1].Name)
	}
}
