package nn

import (
	"testing"

	"gonn/tensor"
)

// snapshotParams deep-copies every named parameter's data.
func snapshotParams(c Child) map[string][]float64 {
	out := make(map[string][]float64)
	for _, p := range c.NamedParameters() {
		out[p.Name] = append([]float64(nil), p.T.Data...)
	}
	return out
}

// requireBitEqual asserts two slices are elementwise bit-identical.
func requireBitEqual(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: %d vs %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: element %d differs: %g vs %g", name, i, got[i], want[i])
		}
	}
}

func TestFunctionalCallMatchesPermanentSwap(t *testing.T) {
	m := NewSequential(NewLinear(4, 3, true), NewLinear(3, 2, true))
	before := snapshotParams(m)

	repW := seededRandn(401, 3, 4) // replaces "0.weight"
	repB := seededRandn(402, 2)    // replaces "1.bias"
	x := seededRandn(403, 5, 4)

	got := FunctionalCall(m, x, map[string]*tensor.Tensor{
		"0.weight": repW,
		"1.bias":   repB,
	})

	// Reference: an identically-shaped module whose params are PERMANENTLY
	// set to m's values with the two replacements applied.
	perm := NewSequential(NewLinear(4, 3, true), NewLinear(3, 2, true))
	for i, p := range perm.NamedParameters() {
		copy(p.T.Data, m.NamedParameters()[i].T.Data)
	}
	for _, p := range perm.NamedParameters() {
		switch p.Name {
		case "0.weight":
			copy(p.T.Data, repW.Data)
		case "1.bias":
			copy(p.T.Data, repB.Data)
		}
	}
	want := perm.Forward(x)
	requireBitEqual(t, "FunctionalCall output", got.Data, want.Data)

	// Originals restored bit-exactly after the call.
	for _, p := range m.NamedParameters() {
		requireBitEqual(t, "restored "+p.Name, p.T.Data, before[p.Name])
	}
}

func TestFunctionalCallFiresHooks(t *testing.T) {
	m := NewLinear(3, 2, true)
	fired := 0
	id := m.RegisterForwardHook(func(Child, *tensor.Tensor, *tensor.Tensor) *tensor.Tensor {
		fired++
		return nil
	})
	defer m.RemoveHook(id)

	FunctionalCall(m, seededRandn(404, 2, 3), map[string]*tensor.Tensor{
		"weight": seededRandn(405, 2, 3),
	})
	if fired != 1 {
		t.Errorf("forward hook fired %d times through FunctionalCall, want 1", fired)
	}
}

func TestFunctionalCallGradMatchesManualSwap(t *testing.T) {
	m := NewLinear(4, 3, true)
	before := snapshotParams(m)
	// Pre-existing gradient: must survive the call untouched (same pointer,
	// same values).
	preGrad := tensor.Ones(3, 4)
	m.Weight.Grad = preGrad

	repW := seededRandn(406, 3, 4)
	x := seededRandn(407, 2, 4)
	loss := func(y *tensor.Tensor) *tensor.Tensor { return y.Square().Mean() }

	lossVal, grads := FunctionalCallGrad(m, x,
		map[string]*tensor.Tensor{"weight": repW}, loss)

	// Manual reference: permanently swap the weight into a fresh Linear and
	// run the identical forward+backward.
	ref := NewLinear(4, 3, true)
	copy(ref.Weight.Data, repW.Data)
	copy(ref.Bias.Data, m.Bias.Data)
	l := ref.Forward(x).Square().Mean()
	l.Backward()

	if lossVal != l.Item() {
		t.Errorf("lossVal = %g, manual swap gives %g", lossVal, l.Item())
	}
	requireBitEqual(t, "grads[weight]", grads["weight"], ref.Weight.Grad.Data)
	requireBitEqual(t, "grads[bias]", grads["bias"], ref.Bias.Grad.Data)

	// Module fully restored: data bit-identical, weight's pre-existing Grad
	// pointer and values back, bias grad still absent.
	for _, p := range m.NamedParameters() {
		requireBitEqual(t, "restored "+p.Name, p.T.Data, before[p.Name])
	}
	if m.Weight.Grad != preGrad {
		t.Errorf("weight Grad pointer was not restored")
	}
	requireBitEqual(t, "pre-existing weight grad", m.Weight.Grad.Data, tensor.Ones(3, 4).Data)
	if m.Bias.Grad != nil {
		t.Errorf("bias Grad leaked out of FunctionalCallGrad")
	}
}

func TestWithReplacedParamsRestoresOnPanic(t *testing.T) {
	m := NewLinear(3, 2, true)
	before := snapshotParams(m)
	origWeightData := m.Weight.Data // slice identity must come back too

	func() {
		defer func() {
			if recover() == nil {
				t.Fatalf("expected the inner panic to propagate")
			}
		}()
		WithReplacedParams(m, map[string]*tensor.Tensor{"weight": seededRandn(408, 2, 3)}, func() {
			panic("boom")
		})
	}()

	if &m.Weight.Data[0] != &origWeightData[0] {
		t.Errorf("weight Data slice was not restored after panic")
	}
	for _, p := range m.NamedParameters() {
		requireBitEqual(t, "post-panic "+p.Name, p.T.Data, before[p.Name])
	}
}

func TestFunctionalCallValidationPanics(t *testing.T) {
	m := NewLinear(4, 3, true)
	x := seededRandn(409, 2, 4)

	mustPanic(t, "shape mismatch", func() {
		FunctionalCall(m, x, map[string]*tensor.Tensor{"weight": seededRandn(410, 4, 3)})
	})
	mustPanic(t, "unknown name", func() {
		FunctionalCall(m, x, map[string]*tensor.Tensor{"weight_orig": seededRandn(411, 3, 4)})
	})
	mustPanic(t, "nil replacement", func() {
		FunctionalCall(m, x, map[string]*tensor.Tensor{"weight": nil})
	})

	// Validation failures must leave the module untouched (no partial swap).
	before := snapshotParams(m)
	mustPanic(t, "partial-swap guard", func() {
		FunctionalCall(m, x, map[string]*tensor.Tensor{
			"weight":  seededRandn(412, 3, 4),
			"missing": seededRandn(413, 3),
		})
	})
	for _, p := range m.NamedParameters() {
		requireBitEqual(t, "after failed validation "+p.Name, p.T.Data, before[p.Name])
	}
}
