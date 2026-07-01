package prune

import (
	"math"
	"testing"

	"gonn/tensor"
)

func wantData(t *testing.T, p *tensor.Tensor, want []float64) {
	t.Helper()
	for i := range want {
		if p.Data[i] != want[i] {
			t.Fatalf("data mismatch at %d: got %v, want %v", i, p.Data, want)
		}
	}
}

func TestL1UnstructuredHandCheck(t *testing.T) {
	p := tensor.New([]float64{1, -3, 2, -4}, 2, 2)
	L1Unstructured(p, 0.5) // k = 2: prunes |1| and |2|
	wantData(t, p, []float64{0, -3, 0, -4})
	if !IsPruned(p) {
		t.Errorf("IsPruned = false after L1Unstructured")
	}
	m := Mask(p)
	for i, want := range []float64{0, 1, 0, 1} {
		if m[i] != want {
			t.Errorf("mask[%d] = %g, want %g", i, m[i], want)
		}
	}
	if o := Orig(p); o[0] != 1 || o[2] != 2 {
		t.Errorf("orig snapshot lost the pruned values: %v", o)
	}
}

func TestGlobalUnstructuredThresholdAcrossTensors(t *testing.T) {
	t1 := tensor.New([]float64{1, 5}, 2)
	t2 := tensor.New([]float64{2, 3}, 2)
	GlobalUnstructured([]*tensor.Tensor{t1, t2}, 0.5) // k = 2: global smallest are 1 and 2
	wantData(t, t1, []float64{0, 5})
	wantData(t, t2, []float64{0, 3})
}

func TestLnStructuredPrunesWholeRows(t *testing.T) {
	p := tensor.New([]float64{
		0.1, -0.1, 0.1, 0.1, // row 0: smallest L2 norm
		5, -5, 5, 5, // row 1: largest
		1, -1, 1, 1, // row 2: middle
	}, 3, 4)
	LnStructured(p, 1.0/3.0, 2, 0) // prune 1 of 3 rows by L2 norm
	wantData(t, p, []float64{
		0, 0, 0, 0,
		5, -5, 5, 5,
		1, -1, 1, 1,
	})
}

func TestLnStructuredInfNormColumns(t *testing.T) {
	p := tensor.New([]float64{
		1, 9, 2, 8,
		-3, 1, 1, 7,
	}, 2, 4)
	// Column max-abs norms: 3, 9, 2, 8 -> prune columns 2 and 0.
	LnStructured(p, 0.5, math.Inf(1), 1)
	wantData(t, p, []float64{
		0, 9, 0, 8,
		0, 1, 0, 7,
	})
}

func TestRandomStructuredPrunesWholeSlices(t *testing.T) {
	data := make([]float64, 12)
	for i := range data {
		data[i] = float64(i + 1) // no natural zeros
	}
	p := tensor.New(data, 4, 3)
	RandomStructured(p, 0.5, 0, 7) // prune 2 of 4 rows
	zeroRows := 0
	for r := 0; r < 4; r++ {
		zeros := 0
		for c := 0; c < 3; c++ {
			if p.Data[r*3+c] == 0 {
				zeros++
			}
		}
		if zeros != 0 && zeros != 3 {
			t.Errorf("row %d partially pruned (%d of 3 zeros); structured pruning must drop whole rows", r, zeros)
		}
		if zeros == 3 {
			zeroRows++
		}
	}
	if zeroRows != 2 {
		t.Errorf("pruned %d rows, want 2", zeroRows)
	}
}

func TestRandomUnstructuredCountAndDeterminism(t *testing.T) {
	mk := func() *tensor.Tensor {
		d := make([]float64, 20)
		for i := range d {
			d[i] = 1
		}
		return tensor.New(d, 20)
	}
	a, b := mk(), mk()
	RandomUnstructured(a, 0.25, 3)
	RandomUnstructured(b, 0.25, 3)
	zeros := 0
	for i := range a.Data {
		if a.Data[i] == 0 {
			zeros++
		}
		if a.Data[i] != b.Data[i] {
			t.Fatalf("same seed produced different masks")
		}
	}
	if zeros != 5 {
		t.Errorf("pruned %d elements, want round(0.25*20) = 5", zeros)
	}
}

func TestReapplyAfterOptimizerStep(t *testing.T) {
	p := tensor.New([]float64{1, 2, 3, 4}, 4)
	L1Unstructured(p, 0.5)
	wantData(t, p, []float64{0, 0, 3, 4})

	// Simulated optimizer step: every entry drifts (weight decay, momentum...).
	for i := range p.Data {
		p.Data[i] += 10
	}
	Reapply(p)
	wantData(t, p, []float64{0, 0, 13, 14})
}

func TestRemoveMakesPermanent(t *testing.T) {
	p := tensor.New([]float64{1, 2, 3, 4}, 4)
	L1Unstructured(p, 0.5)
	Remove(p)
	if IsPruned(p) {
		t.Errorf("IsPruned = true after Remove")
	}
	if Mask(p) != nil || Orig(p) != nil {
		t.Errorf("mask/orig survived Remove")
	}
	wantData(t, p, []float64{0, 0, 3, 4}) // current (masked) data kept

	// Reapply is now a no-op: pruning is permanent, new values stay.
	p.Data[0] = 9
	Reapply(p)
	wantData(t, p, []float64{9, 0, 3, 4})
}

func TestIdentityAndCustomFromMask(t *testing.T) {
	p := tensor.New([]float64{1, 2, 3}, 3)
	Identity(p)
	if !IsPruned(p) {
		t.Errorf("Identity did not register a mask")
	}
	wantData(t, p, []float64{1, 2, 3})

	q := tensor.New([]float64{1, 2, 3, 4}, 4)
	CustomFromMask(q, []float64{1, 0, 1, 0})
	wantData(t, q, []float64{1, 0, 3, 0})
}

func TestIterativePruningCombinesMasks(t *testing.T) {
	p := tensor.New([]float64{1, 2, 3, 4}, 4)
	CustomFromMask(p, []float64{0, 1, 1, 1})
	CustomFromMask(p, []float64{1, 1, 0, 1})
	wantData(t, p, []float64{0, 2, 0, 4})
	m := Mask(p)
	for i, want := range []float64{0, 1, 0, 1} {
		if m[i] != want {
			t.Errorf("combined mask[%d] = %g, want %g", i, m[i], want)
		}
	}
	// Orig still snapshots the values from the first prune.
	if o := Orig(p); o[0] != 1 || o[2] != 3 {
		t.Errorf("orig snapshot changed across iterative prunes: %v", o)
	}
}

func TestAmountValidation(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("amount > 1 did not panic")
		}
	}()
	L1Unstructured(tensor.New([]float64{1, 2}, 2), 1.5)
}

func TestMaskLengthValidation(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("wrong mask length did not panic")
		}
	}()
	Prune(tensor.New([]float64{1, 2}, 2), []float64{1})
}
