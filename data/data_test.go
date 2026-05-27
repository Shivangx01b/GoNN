package data

import (
	"testing"

	"gonn/tensor"
)

func TestTensorDatasetLen(t *testing.T) {
	X := tensor.Randn(10, 4)
	Y := tensor.Zeros(10)
	d := NewTensorDataset(X, Y)
	if d.Len() != 10 {
		t.Fatalf("Len: got %v want 10", d.Len())
	}
	x, y := d.Get(3)
	if len(x.Data) != 4 {
		t.Fatalf("x sample size: got %d want 4", len(x.Data))
	}
	if len(y.Data) != 1 {
		t.Fatalf("y sample size: got %d want 1 (scalar)", len(y.Data))
	}
}

func TestDataLoaderProducesExpectedBatches(t *testing.T) {
	X := tensor.Randn(20, 3)
	Y := tensor.Zeros(20)
	d := NewTensorDataset(X, Y)
	dl := NewDataLoader(d, 4, false)
	count := 0
	for b := range dl.Iter() {
		if b.X.Shape[0] != 4 || b.X.Shape[1] != 3 {
			t.Fatalf("batch X shape: got %v want [4 3]", b.X.Shape)
		}
		count++
	}
	if count != 5 {
		t.Fatalf("loader: expected 5 batches, got %d", count)
	}
}

func TestSyntheticMakeRegression(t *testing.T) {
	X, Y := MakeRegression(50, 5, 0.1, 42)
	if X.Shape[0] != 50 || X.Shape[1] != 5 {
		t.Fatalf("X shape: got %v want [50 5]", X.Shape)
	}
	if Y.Shape[0] != 50 {
		t.Fatalf("Y shape: got %v want [50]", Y.Shape)
	}
}

func TestSyntheticMakeBlobs(t *testing.T) {
	X, Y := MakeBlobs(60, 2, 3, 1)
	if X.Shape[0] != 60 {
		t.Fatalf("X.Shape[0]: got %v want 60", X.Shape[0])
	}
	// Verify cluster ids span [0, 3)
	seen := map[int]bool{}
	for _, v := range Y.Data {
		seen[int(v)] = true
	}
	if len(seen) != 3 {
		t.Fatalf("MakeBlobs: expected 3 distinct labels, got %d", len(seen))
	}
}
