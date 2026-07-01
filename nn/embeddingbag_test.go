package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// newTestBag builds an EmbeddingBag with a hand-set 4x3 weight table.
func newTestBag(t *testing.T, mode string) *EmbeddingBag {
	t.Helper()
	e := NewEmbeddingBag(4, 3, WithBagMode(mode))
	w := []float64{
		1, 2, 3, // row 0
		-1, 0, 5, // row 1
		2, -2, 4, // row 2
		0, 7, -3, // row 3
	}
	copy(e.Weight.Data, w)
	return e
}

func TestEmbeddingBagHandChecked(t *testing.T) {
	// Bags over input [0,1,2,3,0] with offsets [0,2,2,4]:
	//   bag0 = {0,1}, bag1 = {} (empty), bag2 = {2,3}, bag3 = {0} (to end).
	input := tensor.New([]float64{0, 1, 2, 3, 0}, 5)
	offsets := tensor.New([]float64{0, 2, 2, 4}, 4)

	cases := []struct {
		mode string
		want []float64
	}{
		{"sum", []float64{
			0, 2, 8, // r0+r1
			0, 0, 0, // empty
			2, 5, 1, // r2+r3
			1, 2, 3, // r0
		}},
		{"mean", []float64{
			0, 1, 4,
			0, 0, 0,
			1, 2.5, 0.5,
			1, 2, 3,
		}},
		{"max", []float64{
			1, 2, 5, // max(r0, r1)
			0, 0, 0, // empty
			2, 7, 4, // max(r2, r3)
			1, 2, 3,
		}},
	}
	for _, c := range cases {
		e := newTestBag(t, c.mode)
		out := e.Forward(input, offsets)
		if out.Shape[0] != 4 || out.Shape[1] != 3 {
			t.Fatalf("%s: shape %v want [4 3]", c.mode, out.Shape)
		}
		for i, w := range c.want {
			if math.Abs(out.Data[i]-w) > 1e-12 {
				t.Fatalf("%s: data[%d] got %g want %g", c.mode, i, out.Data[i], w)
			}
		}
	}
}

func TestEmbeddingBagDefaultsAndValidation(t *testing.T) {
	e := NewEmbeddingBag(4, 3)
	if e.Mode != "mean" {
		t.Fatalf("default mode: got %q want mean", e.Mode)
	}

	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		f()
	}
	mustPanic("bad mode", func() { NewEmbeddingBag(4, 3, WithBagMode("median")) })
	mustPanic("nonzero first offset", func() {
		e.Forward(tensor.New([]float64{0, 1}, 2), tensor.New([]float64{1}, 1))
	})
	mustPanic("decreasing offsets", func() {
		e.Forward(tensor.New([]float64{0, 1, 2}, 3), tensor.New([]float64{0, 2, 1}, 3))
	})
	mustPanic("index out of range", func() {
		e.Forward(tensor.New([]float64{0, 9}, 2), tensor.New([]float64{0}, 1))
	})
}

func TestEmbeddingBagForward2D(t *testing.T) {
	for _, mode := range []string{"sum", "mean", "max"} {
		e := newTestBag(t, mode)
		in2d := tensor.New([]float64{0, 1, 2, 3, 0, 2}, 2, 3)
		got := e.Forward2D(in2d)

		flat := tensor.New([]float64{0, 1, 2, 3, 0, 2}, 6)
		offsets := tensor.New([]float64{0, 3}, 2)
		want := e.Forward(flat, offsets)
		tensorsEqualExact(t, mode+" Forward2D == flat+offsets", want, got)
		if got.Shape[0] != 2 || got.Shape[1] != 3 {
			t.Fatalf("%s Forward2D shape: got %v want [2 3]", mode, got.Shape)
		}
	}
}

func TestGradCheckEmbeddingBag(t *testing.T) {
	input := tensor.New([]float64{0, 2, 4, 2, 1, 3}, 6)
	offsets := tensor.New([]float64{0, 2, 2, 5}, 4) // includes an empty bag
	for _, mode := range []string{"sum", "mean", "max"} {
		e := NewEmbeddingBag(5, 3, WithBagMode(mode))
		loss := func() *tensor.Tensor { return e.Forward(input, offsets).Square().Mean() }
		gradCheck(t, "EmbeddingBag-"+mode, loss, e.Parameters(), gcEps, gcTol, 0)
	}
}
