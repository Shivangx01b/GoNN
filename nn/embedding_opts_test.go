package nn

import (
	"math"
	"math/rand"
	"testing"

	"gonn/tensor"
)

func mustPanicOpts(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic", name)
		}
	}()
	f()
}

func l2(row []float64) float64 {
	var s float64
	for _, x := range row {
		s += x * x
	}
	return math.Sqrt(s)
}

// Construction: the padding row is zeroed AFTER the full RNG draw, so every
// other weight is bit-identical to the option-free constructor (this is what
// keeps the TestModuleParity goldens valid: same draw sequence, and PyTorch
// likewise fills N(0,1) first and then zeroes the row).
func TestEmbeddingPaddingIdxConstruction(t *testing.T) {
	rand.Seed(4242)
	plain := NewEmbedding(5, 3)
	rand.Seed(4242)
	padded := NewEmbedding(5, 3, WithPaddingIdx(2))

	for r := 0; r < 5; r++ {
		for c := 0; c < 3; c++ {
			got := padded.Weight.Data[r*3+c]
			if r == 2 {
				if got != 0 {
					t.Fatalf("padding row not zeroed: [2][%d] = %g", c, got)
				}
				continue
			}
			if want := plain.Weight.Data[r*3+c]; got != want {
				t.Fatalf("RNG draw order changed: [%d][%d] got %g want %g", r, c, got, want)
			}
		}
	}

	// Negative index wraps Python-style, like PyTorch.
	neg := NewEmbedding(5, 3, WithPaddingIdx(-3))
	if neg.PaddingIdx != 2 {
		t.Fatalf("negative padding_idx: got %d want 2", neg.PaddingIdx)
	}

	mustPanicOpts(t, "padding_idx too large", func() { NewEmbedding(5, 3, WithPaddingIdx(5)) })
	mustPanicOpts(t, "padding_idx too negative", func() { NewEmbedding(5, 3, WithPaddingIdx(-6)) })
	mustPanicOpts(t, "negative max_norm", func() { NewEmbedding(5, 3, WithMaxNorm(-1)) })
	mustPanicOpts(t, "bad norm_type", func() { NewEmbedding(5, 3, WithMaxNorm(1), WithNormType(0)) })
	mustPanicOpts(t, "bag padding_idx out of range", func() { NewEmbeddingBag(4, 3, WithBagPaddingIdx(4)) })
	mustPanicOpts(t, "bag negative max_norm", func() { NewEmbeddingBag(4, 3, WithBagMaxNorm(-2)) })
}

// Forward emits the CURRENT padding-row values at padded positions (the row
// can be updated after construction, PyTorch semantics), the whole output is
// bit-identical to a padding-free Embedding with the same weights, and after
// Backward the padding row's gradient is exactly zero while every other row
// gets the same scatter-add gradient a padding-free run produces.
func TestEmbeddingPaddingIdxForwardBackward(t *testing.T) {
	w := []float64{
		1, 2, 3,
		-1, 0, 5,
		0.5, -0.5, 1.5, // padding row, deliberately nonzero (updated post-init)
		2, -2, 4,
		0, 7, -3,
	}
	pad := NewEmbedding(5, 3, WithPaddingIdx(2))
	copy(pad.Weight.Data, w)
	plain := NewEmbedding(5, 3)
	copy(plain.Weight.Data, w)

	idx := tensor.New([]float64{0, 2, 4, 2, 1}, 5) // pads at positions 1 and 3

	outPad := pad.Forward(idx)
	outPlain := plain.Forward(idx)
	tensorsEqualExact(t, "padding_idx forward values", outPlain, outPad)
	for _, pos := range []int{1, 3} {
		for c := 0; c < 3; c++ {
			if got, want := outPad.Data[pos*3+c], w[2*3+c]; got != want {
				t.Fatalf("pad position %d col %d: got %g want %g", pos, c, got, want)
			}
		}
	}

	pad.Weight.ZeroGrad()
	plain.Weight.ZeroGrad()
	outPad.Square().Mean().Backward()
	outPlain.Square().Mean().Backward()

	gp := pad.Weight.Grad.Data
	gn := plain.Weight.Grad.Data
	for r := 0; r < 5; r++ {
		for c := 0; c < 3; c++ {
			j := r*3 + c
			if r == 2 {
				if gp[j] != 0 {
					t.Fatalf("padding-row grad not exactly zero: [2][%d] = %g", c, gp[j])
				}
				continue
			}
			if gp[j] != gn[j] {
				t.Fatalf("non-pad grad changed by padding_idx: [%d][%d] got %g want %g", r, c, gp[j], gn[j])
			}
		}
	}
	// The control run must have a nonzero grad at the padding row, or the
	// exclusion assertions above would be vacuous.
	nz := false
	for c := 0; c < 3; c++ {
		if gn[2*3+c] != 0 {
			nz = true
		}
	}
	if !nz {
		t.Fatal("control grad at padding row unexpectedly zero")
	}
}

func TestGradCheckEmbeddingPaddingIdx(t *testing.T) {
	e := NewEmbedding(5, 3, WithPaddingIdx(2))
	// Nonzero padding row so the FD probe at the end is meaningful.
	for c := 0; c < 3; c++ {
		e.Weight.Data[2*3+c] = 0.5 * float64(c+1)
	}
	idx := tensor.New([]float64{0, 2, 4, 2, 1}, 5)
	loss := func() *tensor.Tensor { return e.Forward(idx).Square().Mean() }

	w := e.Weight
	w.ZeroGrad()
	loss().Backward()
	analytic := append([]float64(nil), w.Grad.Data...)

	fd := func(j int) float64 {
		orig := w.Data[j]
		w.Data[j] = orig + gcEps
		fp := loss().Item()
		w.Data[j] = orig - gcEps
		fm := loss().Item()
		w.Data[j] = orig
		return (fp - fm) / (2 * gcEps)
	}

	// Finite differences over NON-PAD rows only. The padding row is skipped
	// on purpose: the OUTPUT depends on it (padded positions emit the row
	// verbatim), so FD reports a nonzero derivative there while the analytic
	// gradient is zero BY DEFINITION — exactly PyTorch's padding_idx
	// semantics (gradient excluded, output not).
	for r := 0; r < 5; r++ {
		if r == 2 {
			continue
		}
		for c := 0; c < 3; c++ {
			j := r*3 + c
			num, got := fd(j), analytic[j]
			denom := math.Max(1, math.Abs(num)+math.Abs(got))
			if math.Abs(num-got)/denom > gcTol {
				t.Errorf("weight[%d][%d]: analytic=%.8g numeric=%.8g", r, c, got, num)
			}
		}
	}

	// Targeted pad-row probe: analytic grad exactly zero while the FD
	// derivative is nonzero (the loss really does move with the row). The
	// intentional mismatch documents the semantics above.
	j := 2 * 3
	if analytic[j] != 0 {
		t.Fatalf("pad-row analytic grad: got %g want exact 0", analytic[j])
	}
	if num := fd(j); math.Abs(num) < 1e-6 {
		t.Fatalf("expected nonzero FD derivative through the pad row, got %g", num)
	}
}

func TestEmbeddingMaxNorm(t *testing.T) {
	e := NewEmbedding(4, 3, WithMaxNorm(1.5))
	w := []float64{
		3, 0, 4, // row 0: L2 norm 5, referenced -> renormed to 1.5
		0.3, 0.4, 0, // row 1: L2 norm 0.5, referenced, under limit -> untouched
		2, 2, 1, // row 2: L2 norm 3, NOT referenced -> untouched
		6, 8, 0, // row 3: L2 norm 10, NOT referenced -> untouched
	}
	copy(e.Weight.Data, w)

	idx := tensor.New([]float64{0, 1, 0}, 3) // row 0 twice: dedup renorms once
	out := e.Forward(idx)

	r0 := e.Weight.Data[0:3]
	if n := l2(r0); math.Abs(n-1.5) > 1e-12 {
		t.Fatalf("row 0 norm: got %.17g want 1.5", n)
	}
	want0 := []float64{3 * 1.5 / 5, 0, 4 * 1.5 / 5}
	for c := range want0 {
		if math.Abs(r0[c]-want0[c]) > 1e-12 {
			t.Fatalf("row 0 direction changed: [%d] got %g want %g", c, r0[c], want0[c])
		}
	}
	// Under-limit and unreferenced rows are bit-identical.
	for j := 3; j < 12; j++ {
		if e.Weight.Data[j] != w[j] {
			t.Fatalf("weight[%d] modified: got %g want %g", j, e.Weight.Data[j], w[j])
		}
	}
	// Output reflects the renormalized table (in-place renorm precedes lookup).
	for c := 0; c < 3; c++ {
		if out.Data[c] != e.Weight.Data[c] || out.Data[2*3+c] != e.Weight.Data[c] {
			t.Fatal("output does not reflect renormalized row 0")
		}
		if out.Data[3+c] != e.Weight.Data[3+c] {
			t.Fatal("output row 1 mismatch")
		}
	}
}

func TestEmbeddingMaxNormL1(t *testing.T) {
	e := NewEmbedding(2, 3, WithMaxNorm(2), WithNormType(1))
	w := []float64{1, -2, 3, 0.5, 0.5, -0.5} // L1 norms 6 (over) and 1.5 (under)
	copy(e.Weight.Data, w)
	e.Forward(tensor.New([]float64{0, 1}, 2))

	r0 := e.Weight.Data[0:3]
	if n := math.Abs(r0[0]) + math.Abs(r0[1]) + math.Abs(r0[2]); math.Abs(n-2) > 1e-12 {
		t.Fatalf("row 0 L1 norm: got %.17g want 2", n)
	}
	want := []float64{1.0 / 3, -2.0 / 3, 1}
	for c := range want {
		if math.Abs(r0[c]-want[c]) > 1e-12 {
			t.Fatalf("row 0: [%d] got %g want %g", c, r0[c], want[c])
		}
	}
	for j := 3; j < 6; j++ {
		if e.Weight.Data[j] != w[j] {
			t.Fatalf("row 1 modified: [%d] got %g want %g", j, e.Weight.Data[j], w[j])
		}
	}
}

func TestEmbeddingBagPaddingIdxHandChecked(t *testing.T) {
	// Construction zeroes the padding row after the full draw (same
	// draw-order guarantee as Embedding).
	rand.Seed(777)
	plain := NewEmbeddingBag(4, 3)
	rand.Seed(777)
	pb := NewEmbeddingBag(4, 3, WithBagPaddingIdx(1))
	for r := 0; r < 4; r++ {
		for c := 0; c < 3; c++ {
			got := pb.Weight.Data[r*3+c]
			if r == 1 {
				if got != 0 {
					t.Fatalf("bag padding row not zeroed: [1][%d] = %g", c, got)
				}
				continue
			}
			if want := plain.Weight.Data[r*3+c]; got != want {
				t.Fatalf("bag RNG draw order changed: [%d][%d] got %g want %g", r, c, got, want)
			}
		}
	}

	// Hand-checked reductions with a deliberately NONZERO padding row: the
	// exclusion must come from the index, not from the row being zero.
	w := []float64{
		1, 2, 3, // row 0
		-1, 0, 5, // row 1 = padding
		2, -2, 4, // row 2
		0, 7, -3, // row 3
	}
	// Bags over [0,1,2, 1,3, 1,1]: {0,pad,2}, {pad,3}, {pad,pad}.
	input := tensor.New([]float64{0, 1, 2, 1, 3, 1, 1}, 7)
	offsets := tensor.New([]float64{0, 3, 5}, 3)
	cases := []struct {
		mode string
		want []float64
	}{
		{"sum", []float64{
			3, 0, 7, // r0+r2 (pad skipped)
			0, 7, -3, // r3
			0, 0, 0, // all-pad bag -> zeros
		}},
		{"mean", []float64{
			1.5, 0, 3.5, // (r0+r2)/2 — divides by the NON-PAD count
			0, 7, -3,
			0, 0, 0,
		}},
		{"max", []float64{
			2, 2, 4, // max(r0, r2), pad excluded from the max
			0, 7, -3,
			0, 0, 0,
		}},
	}
	for _, c := range cases {
		e := NewEmbeddingBag(4, 3, WithBagMode(c.mode), WithBagPaddingIdx(1))
		copy(e.Weight.Data, w)
		out := e.Forward(input, offsets)
		if out.Shape[0] != 3 || out.Shape[1] != 3 {
			t.Fatalf("%s: shape %v want [3 3]", c.mode, out.Shape)
		}
		for i, want := range c.want {
			if math.Abs(out.Data[i]-want) > 1e-12 {
				t.Fatalf("%s: data[%d] got %g want %g", c.mode, i, out.Data[i], want)
			}
		}
	}
}

func TestGradCheckEmbeddingBagPaddingIdx(t *testing.T) {
	// Pads interspersed plus an all-pad bag. Unlike Embedding, the output
	// does NOT depend on the padding row at all (entries are excluded from
	// the reduction), so the FULL-weight gradcheck is valid here: FD and the
	// analytic gradient are both zero at the padding row.
	input := tensor.New([]float64{0, 1, 2, 1, 4, 1, 1, 3}, 8)
	offsets := tensor.New([]float64{0, 3, 5, 7}, 4) // {0,pad,2},{pad,4},{pad,pad},{3}
	wdata := []float64{
		0.5, -1.0, 2.0,
		0.25, 0.5, 0.75, // padding row, nonzero on purpose
		-0.5, 1.5, -2.0,
		1.0, 0.0, 0.5,
		-1.5, 2.5, 1.25,
	}
	for _, mode := range []string{"sum", "mean", "max"} {
		e := NewEmbeddingBag(5, 3, WithBagMode(mode), WithBagPaddingIdx(1))
		copy(e.Weight.Data, wdata) // tie-free rows keep the max FD stable
		loss := func() *tensor.Tensor { return e.Forward(input, offsets).Square().Mean() }
		gradCheck(t, "EmbeddingBag-pad-"+mode, loss, e.Parameters(), gcEps, gcTol, 0)

		e.Weight.ZeroGrad()
		loss().Backward()
		for c := 0; c < 3; c++ {
			if g := e.Weight.Grad.Data[1*3+c]; g != 0 {
				t.Fatalf("%s: padding-row grad not exactly zero: [1][%d] = %g", mode, c, g)
			}
		}
	}
}

func TestEmbeddingBagMaxNorm(t *testing.T) {
	e := NewEmbeddingBag(3, 3, WithBagMode("sum"), WithBagMaxNorm(1.0))
	w := []float64{
		3, 0, 4, // row 0: L2 norm 5, referenced -> renormed to 1
		0.1, 0.2, 0.2, // row 1: L2 norm 0.3, referenced -> untouched
		8, 6, 0, // row 2: L2 norm 10, NOT referenced -> untouched
	}
	copy(e.Weight.Data, w)

	out := e.Forward(tensor.New([]float64{0, 1}, 2), tensor.New([]float64{0}, 1))

	if n := l2(e.Weight.Data[0:3]); math.Abs(n-1) > 1e-12 {
		t.Fatalf("row 0 norm: got %.17g want 1", n)
	}
	for j := 3; j < 9; j++ {
		if e.Weight.Data[j] != w[j] {
			t.Fatalf("weight[%d] modified: got %g want %g", j, e.Weight.Data[j], w[j])
		}
	}
	for c := 0; c < 3; c++ {
		want := e.Weight.Data[c] + e.Weight.Data[3+c] // renormed r0 + r1
		if math.Abs(out.Data[c]-want) > 1e-12 {
			t.Fatalf("out[%d]: got %g want %g", c, out.Data[c], want)
		}
	}
}
