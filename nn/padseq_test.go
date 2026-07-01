package nn

import (
	"testing"

	"gonn/tensor"
)

func TestPadSequenceBatchFirst(t *testing.T) {
	s0 := seededRandn(601, 2, 3)
	s1 := seededRandn(602, 4, 3)
	s2 := seededRandn(603, 3, 3)

	padded := PadSequence([]*tensor.Tensor{s0, s1, s2}, true, -7)
	if padded.Shape[0] != 3 || padded.Shape[1] != 4 || padded.Shape[2] != 3 {
		t.Fatalf("PadSequence shape: got %v want [3 4 3]", padded.Shape)
	}

	// Original values in place, padValue in the tail.
	seqs := []*tensor.Tensor{s0, s1, s2}
	lengths := []int{2, 4, 3}
	for b, s := range seqs {
		for tt := 0; tt < 4; tt++ {
			for f := 0; f < 3; f++ {
				got := padded.Data[(b*4+tt)*3+f]
				if tt < lengths[b] {
					if got != s.Data[tt*3+f] {
						t.Fatalf("padded[%d,%d,%d]: got %g want %g", b, tt, f, got, s.Data[tt*3+f])
					}
				} else if got != -7 {
					t.Fatalf("padded[%d,%d,%d]: got %g want pad -7", b, tt, f, got)
				}
			}
		}
	}

	// Round-trip.
	back := UnpadSequence(padded, lengths)
	for i, s := range seqs {
		tensorsEqualExact(t, "unpad round-trip", s, back[i])
	}
}

func TestPadSequenceTimeFirst(t *testing.T) {
	s0 := seededRandn(611, 2, 3)
	s1 := seededRandn(612, 4, 3)

	padded := PadSequence([]*tensor.Tensor{s0, s1}, false, 0)
	if padded.Shape[0] != 4 || padded.Shape[1] != 2 || padded.Shape[2] != 3 {
		t.Fatalf("time-first shape: got %v want [4 2 3]", padded.Shape)
	}
	// padded[t, b, f] must match the batch-first layout transposed.
	bf := PadSequence([]*tensor.Tensor{s0, s1}, true, 0)
	for tt := 0; tt < 4; tt++ {
		for b := 0; b < 2; b++ {
			for f := 0; f < 3; f++ {
				if padded.Data[(tt*2+b)*3+f] != bf.Data[(b*4+tt)*3+f] {
					t.Fatalf("time-first/batch-first mismatch at t=%d b=%d f=%d", tt, b, f)
				}
			}
		}
	}
}

func TestPadSequenceValidation(t *testing.T) {
	mustPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic", name)
			}
		}()
		f()
	}
	mustPanic("empty list", func() { PadSequence(nil, true, 0) })
	mustPanic("feature mismatch", func() {
		PadSequence([]*tensor.Tensor{tensor.Zeros(2, 3), tensor.Zeros(2, 4)}, true, 0)
	})
	mustPanic("bad rank", func() {
		PadSequence([]*tensor.Tensor{tensor.Zeros(2, 3, 1)}, true, 0)
	})
	mustPanic("bad lengths count", func() {
		UnpadSequence(tensor.Zeros(2, 3, 4), []int{1})
	})
	mustPanic("length out of range", func() {
		UnpadSequence(tensor.Zeros(2, 3, 4), []int{1, 5})
	})
}

func TestGradCheckPadSequence(t *testing.T) {
	s0 := seededRandn(621, 2, 3).SetRequiresGrad(true)
	s1 := seededRandn(622, 4, 3).SetRequiresGrad(true)
	s2 := seededRandn(623, 1, 3).SetRequiresGrad(true)

	loss := func() *tensor.Tensor {
		return PadSequence([]*tensor.Tensor{s0, s1, s2}, true, 0.5).Square().Mean()
	}
	gradCheck(t, "PadSequence", loss, []*tensor.Tensor{s0, s1, s2}, gcEps, gcTol, 0)

	// And through UnpadSequence back to the padded tensor.
	padded := seededRandn(624, 2, 3, 4).SetRequiresGrad(true)
	loss2 := func() *tensor.Tensor {
		parts := UnpadSequence(padded, []int{2, 3})
		return parts[0].Square().Mean().Add(parts[1].Square().Mean())
	}
	gradCheck(t, "UnpadSequence", loss2, []*tensor.Tensor{padded}, gcEps, gcTol, 0)
}
