package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

func twoGroupOpt(lr1, lr2 float64) *SGD {
	return NewSGDGroups([]Group{
		{Params: []*tensor.Tensor{scalarParam(0)}, LR: lr1},
		{Params: []*tensor.Tensor{scalarParam(0)}, LR: lr2},
	})
}

func groupLRs(o Optimizer) []float64 {
	gs := o.Groups()
	out := make([]float64, len(gs))
	for i, g := range gs {
		out[i] = g.LR
	}
	return out
}

func TestStepLRPreservesGroupRatio(t *testing.T) {
	opt := twoGroupOpt(1.0, 0.5)
	s := NewStepLR(opt, 5, 0.5)
	for i := 0; i < 5; i++ {
		s.Step()
	}
	lrs := groupLRs(opt)
	if math.Abs(lrs[0]-0.5) > 1e-12 || math.Abs(lrs[1]-0.25) > 1e-12 {
		t.Fatalf("group LRs after decay = %v, want [0.5 0.25]", lrs)
	}
}

func TestCosineAnnealingPerGroupBase(t *testing.T) {
	opt := twoGroupOpt(1.0, 0.5)
	s := NewCosineAnnealingLR(opt, 10, 0)
	for i := 0; i < 5; i++ { // midpoint: cos(pi/2)=0 -> lr = base/2
		s.Step()
	}
	lrs := groupLRs(opt)
	if math.Abs(lrs[0]-0.5) > 1e-12 || math.Abs(lrs[1]-0.25) > 1e-12 {
		t.Fatalf("cosine midpoint LRs = %v, want [0.5 0.25]", lrs)
	}
}

func TestLinearLRPerGroupBase(t *testing.T) {
	opt := twoGroupOpt(1.0, 0.4)
	s := NewLinearLR(opt, 0.5, 1.0, 10)
	lrs := groupLRs(opt) // start factor applied at construction
	if math.Abs(lrs[0]-0.5) > 1e-12 || math.Abs(lrs[1]-0.2) > 1e-12 {
		t.Fatalf("start LRs = %v, want [0.5 0.2]", lrs)
	}
	for i := 0; i < 10; i++ {
		s.Step()
	}
	lrs = groupLRs(opt)
	if math.Abs(lrs[0]-1.0) > 1e-12 || math.Abs(lrs[1]-0.4) > 1e-12 {
		t.Fatalf("end LRs = %v, want [1 0.4]", lrs)
	}
}

func TestReduceLROnPlateauGroups(t *testing.T) {
	opt := twoGroupOpt(1.0, 0.5)
	s := NewReduceLROnPlateau(opt, 0.1, 0.0, 1)
	s.Step(1.0) // best
	s.Step(1.0) // bad 1
	s.Step(1.0) // bad 2 > patience -> decay
	lrs := groupLRs(opt)
	if math.Abs(lrs[0]-0.1) > 1e-12 || math.Abs(lrs[1]-0.05) > 1e-12 {
		t.Fatalf("plateau LRs = %v, want [0.1 0.05]", lrs)
	}
}

func TestOneCycleLRPeaksAtSharedMax(t *testing.T) {
	opt := twoGroupOpt(0.1, 0.05)
	s := NewOneCycleLR(opt, 1.0, 10)
	for i := 0; i < 5; i++ { // midpoint: every group at maxLR
		s.Step()
	}
	lrs := groupLRs(opt)
	if math.Abs(lrs[0]-1.0) > 1e-12 || math.Abs(lrs[1]-1.0) > 1e-12 {
		t.Fatalf("one-cycle midpoint LRs = %v, want [1 1]", lrs)
	}
	for i := 0; i < 5; i++ {
		s.Step()
	}
	lrs = groupLRs(opt)
	if math.Abs(lrs[0]-0.1) > 1e-12 || math.Abs(lrs[1]-0.05) > 1e-12 {
		t.Fatalf("one-cycle end LRs = %v, want [0.1 0.05]", lrs)
	}
}
