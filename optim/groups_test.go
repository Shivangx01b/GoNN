package optim

import (
	"math"
	"testing"

	"gonn/tensor"
)

func scalarParam(v float64) *tensor.Tensor {
	p := tensor.New([]float64{v}, 1).SetRequiresGrad(true)
	p.Grad = tensor.New([]float64{0}, 1)
	return p
}

// quadGrad sets p.Grad for loss 0.5*(p-target)^2.
func quadGrad(p *tensor.Tensor, target float64) {
	p.Grad.Data[0] = p.Data[0] - target
}

func TestGroupsDifferentLRs(t *testing.T) {
	fast := scalarParam(0)
	slow := scalarParam(0)
	opt := NewSGDGroups([]Group{
		{Params: []*tensor.Tensor{fast}, LR: 0.5},
		{Params: []*tensor.Tensor{slow}, LR: 0.05},
	})
	quadGrad(fast, 1)
	quadGrad(slow, 1)
	opt.Step()
	// One SGD step: p -= lr * (p - 1) => p = lr.
	if math.Abs(fast.Data[0]-0.5) > 1e-12 {
		t.Fatalf("fast group after step = %v, want 0.5", fast.Data[0])
	}
	if math.Abs(slow.Data[0]-0.05) > 1e-12 {
		t.Fatalf("slow group after step = %v, want 0.05", slow.Data[0])
	}
}

func TestGroupZeroLRFreezes(t *testing.T) {
	frozen := scalarParam(3)
	live := scalarParam(3)
	opt := NewAdamGroups([]Group{
		{Params: []*tensor.Tensor{frozen}, LR: 0},
		{Params: []*tensor.Tensor{live}, LR: 0.1},
	})
	for i := 0; i < 50; i++ {
		quadGrad(frozen, 0)
		quadGrad(live, 0)
		opt.Step()
	}
	if frozen.Data[0] != 3 {
		t.Fatalf("LR=0 group moved: %v", frozen.Data[0])
	}
	if math.Abs(live.Data[0]-3) < 0.5 {
		t.Fatalf("live group did not move: %v", live.Data[0])
	}
}

func TestGroupWeightDecayPerGroup(t *testing.T) {
	decayed := scalarParam(1)
	plain := scalarParam(1)
	opt := NewSGDGroups([]Group{
		{Params: []*tensor.Tensor{decayed}, LR: 0.1, WeightDecay: 1.0},
		{Params: []*tensor.Tensor{plain}, LR: 0.1},
	})
	// Zero gradient: only weight decay moves the decayed param.
	opt.Step()
	if math.Abs(decayed.Data[0]-0.9) > 1e-12 { // p -= lr * wd * p
		t.Fatalf("decayed = %v, want 0.9", decayed.Data[0])
	}
	if plain.Data[0] != 1 {
		t.Fatalf("plain moved: %v", plain.Data[0])
	}
}

func TestSetLRSetsAllGroups(t *testing.T) {
	a, b := scalarParam(0), scalarParam(0)
	opt := NewSGDGroups([]Group{
		{Params: []*tensor.Tensor{a}, LR: 1},
		{Params: []*tensor.Tensor{b}, LR: 2},
	})
	opt.SetLR(0.25)
	for i, g := range opt.Groups() {
		if g.LR != 0.25 {
			t.Fatalf("group %d LR = %v, want 0.25", i, g.LR)
		}
	}
	if opt.LR() != 0.25 {
		t.Fatalf("LR() = %v, want 0.25", opt.LR())
	}
}

func TestGroupsLiveMutation(t *testing.T) {
	p := scalarParam(0)
	opt := NewSGD([]*tensor.Tensor{p}, 0.1)
	opt.Groups()[0].LR = 0.5
	quadGrad(p, 1)
	opt.Step()
	if math.Abs(p.Data[0]-0.5) > 1e-12 {
		t.Fatalf("mutated group LR not used: p = %v, want 0.5", p.Data[0])
	}
}

func TestParametersGroupOrder(t *testing.T) {
	a, b, c := scalarParam(0), scalarParam(0), scalarParam(0)
	opt := NewAdamGroups([]Group{
		{Params: []*tensor.Tensor{a, b}, LR: 0.1},
		{Params: []*tensor.Tensor{c}, LR: 0.2},
	})
	ps := opt.Parameters()
	if len(ps) != 3 || ps[0] != a || ps[1] != b || ps[2] != c {
		t.Fatal("Parameters() not in group order")
	}
}
