package optim

import "gonn/tensor"

// Group is a parameter group: a set of tensors sharing a learning rate and
// weight decay — the analog of a PyTorch param_group. Build groups from
// nn.NamedParameters()/nn.FilterParams (e.g. to exempt biases and norm
// weights from decay, or to give a head a higher LR than the backbone):
//
//	opt := optim.NewAdamWGroups([]optim.Group{
//	    {Params: backbone, LR: 1e-4, WeightDecay: 0.01},
//	    {Params: head, LR: 1e-3},
//	})
//
// Fields are literal values — there is no "inherit" sentinel. LR: 0
// legitimately freezes a group, and WeightDecay: 0 means no decay. Note that
// the NewXXXGroups constructors do NOT inject an optimizer's default weight
// decay into groups (only the single-list NewXXX constructors do, where
// applicable, e.g. AdamW's 0.01).
type Group struct {
	Params      []*tensor.Tensor
	LR          float64
	WeightDecay float64
}
