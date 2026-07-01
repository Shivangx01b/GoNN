package optim

import (
	"math"

	"gonn/tensor"
)

// SparseAdam implements a dense-equivalent of torch.optim.SparseAdam.
//
// GoNN has no sparse tensor type, so this implementation emulates SparseAdam's
// defining behaviour: it only updates the moment estimates and parameter
// entries whose gradient is nonzero on the current step, and it maintains a
// per-entry step count for bias correction. Entries whose gradient is zero on a
// given step are left untouched (no state decay), exactly as a true SparseAdam
// would skip absent rows. This is the dense projection of SparseAdam onto the
// nonzero-gradient entries.
//
// Note: SparseAdam does not support weight decay (neither does PyTorch's), so
// none is provided here.
type SparseAdam struct {
	baseOptimizer
	beta1 float64
	beta2 float64
	eps   float64
}

// SparseAdamOption configures a SparseAdam optimizer.
type SparseAdamOption func(*SparseAdam)

// WithSparseAdamBeta1 sets the beta1 coefficient.
func WithSparseAdamBeta1(b float64) SparseAdamOption { return func(s *SparseAdam) { s.beta1 = b } }

// WithSparseAdamBeta2 sets the beta2 coefficient.
func WithSparseAdamBeta2(b float64) SparseAdamOption { return func(s *SparseAdam) { s.beta2 = b } }

// WithSparseAdamEps sets the epsilon term.
func WithSparseAdamEps(e float64) SparseAdamOption { return func(s *SparseAdam) { s.eps = e } }

// NewSparseAdam constructs a SparseAdam optimizer with defaults beta1=0.9,
// beta2=0.999, eps=1e-8. Group.WeightDecay is ignored by SparseAdam.
func NewSparseAdam(params []*tensor.Tensor, lr float64, opts ...SparseAdamOption) *SparseAdam {
	return NewSparseAdamGroups(singleGroup(params, lr), opts...)
}

// NewSparseAdamGroups constructs a SparseAdam optimizer over explicit
// parameter groups. Group.WeightDecay is ignored by SparseAdam.
func NewSparseAdamGroups(groups []Group, opts ...SparseAdamOption) *SparseAdam {
	s := &SparseAdam{
		baseOptimizer: newBase(groups),
		beta1:         0.9,
		beta2:         0.999,
		eps:           1e-8,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step performs a single SparseAdam update, touching only nonzero-grad entries.
func (s *SparseAdam) Step() {
	s.forEachParam(func(grp *Group, p *tensor.Tensor, data, grad []float64, st *State) {
		m := st.Buf("m", len(data))
		v := st.Buf("v", len(data))
		cnt := st.Buf("step", len(data))
		for i := range data {
			g := grad[i]
			if g == 0 {
				continue
			}
			cnt[i]++
			t := cnt[i]
			bc1 := 1 - math.Pow(s.beta1, t)
			bc2 := 1 - math.Pow(s.beta2, t)
			m[i] = s.beta1*m[i] + (1-s.beta1)*g
			v[i] = s.beta2*v[i] + (1-s.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			data[i] -= grp.LR * mHat / (math.Sqrt(vHat) + s.eps)
		}
	})
}
