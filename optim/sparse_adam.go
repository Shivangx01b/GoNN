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
	params []*tensor.Tensor
	lr     float64
	beta1  float64
	beta2  float64
	eps    float64

	m    map[*tensor.Tensor][]float64
	v    map[*tensor.Tensor][]float64
	step map[*tensor.Tensor][]int // per-entry update count
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
// beta2=0.999, eps=1e-8.
func NewSparseAdam(params []*tensor.Tensor, lr float64, opts ...SparseAdamOption) *SparseAdam {
	s := &SparseAdam{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-8,
		m:      make(map[*tensor.Tensor][]float64),
		v:      make(map[*tensor.Tensor][]float64),
		step:   make(map[*tensor.Tensor][]int),
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Step performs a single SparseAdam update, touching only nonzero-grad entries.
func (s *SparseAdam) Step() {
	for _, p := range s.params {
		if p == nil || p.Grad == nil {
			continue
		}
		grad := p.Grad.Data
		data := p.Data
		m := s.m[p]
		if m == nil {
			m = make([]float64, len(data))
			s.m[p] = m
		}
		v := s.v[p]
		if v == nil {
			v = make([]float64, len(data))
			s.v[p] = v
		}
		cnt := s.step[p]
		if cnt == nil {
			cnt = make([]int, len(data))
			s.step[p] = cnt
		}
		for i := range data {
			g := grad[i]
			if g == 0 {
				continue
			}
			cnt[i]++
			t := float64(cnt[i])
			bc1 := 1 - math.Pow(s.beta1, t)
			bc2 := 1 - math.Pow(s.beta2, t)
			m[i] = s.beta1*m[i] + (1-s.beta1)*g
			v[i] = s.beta2*v[i] + (1-s.beta2)*g*g
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			data[i] -= s.lr * mHat / (math.Sqrt(vHat) + s.eps)
		}
	}
}

// ZeroGrad zeros the gradients of all parameters.
func (s *SparseAdam) ZeroGrad() { zeroGradAll(s.params) }

// Parameters returns the parameter list.
func (s *SparseAdam) Parameters() []*tensor.Tensor { return s.params }

// LR returns the current learning rate.
func (s *SparseAdam) LR() float64 { return s.lr }

// SetLR updates the learning rate.
func (s *SparseAdam) SetLR(lr float64) { s.lr = lr }
