package optim

// State holds an optimizer's per-parameter state as named buffers and named
// scalars, created lazily on first access. Buffer names are part of each
// optimizer's stable state layout (frozen for a future StateDict):
//
//	Adam/AdamW/NAdam/RAdam/LAMB/SparseAdam:  "m", "v"
//	Adamax:                                  "m", "u"
//	SGD (momentum):                          "momentum_buf"
//	RMSprop:                                 "square_avg", "momentum_buf"
//	Adagrad:                                 "sum"
//	Adadelta:                                "square_avg", "acc_delta"
//	Rprop:                                   "prev_grad", "step_size"
//	Lion:                                    "m"
//	Adafactor:                               "v"           + scalar "step"
//	ASGD:                                    "ax"          + scalars "eta", "mu", "step"
//	SparseAdam (extra):                      "step" (per-element counts)
//
// Named buffers were chosen over generic typed state structs: the base
// optimizer stays non-generic (no type parameter infecting the Optimizer
// interface), and the map layout serializes directly. The map lookups happen
// once per parameter per Step, not per element — the cost is noise.
type State struct {
	bufs    map[string][]float64
	scalars map[string]float64
}

// Buf returns the named buffer of length n, creating it zero-filled on first
// access.
func (s *State) Buf(name string, n int) []float64 {
	if s.bufs == nil {
		s.bufs = make(map[string][]float64)
	}
	b := s.bufs[name]
	if b == nil {
		b = make([]float64, n)
		s.bufs[name] = b
	}
	return b
}

// BufFill returns the named buffer of length n, creating it filled with the
// given value on first access (e.g. Rprop's step sizes start at the LR).
func (s *State) BufFill(name string, n int, fill float64) []float64 {
	if s.bufs == nil {
		s.bufs = make(map[string][]float64)
	}
	b := s.bufs[name]
	if b == nil {
		b = make([]float64, n)
		for i := range b {
			b[i] = fill
		}
		s.bufs[name] = b
	}
	return b
}

// Has reports whether the named buffer already exists (useful for
// first-touch detection before Buf creates it).
func (s *State) Has(name string) bool {
	_, ok := s.bufs[name]
	return ok
}

// Scalar returns the named scalar (0 if never set).
func (s *State) Scalar(name string) float64 { return s.scalars[name] }

// SetScalar stores a named scalar.
func (s *State) SetScalar(name string, v float64) {
	if s.scalars == nil {
		s.scalars = make(map[string]float64)
	}
	s.scalars[name] = v
}
