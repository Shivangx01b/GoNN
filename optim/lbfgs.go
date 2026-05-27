package optim

import (
	"math"

	"gonn/tensor"
)

// LBFGS implements the limited-memory BFGS quasi-Newton optimizer.
//
// Unlike first-order optimizers, LBFGS needs to evaluate the loss several
// times per Step (for the line search). The caller therefore passes a
// closure that re-runs the forward pass, clears any old grads, calls
// Backward, and returns the loss value.
type LBFGS struct {
	params      []*tensor.Tensor
	lr          float64
	HistorySize int
	MaxIter     int

	// State carried across Step calls.
	prevFlatGrad []float64
	prevFlatParam []float64
	sHistory      [][]float64
	yHistory      [][]float64
	rhoHistory    []float64
	hasPrev       bool
}

// LBFGSOption configures an LBFGS optimizer.
type LBFGSOption func(*LBFGS)

// WithLBFGSHistorySize sets the number of (s, y) pairs to retain.
func WithLBFGSHistorySize(n int) LBFGSOption { return func(l *LBFGS) { l.HistorySize = n } }

// WithLBFGSMaxIter sets the maximum inner-iteration budget per Step.
func WithLBFGSMaxIter(n int) LBFGSOption { return func(l *LBFGS) { l.MaxIter = n } }

// NewLBFGS constructs an LBFGS optimizer with defaults
// HistorySize=10, MaxIter=20, lr=1.
func NewLBFGS(params []*tensor.Tensor, lr float64, opts ...LBFGSOption) *LBFGS {
	if lr == 0 {
		lr = 1
	}
	l := &LBFGS{
		params:      params,
		lr:          lr,
		HistorySize: 10,
		MaxIter:     20,
	}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

// totalNumel returns the number of scalar parameters across all tensors.
func (l *LBFGS) totalNumel() int {
	n := 0
	for _, p := range l.params {
		if p == nil {
			continue
		}
		n += len(p.Data)
	}
	return n
}

// gatherFlatGrad returns a fresh flat vector of gradients for all params.
// Missing grads are treated as zero.
func (l *LBFGS) gatherFlatGrad() []float64 {
	out := make([]float64, 0, l.totalNumel())
	for _, p := range l.params {
		if p == nil {
			continue
		}
		if p.Grad == nil {
			out = append(out, make([]float64, len(p.Data))...)
			continue
		}
		out = append(out, p.Grad.Data...)
	}
	return out
}

// gatherFlatParam returns a fresh flat copy of param data.
func (l *LBFGS) gatherFlatParam() []float64 {
	out := make([]float64, 0, l.totalNumel())
	for _, p := range l.params {
		if p == nil {
			continue
		}
		out = append(out, p.Data...)
	}
	return out
}

// scatterParam writes flat back into the underlying parameter tensors.
func (l *LBFGS) scatterParam(flat []float64) {
	off := 0
	for _, p := range l.params {
		if p == nil {
			continue
		}
		copy(p.Data, flat[off:off+len(p.Data)])
		off += len(p.Data)
	}
}

// addToParams updates params <- params + alpha*direction (direction is flat).
func (l *LBFGS) addToParams(alpha float64, direction []float64) {
	off := 0
	for _, p := range l.params {
		if p == nil {
			continue
		}
		for i := range p.Data {
			p.Data[i] += alpha * direction[off+i]
		}
		off += len(p.Data)
	}
}

// twoLoop runs the L-BFGS two-loop recursion on the current grad to produce a
// search direction (returned as the negative descent direction r, so the step
// is x <- x - alpha * r equivalently x <- x + alpha * (-r)).
func (l *LBFGS) twoLoop(grad []float64) []float64 {
	q := make([]float64, len(grad))
	copy(q, grad)
	k := len(l.sHistory)
	alpha := make([]float64, k)
	for i := k - 1; i >= 0; i-- {
		s := l.sHistory[i]
		y := l.yHistory[i]
		rho := l.rhoHistory[i]
		ai := rho * dot(s, q)
		alpha[i] = ai
		for j := range q {
			q[j] -= ai * y[j]
		}
	}
	// Initial Hessian approximation H_0 = gamma * I.
	gamma := 1.0
	if k > 0 {
		s := l.sHistory[k-1]
		y := l.yHistory[k-1]
		ys := dot(y, s)
		yy := dot(y, y)
		if yy > 0 {
			gamma = ys / yy
		}
	}
	r := make([]float64, len(q))
	for j := range r {
		r[j] = gamma * q[j]
	}
	for i := 0; i < k; i++ {
		s := l.sHistory[i]
		y := l.yHistory[i]
		rho := l.rhoHistory[i]
		bi := rho * dot(y, r)
		for j := range r {
			r[j] += s[j] * (alpha[i] - bi)
		}
	}
	return r
}

// Step performs one LBFGS update by repeatedly invoking closure to evaluate
// the loss and gradients. Returns the final loss value seen.
func (l *LBFGS) Step(closure func() float64) float64 {
	// Initial evaluation establishes loss and gradient at the current point.
	loss := closure()
	flatGrad := l.gatherFlatGrad()

	nIter := l.MaxIter
	if nIter <= 0 {
		nIter = 1
	}

	for iter := 0; iter < nIter; iter++ {
		// Direction: descent = - H * grad. twoLoop returns H*grad ~ r.
		r := l.twoLoop(flatGrad)
		// Direction d = -r (we'll move params by + alpha * d).
		d := make([]float64, len(r))
		for j := range d {
			d[j] = -r[j]
		}

		// Directional derivative for Armijo-style check.
		gd := dot(flatGrad, d)
		if gd >= 0 {
			// Not a descent direction (can happen early or with stale history).
			// Fall back to steepest descent.
			for j := range d {
				d[j] = -flatGrad[j]
			}
			gd = dot(flatGrad, d)
			if gd >= 0 {
				return loss
			}
		}

		// Save x_k before stepping so we can compute s = x_{k+1} - x_k.
		xPrev := l.gatherFlatParam()
		gPrev := make([]float64, len(flatGrad))
		copy(gPrev, flatGrad)

		// Backtracking line search.
		alpha := l.lr
		const c1 = 1e-4
		const shrink = 0.5
		const maxLS = 20

		var newLoss float64
		var newGrad []float64
		accepted := false
		for ls := 0; ls < maxLS; ls++ {
			// Apply trial step: x = xPrev + alpha * d
			l.scatterParam(xPrev)
			l.addToParams(alpha, d)
			newLoss = closure()
			if !math.IsNaN(newLoss) && !math.IsInf(newLoss, 0) &&
				newLoss <= loss+c1*alpha*gd {
				newGrad = l.gatherFlatGrad()
				accepted = true
				break
			}
			alpha *= shrink
			if alpha < 1e-20 {
				break
			}
		}
		if !accepted {
			// Restore the previous point and bail out.
			l.scatterParam(xPrev)
			return loss
		}

		// Build s and y.
		s := make([]float64, len(xPrev))
		y := make([]float64, len(gPrev))
		xNow := l.gatherFlatParam()
		for j := range s {
			s[j] = xNow[j] - xPrev[j]
			y[j] = newGrad[j] - gPrev[j]
		}
		ys := dot(y, s)
		// Only store a curvature pair if ys is positive (BFGS condition).
		if ys > 1e-10 {
			rho := 1.0 / ys
			l.sHistory = append(l.sHistory, s)
			l.yHistory = append(l.yHistory, y)
			l.rhoHistory = append(l.rhoHistory, rho)
			if len(l.sHistory) > l.HistorySize {
				l.sHistory = l.sHistory[1:]
				l.yHistory = l.yHistory[1:]
				l.rhoHistory = l.rhoHistory[1:]
			}
		}

		loss = newLoss
		flatGrad = newGrad

		// Convergence: tiny gradient norm.
		if dot(flatGrad, flatGrad) < 1e-20 {
			break
		}
	}

	return loss
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// ZeroGrad zeros the gradients of all parameters.
func (l *LBFGS) ZeroGrad() { zeroGradAll(l.params) }

// Parameters returns the parameter list.
func (l *LBFGS) Parameters() []*tensor.Tensor { return l.params }

// LR returns the (initial line-search) step size.
func (l *LBFGS) LR() float64 { return l.lr }

// SetLR updates the initial line-search step size.
func (l *LBFGS) SetLR(lr float64) { l.lr = lr }
