package ml

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// LDA is Linear Discriminant Analysis.
// It estimates class-conditional Gaussians with a shared covariance matrix.
// Used as a classifier and (optionally) for supervised dimensionality reduction.
type LDA struct {
	classes  []int
	priors   []float64
	means    [][]float64
	covInv   *mat.Dense
	logDet   float64
	dim      int
	// Dimensionality-reduction components (Fisher discriminants).
	NComponents int
	Components  [][]float64 // [NComponents][d]
}

func (m *LDA) Fit(X [][]float64, y []int) {
	n, d := shapeOf(X)
	m.dim = d
	m.classes = uniqueInts(y)
	K := len(m.classes)
	m.priors = make([]float64, K)
	m.means = make([][]float64, K)
	counts := make([]int, K)
	for k := range m.means {
		m.means[k] = make([]float64, d)
	}
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}
	for i := 0; i < n; i++ {
		k := classIdx[y[i]]
		counts[k]++
		for j := 0; j < d; j++ {
			m.means[k][j] += X[i][j]
		}
	}
	for k := 0; k < K; k++ {
		m.priors[k] = float64(counts[k]) / float64(n)
		for j := 0; j < d; j++ {
			m.means[k][j] /= float64(counts[k])
		}
	}
	// Shared (within-class) covariance.
	cov := mat.NewDense(d, d, nil)
	for i := 0; i < n; i++ {
		k := classIdx[y[i]]
		diff := make([]float64, d)
		for j := 0; j < d; j++ {
			diff[j] = X[i][j] - m.means[k][j]
		}
		for a := 0; a < d; a++ {
			for b := 0; b < d; b++ {
				cov.Set(a, b, cov.At(a, b)+diff[a]*diff[b])
			}
		}
	}
	denom := float64(n - K)
	if denom < 1 {
		denom = 1
	}
	cov.Apply(func(_, _ int, v float64) float64 { return v / denom }, cov)
	// Regularize for numerical stability.
	for i := 0; i < d; i++ {
		cov.Set(i, i, cov.At(i, i)+1e-6)
	}
	var inv mat.Dense
	if err := inv.Inverse(cov); err != nil {
		// Fallback: pseudo-inverse via SVD.
		var svd mat.SVD
		if !svd.Factorize(cov, mat.SVDFull) {
			panic("LDA: covariance inversion failed")
		}
		var u, v mat.Dense
		svd.UTo(&u)
		svd.VTo(&v)
		s := svd.Values(nil)
		sInv := mat.NewDense(d, d, nil)
		for i := 0; i < d; i++ {
			if s[i] > 1e-12 {
				sInv.Set(i, i, 1.0/s[i])
			}
		}
		var tmp mat.Dense
		tmp.Mul(&v, sInv)
		inv.Mul(&tmp, u.T())
	}
	m.covInv = &inv
	// log|Sigma| via LU determinant on cov.
	var lu mat.LU
	lu.Factorize(cov)
	det := lu.Det()
	if det <= 0 {
		det = 1e-300
	}
	m.logDet = math.Log(det)

	// Fisher discriminants for dimensionality reduction.
	if m.NComponents > 0 {
		m.computeFisher(X, y, classIdx, cov)
	}
}

// computeFisher solves the generalized eigenvalue problem S_w^-1 S_b for the
// top components.
func (m *LDA) computeFisher(X [][]float64, y []int, classIdx map[int]int, sw *mat.Dense) {
	_, d := shapeOf(X)
	K := len(m.classes)
	overall := meanCol(X)
	counts := make([]int, K)
	for i := range y {
		counts[classIdx[y[i]]]++
	}
	sb := mat.NewDense(d, d, nil)
	for k := 0; k < K; k++ {
		diff := make([]float64, d)
		for j := 0; j < d; j++ {
			diff[j] = m.means[k][j] - overall[j]
		}
		n_k := float64(counts[k])
		for a := 0; a < d; a++ {
			for b := 0; b < d; b++ {
				sb.Set(a, b, sb.At(a, b)+n_k*diff[a]*diff[b])
			}
		}
	}
	var swInv mat.Dense
	if err := swInv.Inverse(sw); err != nil {
		return
	}
	var M mat.Dense
	M.Mul(&swInv, sb)
	var eig mat.Eigen
	if ok := eig.Factorize(&M, mat.EigenRight); !ok {
		return
	}
	vals := eig.Values(nil)
	cV := mat.NewCDense(d, d, nil)
	eig.VectorsTo(cV)
	// Sort indices by descending |eigenvalue|.
	idx := make([]int, len(vals))
	for i := range idx {
		idx[i] = i
	}
	// simple selection sort, small d.
	for i := 0; i < len(idx); i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if real(vals[idx[j]]) > real(vals[idx[best]]) {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	nc := m.NComponents
	if nc > d {
		nc = d
	}
	m.Components = make([][]float64, nc)
	for i := 0; i < nc; i++ {
		col := idx[i]
		row := make([]float64, d)
		for j := 0; j < d; j++ {
			row[j] = real(cV.At(j, col))
		}
		m.Components[i] = row
	}
}

// Transform projects X onto the top Fisher discriminants.
func (m *LDA) Transform(X [][]float64) [][]float64 {
	if m.NComponents == 0 || m.Components == nil {
		return X
	}
	out := make([][]float64, len(X))
	for i, x := range X {
		row := make([]float64, len(m.Components))
		for c, comp := range m.Components {
			row[c] = dot(x, comp)
		}
		out[i] = row
	}
	return out
}

// discriminant returns the Gaussian log-likelihood + log-prior for each class.
func (m *LDA) discriminant(x []float64) []float64 {
	K := len(m.classes)
	out := make([]float64, K)
	d := m.dim
	for k := 0; k < K; k++ {
		diff := make([]float64, d)
		for j := 0; j < d; j++ {
			diff[j] = x[j] - m.means[k][j]
		}
		// quad = diff^T * covInv * diff
		quad := 0.0
		for a := 0; a < d; a++ {
			s := 0.0
			for b := 0; b < d; b++ {
				s += m.covInv.At(a, b) * diff[b]
			}
			quad += diff[a] * s
		}
		out[k] = -0.5*quad - 0.5*m.logDet + math.Log(m.priors[k]+1e-300)
	}
	return out
}

func (m *LDA) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		d := m.discriminant(x)
		out[i] = m.classes[argmaxF(d)]
	}
	return out
}

// PredictProba returns posterior class probabilities.
func (m *LDA) PredictProba(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, x := range X {
		out[i] = softmax(m.discriminant(x))
	}
	return out
}

// QDA is Quadratic Discriminant Analysis: per-class covariance matrices.
type QDA struct {
	classes []int
	priors  []float64
	means   [][]float64
	covInv  []*mat.Dense
	logDet  []float64
	dim     int
}

func (m *QDA) Fit(X [][]float64, y []int) {
	n, d := shapeOf(X)
	m.dim = d
	m.classes = uniqueInts(y)
	K := len(m.classes)
	m.priors = make([]float64, K)
	m.means = make([][]float64, K)
	m.covInv = make([]*mat.Dense, K)
	m.logDet = make([]float64, K)
	counts := make([]int, K)
	for k := range m.means {
		m.means[k] = make([]float64, d)
	}
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}
	for i := 0; i < n; i++ {
		k := classIdx[y[i]]
		counts[k]++
		for j := 0; j < d; j++ {
			m.means[k][j] += X[i][j]
		}
	}
	for k := 0; k < K; k++ {
		m.priors[k] = float64(counts[k]) / float64(n)
		for j := 0; j < d; j++ {
			m.means[k][j] /= float64(counts[k])
		}
	}
	// Per-class covariance.
	covs := make([]*mat.Dense, K)
	for k := 0; k < K; k++ {
		covs[k] = mat.NewDense(d, d, nil)
	}
	for i := 0; i < n; i++ {
		k := classIdx[y[i]]
		diff := make([]float64, d)
		for j := 0; j < d; j++ {
			diff[j] = X[i][j] - m.means[k][j]
		}
		for a := 0; a < d; a++ {
			for b := 0; b < d; b++ {
				covs[k].Set(a, b, covs[k].At(a, b)+diff[a]*diff[b])
			}
		}
	}
	for k := 0; k < K; k++ {
		denom := float64(counts[k] - 1)
		if denom < 1 {
			denom = 1
		}
		covs[k].Apply(func(_, _ int, v float64) float64 { return v / denom }, covs[k])
		for i := 0; i < d; i++ {
			covs[k].Set(i, i, covs[k].At(i, i)+1e-6)
		}
		var inv mat.Dense
		if err := inv.Inverse(covs[k]); err != nil {
			// Pseudo-inverse fallback.
			var svd mat.SVD
			if !svd.Factorize(covs[k], mat.SVDFull) {
				panic("QDA: covariance inversion failed")
			}
			var u, v mat.Dense
			svd.UTo(&u)
			svd.VTo(&v)
			s := svd.Values(nil)
			sInv := mat.NewDense(d, d, nil)
			for i := 0; i < d; i++ {
				if s[i] > 1e-12 {
					sInv.Set(i, i, 1.0/s[i])
				}
			}
			var tmp mat.Dense
			tmp.Mul(&v, sInv)
			inv.Mul(&tmp, u.T())
		}
		m.covInv[k] = &inv
		var lu mat.LU
		lu.Factorize(covs[k])
		det := lu.Det()
		if det <= 0 {
			det = 1e-300
		}
		m.logDet[k] = math.Log(det)
	}
}

func (m *QDA) discriminant(x []float64) []float64 {
	K := len(m.classes)
	out := make([]float64, K)
	d := m.dim
	for k := 0; k < K; k++ {
		diff := make([]float64, d)
		for j := 0; j < d; j++ {
			diff[j] = x[j] - m.means[k][j]
		}
		quad := 0.0
		for a := 0; a < d; a++ {
			s := 0.0
			for b := 0; b < d; b++ {
				s += m.covInv[k].At(a, b) * diff[b]
			}
			quad += diff[a] * s
		}
		out[k] = -0.5*quad - 0.5*m.logDet[k] + math.Log(m.priors[k]+1e-300)
	}
	return out
}

func (m *QDA) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		d := m.discriminant(x)
		out[i] = m.classes[argmaxF(d)]
	}
	return out
}

func (m *QDA) PredictProba(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, x := range X {
		out[i] = softmax(m.discriminant(x))
	}
	return out
}
