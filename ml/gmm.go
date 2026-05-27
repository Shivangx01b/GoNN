package ml

import (
	"math"
	"math/rand"
)

// GaussianMixture is a Gaussian mixture model fit via Expectation-Maximization.
//
// Covariance type is diagonal (per-component, per-feature variance) which is
// fast and works well in moderate dimension.
type GaussianMixture struct {
	NComponents int
	MaxIter     int
	Tol         float64
	Seed        int64

	// Fitted parameters
	Weights     []float64     // [K]
	Means       [][]float64   // [K][d]
	Covariances [][][]float64 // [K][d][d] -- stored as full matrices (diagonal in practice)
	// Stored separately for fast scoring.
	variances  [][]float64 // [K][d]
	logDets    []float64
	dim        int
	converged  bool
	LogLikelihood float64
}

func (m *GaussianMixture) Fit(X [][]float64) {
	if m.NComponents == 0 {
		m.NComponents = 1
	}
	if m.MaxIter == 0 {
		m.MaxIter = 100
	}
	if m.Tol == 0 {
		m.Tol = 1e-4
	}
	n, d := shapeOf(X)
	m.dim = d
	K := m.NComponents
	rng := rand.New(rand.NewSource(m.Seed))

	// Initialize with k-means++ style choice from data points.
	m.Means = make([][]float64, K)
	idxs := rng.Perm(n)[:K]
	for k, i := range idxs {
		m.Means[k] = append([]float64(nil), X[i]...)
	}
	m.Weights = make([]float64, K)
	for k := range m.Weights {
		m.Weights[k] = 1.0 / float64(K)
	}
	// Init variances from overall feature variance.
	overall := meanCol(X)
	varOverall := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			df := X[i][j] - overall[j]
			varOverall[j] += df * df
		}
	}
	for j := 0; j < d; j++ {
		varOverall[j] /= float64(n)
		if varOverall[j] < 1e-6 {
			varOverall[j] = 1e-6
		}
	}
	m.variances = make([][]float64, K)
	for k := 0; k < K; k++ {
		m.variances[k] = append([]float64(nil), varOverall...)
	}
	m.logDets = make([]float64, K)
	m.recomputeLogDets()

	prevLL := math.Inf(-1)
	resp := make([][]float64, n)
	for i := range resp {
		resp[i] = make([]float64, K)
	}
	for iter := 0; iter < m.MaxIter; iter++ {
		// E-step
		ll := 0.0
		for i := 0; i < n; i++ {
			logp := make([]float64, K)
			for k := 0; k < K; k++ {
				logp[k] = math.Log(m.Weights[k]+1e-300) + m.logGaussian(X[i], k)
			}
			// log-sum-exp
			maxlp := logp[0]
			for _, v := range logp[1:] {
				if v > maxlp {
					maxlp = v
				}
			}
			sum := 0.0
			for k := 0; k < K; k++ {
				logp[k] = math.Exp(logp[k] - maxlp)
				sum += logp[k]
			}
			for k := 0; k < K; k++ {
				resp[i][k] = logp[k] / sum
			}
			ll += maxlp + math.Log(sum)
		}
		// M-step
		for k := 0; k < K; k++ {
			nk := 0.0
			for i := 0; i < n; i++ {
				nk += resp[i][k]
			}
			if nk < 1e-12 {
				nk = 1e-12
			}
			m.Weights[k] = nk / float64(n)
			// Mean
			for j := 0; j < d; j++ {
				s := 0.0
				for i := 0; i < n; i++ {
					s += resp[i][k] * X[i][j]
				}
				m.Means[k][j] = s / nk
			}
			// Diagonal covariance
			for j := 0; j < d; j++ {
				s := 0.0
				for i := 0; i < n; i++ {
					df := X[i][j] - m.Means[k][j]
					s += resp[i][k] * df * df
				}
				v := s / nk
				if v < 1e-6 {
					v = 1e-6
				}
				m.variances[k][j] = v
			}
		}
		m.recomputeLogDets()
		if math.Abs(ll-prevLL) < m.Tol {
			m.converged = true
			m.LogLikelihood = ll
			break
		}
		prevLL = ll
		m.LogLikelihood = ll
	}

	// Materialize full covariance matrices for external consumers.
	m.Covariances = make([][][]float64, K)
	for k := 0; k < K; k++ {
		m.Covariances[k] = make([][]float64, d)
		for j := 0; j < d; j++ {
			m.Covariances[k][j] = make([]float64, d)
			m.Covariances[k][j][j] = m.variances[k][j]
		}
	}
}

func (m *GaussianMixture) recomputeLogDets() {
	for k := 0; k < m.NComponents; k++ {
		s := 0.0
		for j := 0; j < m.dim; j++ {
			s += math.Log(m.variances[k][j])
		}
		m.logDets[k] = s
	}
}

// logGaussian returns log N(x | mu_k, diag(sigma2_k)).
func (m *GaussianMixture) logGaussian(x []float64, k int) float64 {
	d := m.dim
	quad := 0.0
	for j := 0; j < d; j++ {
		df := x[j] - m.Means[k][j]
		quad += df * df / m.variances[k][j]
	}
	return -0.5*(float64(d)*math.Log(2*math.Pi)+m.logDets[k]+quad)
}

// PredictProba returns posterior probabilities over components.
func (m *GaussianMixture) PredictProba(X [][]float64) [][]float64 {
	K := m.NComponents
	out := make([][]float64, len(X))
	for i, x := range X {
		logp := make([]float64, K)
		for k := 0; k < K; k++ {
			logp[k] = math.Log(m.Weights[k]+1e-300) + m.logGaussian(x, k)
		}
		out[i] = softmax(logp)
	}
	return out
}

func (m *GaussianMixture) Predict(X [][]float64) []int {
	probs := m.PredictProba(X)
	out := make([]int, len(X))
	for i, p := range probs {
		out[i] = argmaxF(p)
	}
	return out
}
