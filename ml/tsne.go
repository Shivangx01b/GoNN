package ml

import (
	"math"
	"math/rand"
)

// TSNE implements vanilla (non Barnes-Hut) t-distributed Stochastic Neighbor
// Embedding. Suitable for hundreds to a few thousand points; O(N^2) per iter.
type TSNE struct {
	NComponents  int
	Perplexity   float64
	LearningRate float64
	NIter        int
	Seed         int64
	EarlyExaggeration float64

	Embedding [][]float64
}

func (m *TSNE) Fit(X [][]float64) {
	if m.NComponents == 0 {
		m.NComponents = 2
	}
	if m.Perplexity == 0 {
		m.Perplexity = 30
	}
	if m.LearningRate == 0 {
		m.LearningRate = 200
	}
	if m.NIter == 0 {
		m.NIter = 500
	}
	if m.EarlyExaggeration == 0 {
		m.EarlyExaggeration = 12.0
	}
	n, _ := shapeOf(X)
	if n < 2 {
		m.Embedding = make([][]float64, n)
		for i := range m.Embedding {
			m.Embedding[i] = make([]float64, m.NComponents)
		}
		return
	}
	if m.Perplexity > float64(n-1)/3 {
		m.Perplexity = float64(n-1) / 3
		if m.Perplexity < 1 {
			m.Perplexity = 1
		}
	}
	rng := rand.New(rand.NewSource(m.Seed))

	// Compute squared distances in input space.
	dist2 := make([][]float64, n)
	for i := 0; i < n; i++ {
		dist2[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i != j {
				dist2[i][j] = squaredEuclidean(X[i], X[j])
			}
		}
	}

	// Compute conditional probabilities P_{j|i} via binary search for sigma.
	logPerp := math.Log(m.Perplexity)
	P := make([][]float64, n)
	for i := 0; i < n; i++ {
		P[i] = computeProw(dist2[i], i, logPerp)
	}
	// Symmetrize and normalize.
	totalP := 0.0
	Psym := make([][]float64, n)
	for i := 0; i < n; i++ {
		Psym[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Psym[i][j] = P[i][j] + P[j][i]
			totalP += Psym[i][j]
		}
	}
	if totalP == 0 {
		totalP = 1
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Psym[i][j] /= totalP
			if Psym[i][j] < 1e-12 {
				Psym[i][j] = 1e-12
			}
		}
	}

	// Initialize embedding from a small Gaussian.
	d := m.NComponents
	Y := make([][]float64, n)
	for i := 0; i < n; i++ {
		Y[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			Y[i][j] = rng.NormFloat64() * 1e-4
		}
	}
	dY := make([][]float64, n)
	for i := 0; i < n; i++ {
		dY[i] = make([]float64, d)
	}
	prevDY := make([][]float64, n)
	for i := 0; i < n; i++ {
		prevDY[i] = make([]float64, d)
	}

	exaggeration := m.EarlyExaggeration
	for iter := 0; iter < m.NIter; iter++ {
		// Phase out early exaggeration.
		if iter == 100 {
			exaggeration = 1.0
		}
		// Compute low-dim affinities Q (Student's t-distribution).
		qNum := make([][]float64, n)
		for i := 0; i < n; i++ {
			qNum[i] = make([]float64, n)
		}
		qSum := 0.0
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				ds := 0.0
				for k := 0; k < d; k++ {
					df := Y[i][k] - Y[j][k]
					ds += df * df
				}
				qNum[i][j] = 1.0 / (1.0 + ds)
				qSum += qNum[i][j]
			}
		}
		if qSum == 0 {
			qSum = 1
		}
		// Gradient: dC/dY_i = 4 * sum_j (P_ij - Q_ij) * q_num_ij * (Y_i - Y_j)
		for i := 0; i < n; i++ {
			for k := 0; k < d; k++ {
				dY[i][k] = 0
			}
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				q := qNum[i][j] / qSum
				if q < 1e-12 {
					q = 1e-12
				}
				coef := 4.0 * (exaggeration*Psym[i][j] - q) * qNum[i][j]
				for k := 0; k < d; k++ {
					dY[i][k] += coef * (Y[i][k] - Y[j][k])
				}
			}
		}
		// Momentum.
		momentum := 0.5
		if iter > 250 {
			momentum = 0.8
		}
		for i := 0; i < n; i++ {
			for k := 0; k < d; k++ {
				step := momentum*prevDY[i][k] - m.LearningRate*dY[i][k]
				Y[i][k] += step
				prevDY[i][k] = step
			}
		}
		// Center to prevent drift.
		mean := make([]float64, d)
		for i := 0; i < n; i++ {
			for k := 0; k < d; k++ {
				mean[k] += Y[i][k]
			}
		}
		for k := 0; k < d; k++ {
			mean[k] /= float64(n)
		}
		for i := 0; i < n; i++ {
			for k := 0; k < d; k++ {
				Y[i][k] -= mean[k]
			}
		}
	}
	m.Embedding = Y
}

// computeProw runs binary search on sigma so that the perplexity of the
// row matches the target.
func computeProw(distRow []float64, i int, logPerp float64) []float64 {
	n := len(distRow)
	beta := 1.0
	betaMin := math.Inf(-1)
	betaMax := math.Inf(1)
	out := make([]float64, n)
	for iter := 0; iter < 50; iter++ {
		// Compute P_{j|i} with current beta.
		sumP := 0.0
		for j := 0; j < n; j++ {
			if j == i {
				out[j] = 0
				continue
			}
			out[j] = math.Exp(-distRow[j] * beta)
			sumP += out[j]
		}
		if sumP == 0 {
			sumP = 1e-12
		}
		H := 0.0
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			out[j] /= sumP
		}
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			H += out[j] * distRow[j]
		}
		H = H*beta + math.Log(sumP)
		diff := H - logPerp
		if math.Abs(diff) < 1e-5 {
			break
		}
		if diff > 0 {
			betaMin = beta
			if math.IsInf(betaMax, 1) {
				beta *= 2
			} else {
				beta = (beta + betaMax) / 2
			}
		} else {
			betaMax = beta
			if math.IsInf(betaMin, -1) {
				beta /= 2
			} else {
				beta = (beta + betaMin) / 2
			}
		}
	}
	return out
}

// FitTransform fits and returns the embedding.
func (m *TSNE) FitTransform(X [][]float64) [][]float64 {
	m.Fit(X)
	return m.Embedding
}
