package ml

import (
	"math"
	"math/rand"
)

// AdaBoostClassifier implements the SAMME multi-class AdaBoost algorithm with
// DecisionTreeClassifier base learners.
type AdaBoostClassifier struct {
	NEstimators  int
	LearningRate float64
	MaxDepth     int
	Seed         int64

	estimators   []*DecisionTreeClassifier
	estWeights   []float64 // alpha for each estimator
	classes      []int
	nClasses     int
}

func (m *AdaBoostClassifier) Fit(X [][]float64, y []int) {
	if m.NEstimators == 0 {
		m.NEstimators = 50
	}
	if m.LearningRate == 0 {
		m.LearningRate = 1.0
	}
	if m.MaxDepth == 0 {
		m.MaxDepth = 1 // decision stumps
	}
	m.classes = uniqueInts(y)
	m.nClasses = len(m.classes)
	K := float64(m.nClasses)
	n := len(X)

	// Sample weights normalized to sum to 1.
	w := make([]float64, n)
	for i := range w {
		w[i] = 1.0 / float64(n)
	}

	m.estimators = nil
	m.estWeights = nil

	rng := rand.New(rand.NewSource(m.Seed))

	for t := 0; t < m.NEstimators; t++ {
		// Weighted resampling so the base learner sees the distribution.
		idx := weightedSample(w, n, rng)
		Xb := make([][]float64, n)
		yb := make([]int, n)
		for i, k := range idx {
			Xb[i] = X[k]
			yb[i] = y[k]
		}
		tree := &DecisionTreeClassifier{
			MaxDepth: m.MaxDepth,
			Seed:     rng.Int63(),
		}
		tree.Fit(Xb, yb)
		// Evaluate on original data with sample weights.
		preds := tree.Predict(X)
		errSum := 0.0
		for i := 0; i < n; i++ {
			if preds[i] != y[i] {
				errSum += w[i]
			}
		}
		// Clamp error
		if errSum >= 1.0-1e-12 {
			// Worse than chance; stop.
			if len(m.estimators) == 0 {
				m.estimators = append(m.estimators, tree)
				m.estWeights = append(m.estWeights, 1.0)
			}
			break
		}
		if errSum < 1e-12 {
			// Perfect tree: assign a large weight and stop.
			alpha := m.LearningRate * (math.Log((1-errSum+1e-12)/(errSum+1e-12)) + math.Log(K-1))
			m.estimators = append(m.estimators, tree)
			m.estWeights = append(m.estWeights, alpha)
			break
		}
		// SAMME estimator weight
		alpha := m.LearningRate * (math.Log((1-errSum)/errSum) + math.Log(K-1))
		m.estimators = append(m.estimators, tree)
		m.estWeights = append(m.estWeights, alpha)
		// Update sample weights
		for i := 0; i < n; i++ {
			if preds[i] != y[i] {
				w[i] *= math.Exp(alpha)
			}
		}
		// Renormalize
		sum := 0.0
		for _, v := range w {
			sum += v
		}
		if sum > 0 {
			for i := range w {
				w[i] /= sum
			}
		}
	}
}

// weightedSample returns n indices sampled with replacement according to w.
func weightedSample(w []float64, n int, r *rand.Rand) []int {
	// Build cumulative distribution.
	cum := make([]float64, len(w))
	sum := 0.0
	for i, v := range w {
		sum += v
		cum[i] = sum
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		u := r.Float64() * sum
		// Binary search.
		lo, hi := 0, len(cum)-1
		for lo < hi {
			mid := (lo + hi) / 2
			if cum[mid] < u {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		out[i] = lo
	}
	return out
}

// PredictProba returns class probabilities via softmax over weighted votes.
func (m *AdaBoostClassifier) PredictProba(X [][]float64) [][]float64 {
	n := len(X)
	// Score matrix [n][K]
	scores := make([][]float64, n)
	for i := range scores {
		scores[i] = make([]float64, m.nClasses)
	}
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}
	for t, tree := range m.estimators {
		preds := tree.Predict(X)
		for i, p := range preds {
			if ci, ok := classIdx[p]; ok {
				scores[i][ci] += m.estWeights[t]
			}
		}
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = softmax(scores[i])
	}
	return out
}

func (m *AdaBoostClassifier) Predict(X [][]float64) []int {
	probs := m.PredictProba(X)
	out := make([]int, len(X))
	for i, p := range probs {
		out[i] = m.classes[argmaxF(p)]
	}
	return out
}
