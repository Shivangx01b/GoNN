package ml

import (
	"math"
)

// GaussianNB is a Gaussian Naive Bayes classifier.
type GaussianNB struct {
	classes []int
	priors  []float64
	mean    [][]float64 // [nClasses][d]
	varc    [][]float64 // [nClasses][d]
}

func (m *GaussianNB) Fit(X [][]float64, y []int) {
	n, d := shapeOf(X)
	m.classes = uniqueInts(y)
	K := len(m.classes)
	m.priors = make([]float64, K)
	m.mean = make([][]float64, K)
	m.varc = make([][]float64, K)
	counts := make([]int, K)
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
		m.mean[i] = make([]float64, d)
		m.varc[i] = make([]float64, d)
	}
	for i := 0; i < n; i++ {
		c := classIdx[y[i]]
		counts[c]++
		for j := 0; j < d; j++ {
			m.mean[c][j] += X[i][j]
		}
	}
	for c := 0; c < K; c++ {
		for j := 0; j < d; j++ {
			m.mean[c][j] /= float64(counts[c])
		}
	}
	for i := 0; i < n; i++ {
		c := classIdx[y[i]]
		for j := 0; j < d; j++ {
			diff := X[i][j] - m.mean[c][j]
			m.varc[c][j] += diff * diff
		}
	}
	for c := 0; c < K; c++ {
		for j := 0; j < d; j++ {
			m.varc[c][j] = m.varc[c][j]/float64(counts[c]) + 1e-9
		}
		m.priors[c] = float64(counts[c]) / float64(n)
	}
}

func (m *GaussianNB) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		K := len(m.classes)
		logp := make([]float64, K)
		for c := 0; c < K; c++ {
			lp := math.Log(m.priors[c])
			for j := 0; j < len(x); j++ {
				v := m.varc[c][j]
				diff := x[j] - m.mean[c][j]
				lp += -0.5*math.Log(2*math.Pi*v) - (diff*diff)/(2*v)
			}
			logp[c] = lp
		}
		out[i] = m.classes[argmaxF(logp)]
	}
	return out
}

// MultinomialNB for count features with Laplace smoothing (alpha=1.0).
type MultinomialNB struct {
	Alpha       float64
	classes     []int
	priorsLog   []float64
	featLogProb [][]float64 // [nClasses][d]
}

func (m *MultinomialNB) Fit(X [][]float64, y []int) {
	if m.Alpha == 0 {
		m.Alpha = 1.0
	}
	n, d := shapeOf(X)
	m.classes = uniqueInts(y)
	K := len(m.classes)
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}
	counts := make([]int, K)
	featCount := make([][]float64, K)
	for c := 0; c < K; c++ {
		featCount[c] = make([]float64, d)
	}
	for i := 0; i < n; i++ {
		c := classIdx[y[i]]
		counts[c]++
		for j := 0; j < d; j++ {
			featCount[c][j] += X[i][j]
		}
	}
	m.priorsLog = make([]float64, K)
	m.featLogProb = make([][]float64, K)
	for c := 0; c < K; c++ {
		m.priorsLog[c] = math.Log(float64(counts[c]) / float64(n))
		total := 0.0
		for j := 0; j < d; j++ {
			total += featCount[c][j]
		}
		denom := total + m.Alpha*float64(d)
		m.featLogProb[c] = make([]float64, d)
		for j := 0; j < d; j++ {
			m.featLogProb[c][j] = math.Log((featCount[c][j] + m.Alpha) / denom)
		}
	}
}

func (m *MultinomialNB) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		K := len(m.classes)
		logp := make([]float64, K)
		for c := 0; c < K; c++ {
			s := m.priorsLog[c]
			for j, v := range x {
				s += v * m.featLogProb[c][j]
			}
			logp[c] = s
		}
		out[i] = m.classes[argmaxF(logp)]
	}
	return out
}

// BernoulliNB treats features as binary (0/1) with Laplace smoothing.
type BernoulliNB struct {
	Alpha     float64
	Binarize  float64 // threshold above which feature counts as 1
	classes   []int
	priorsLog []float64
	probT     [][]float64 // P(feat=1|class)
	probF     [][]float64 // P(feat=0|class)
}

func (m *BernoulliNB) Fit(X [][]float64, y []int) {
	if m.Alpha == 0 {
		m.Alpha = 1.0
	}
	n, d := shapeOf(X)
	m.classes = uniqueInts(y)
	K := len(m.classes)
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}
	counts := make([]int, K)
	featOn := make([][]float64, K)
	for c := 0; c < K; c++ {
		featOn[c] = make([]float64, d)
	}
	for i := 0; i < n; i++ {
		c := classIdx[y[i]]
		counts[c]++
		for j := 0; j < d; j++ {
			if X[i][j] > m.Binarize {
				featOn[c][j]++
			}
		}
	}
	m.priorsLog = make([]float64, K)
	m.probT = make([][]float64, K)
	m.probF = make([][]float64, K)
	for c := 0; c < K; c++ {
		m.priorsLog[c] = math.Log(float64(counts[c]) / float64(n))
		m.probT[c] = make([]float64, d)
		m.probF[c] = make([]float64, d)
		for j := 0; j < d; j++ {
			p := (featOn[c][j] + m.Alpha) / (float64(counts[c]) + 2*m.Alpha)
			m.probT[c][j] = math.Log(p)
			m.probF[c][j] = math.Log(1 - p)
		}
	}
}

func (m *BernoulliNB) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		K := len(m.classes)
		logp := make([]float64, K)
		for c := 0; c < K; c++ {
			s := m.priorsLog[c]
			for j, v := range x {
				if v > m.Binarize {
					s += m.probT[c][j]
				} else {
					s += m.probF[c][j]
				}
			}
			logp[c] = s
		}
		out[i] = m.classes[argmaxF(logp)]
	}
	return out
}
