package ml

import (
	"math/rand"
)

// LinearSVC is a binary/multiclass linear SVM trained via subgradient SGD on hinge loss.
type LinearSVC struct {
	C       float64 // inverse regularization (higher = less reg)
	MaxIter int
	LR      float64
	Seed    int64

	classes []int
	// Binary: single weight + bias. Multiclass: per-class via OvR.
	Weights [][]float64
	Bias    []float64
}

func (m *LinearSVC) defaults() {
	if m.C == 0 {
		m.C = 1.0
	}
	if m.MaxIter == 0 {
		m.MaxIter = 1000
	}
	if m.LR == 0 {
		m.LR = 0.01
	}
}

func (m *LinearSVC) Fit(X [][]float64, y []int) {
	m.defaults()
	m.classes = uniqueInts(y)
	_, d := shapeOf(X)
	if len(m.classes) <= 2 {
		pos := m.classes[len(m.classes)-1]
		yb := make([]float64, len(y))
		for i, v := range y {
			if v == pos {
				yb[i] = 1
			} else {
				yb[i] = -1
			}
		}
		w, b := m.fitBinary(X, yb, d)
		m.Weights = [][]float64{w}
		m.Bias = []float64{b}
		return
	}
	m.Weights = make([][]float64, len(m.classes))
	m.Bias = make([]float64, len(m.classes))
	for ci, c := range m.classes {
		yb := make([]float64, len(y))
		for i, v := range y {
			if v == c {
				yb[i] = 1
			} else {
				yb[i] = -1
			}
		}
		w, b := m.fitBinary(X, yb, d)
		m.Weights[ci] = w
		m.Bias[ci] = b
	}
}

func (m *LinearSVC) fitBinary(X [][]float64, y []float64, d int) ([]float64, float64) {
	r := rand.New(rand.NewSource(m.Seed))
	w := make([]float64, d)
	b := 0.0
	n := len(X)
	lambda := 1.0 / m.C
	for it := 0; it < m.MaxIter; it++ {
		perm := randPerm(n, r)
		// Decay LR
		lr := m.LR / (1 + float64(it)*0.01)
		for _, i := range perm {
			margin := y[i] * (b + dot(X[i], w))
			if margin < 1 {
				// gradient: lambda*w - y*x for w; -y for b
				for j := 0; j < d; j++ {
					w[j] -= lr * (lambda*w[j] - y[i]*X[i][j])
				}
				b -= lr * (-y[i])
			} else {
				for j := 0; j < d; j++ {
					w[j] -= lr * lambda * w[j]
				}
			}
		}
	}
	return w, b
}

func (m *LinearSVC) decision(x []float64) []float64 {
	scores := make([]float64, len(m.Weights))
	for c := range m.Weights {
		scores[c] = m.Bias[c] + dot(x, m.Weights[c])
	}
	return scores
}

func (m *LinearSVC) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		if len(m.classes) <= 2 {
			s := m.Bias[0] + dot(x, m.Weights[0])
			if s >= 0 {
				out[i] = m.classes[len(m.classes)-1]
			} else {
				out[i] = m.classes[0]
			}
		} else {
			scores := m.decision(x)
			out[i] = m.classes[argmaxF(scores)]
		}
	}
	return out
}
