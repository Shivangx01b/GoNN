package ml

import (
	"math"
	"math/rand"
)

// LogisticRegression supports binary (2 classes) and multiclass via one-vs-rest.
type LogisticRegression struct {
	LR      float64
	MaxIter int
	Tol     float64
	Seed    int64

	// For binary: single weight vector + bias. For multiclass: per-class.
	Weights [][]float64
	Bias    []float64
	classes []int
}

func (m *LogisticRegression) defaults() {
	if m.LR == 0 {
		m.LR = 0.01
	}
	if m.MaxIter == 0 {
		m.MaxIter = 200
	}
	if m.Tol == 0 {
		m.Tol = 1e-4
	}
}

func (m *LogisticRegression) Fit(X [][]float64, y []int) {
	m.defaults()
	m.classes = uniqueInts(y)
	_, d := shapeOf(X)
	if len(m.classes) <= 2 {
		// Binary: classes treated as 0/1 (use second class as positive).
		pos := m.classes[len(m.classes)-1]
		yb := make([]float64, len(y))
		for i, v := range y {
			if v == pos {
				yb[i] = 1
			}
		}
		w, b := m.fitBinary(X, yb, d)
		m.Weights = [][]float64{w}
		m.Bias = []float64{b}
		return
	}
	// Multiclass one-vs-rest
	m.Weights = make([][]float64, len(m.classes))
	m.Bias = make([]float64, len(m.classes))
	for ci, c := range m.classes {
		yb := make([]float64, len(y))
		for i, v := range y {
			if v == c {
				yb[i] = 1
			}
		}
		w, b := m.fitBinary(X, yb, d)
		m.Weights[ci] = w
		m.Bias[ci] = b
	}
}

func (m *LogisticRegression) fitBinary(X [][]float64, y []float64, d int) ([]float64, float64) {
	r := rand.New(rand.NewSource(m.Seed))
	w := make([]float64, d)
	b := 0.0
	n := len(X)
	prevLoss := math.Inf(1)
	for it := 0; it < m.MaxIter; it++ {
		perm := randPerm(n, r)
		loss := 0.0
		for _, i := range perm {
			z := b + dot(X[i], w)
			p := sigmoid(z)
			err := p - y[i]
			for j := 0; j < d; j++ {
				w[j] -= m.LR * err * X[i][j]
			}
			b -= m.LR * err
			// Clipped log loss
			pp := math.Max(math.Min(p, 1-1e-12), 1e-12)
			loss += -(y[i]*math.Log(pp) + (1-y[i])*math.Log(1-pp))
		}
		loss /= float64(n)
		if math.Abs(prevLoss-loss) < m.Tol {
			break
		}
		prevLoss = loss
	}
	return w, b
}

// PredictProba returns class probabilities, shape [n][nClasses].
func (m *LogisticRegression) PredictProba(X [][]float64) [][]float64 {
	n := len(X)
	if len(m.classes) <= 2 {
		out := make([][]float64, n)
		for i := 0; i < n; i++ {
			p := sigmoid(m.Bias[0] + dot(X[i], m.Weights[0]))
			out[i] = []float64{1 - p, p}
		}
		return out
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, len(m.classes))
		var sum float64
		for c := range m.classes {
			row[c] = sigmoid(m.Bias[c] + dot(X[i], m.Weights[c]))
			sum += row[c]
		}
		if sum > 0 {
			for c := range row {
				row[c] /= sum
			}
		}
		out[i] = row
	}
	return out
}

func (m *LogisticRegression) Predict(X [][]float64) []int {
	probs := m.PredictProba(X)
	out := make([]int, len(X))
	for i := range probs {
		out[i] = m.classes[argmaxF(probs[i])]
	}
	return out
}
