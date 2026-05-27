package ml

import (
	"math"
	"math/rand"
)

// RandomForestClassifier is a bagged ensemble of DecisionTreeClassifier.
type RandomForestClassifier struct {
	NEstimators     int
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Seed            int64

	trees   []*DecisionTreeClassifier
	classes []int
}

func (m *RandomForestClassifier) Fit(X [][]float64, y []int) {
	if m.NEstimators == 0 {
		m.NEstimators = 100
	}
	_, d := shapeOf(X)
	if m.MaxFeatures == 0 {
		m.MaxFeatures = int(math.Sqrt(float64(d)))
		if m.MaxFeatures < 1 {
			m.MaxFeatures = 1
		}
	}
	m.classes = uniqueInts(y)
	r := rand.New(rand.NewSource(m.Seed))
	n := len(X)
	m.trees = make([]*DecisionTreeClassifier, m.NEstimators)
	for t := 0; t < m.NEstimators; t++ {
		idx := make([]int, n)
		for i := 0; i < n; i++ {
			idx[i] = r.Intn(n)
		}
		Xb := make([][]float64, n)
		yb := make([]int, n)
		for i, k := range idx {
			Xb[i] = X[k]
			yb[i] = y[k]
		}
		tree := &DecisionTreeClassifier{
			MaxDepth:        m.MaxDepth,
			MinSamplesSplit: m.MinSamplesSplit,
			MaxFeatures:     m.MaxFeatures,
			Seed:            r.Int63(),
		}
		tree.Fit(Xb, yb)
		m.trees[t] = tree
	}
}

// PredictProba averages per-tree probabilities, aligned to m.classes.
func (m *RandomForestClassifier) PredictProba(X [][]float64) [][]float64 {
	n := len(X)
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, len(m.classes))
	}
	classIdx := map[int]int{}
	for i, c := range m.classes {
		classIdx[c] = i
	}
	for _, tree := range m.trees {
		probs := tree.PredictProba(X)
		// Align tree.classes to forest.classes
		treeIdx := make([]int, len(tree.classes))
		for j, c := range tree.classes {
			treeIdx[j] = classIdx[c]
		}
		for i := 0; i < n; i++ {
			for j, p := range probs[i] {
				out[i][treeIdx[j]] += p
			}
		}
	}
	inv := 1.0 / float64(len(m.trees))
	for i := range out {
		for j := range out[i] {
			out[i][j] *= inv
		}
	}
	return out
}

func (m *RandomForestClassifier) Predict(X [][]float64) []int {
	probs := m.PredictProba(X)
	out := make([]int, len(X))
	for i, row := range probs {
		out[i] = m.classes[argmaxF(row)]
	}
	return out
}

// RandomForestRegressor is a bagged ensemble of DecisionTreeRegressor.
type RandomForestRegressor struct {
	NEstimators     int
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Seed            int64

	trees []*DecisionTreeRegressor
}

func (m *RandomForestRegressor) Fit(X [][]float64, y []float64) {
	if m.NEstimators == 0 {
		m.NEstimators = 100
	}
	_, d := shapeOf(X)
	if m.MaxFeatures == 0 {
		m.MaxFeatures = d / 3
		if m.MaxFeatures < 1 {
			m.MaxFeatures = 1
		}
	}
	r := rand.New(rand.NewSource(m.Seed))
	n := len(X)
	m.trees = make([]*DecisionTreeRegressor, m.NEstimators)
	for t := 0; t < m.NEstimators; t++ {
		idx := make([]int, n)
		for i := 0; i < n; i++ {
			idx[i] = r.Intn(n)
		}
		Xb := make([][]float64, n)
		yb := make([]float64, n)
		for i, k := range idx {
			Xb[i] = X[k]
			yb[i] = y[k]
		}
		tree := &DecisionTreeRegressor{
			MaxDepth:        m.MaxDepth,
			MinSamplesSplit: m.MinSamplesSplit,
			MaxFeatures:     m.MaxFeatures,
			Seed:            r.Int63(),
		}
		tree.Fit(Xb, yb)
		m.trees[t] = tree
	}
}

func (m *RandomForestRegressor) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for _, tree := range m.trees {
		preds := tree.Predict(X)
		for i, p := range preds {
			out[i] += p
		}
	}
	inv := 1.0 / float64(len(m.trees))
	for i := range out {
		out[i] *= inv
	}
	return out
}
