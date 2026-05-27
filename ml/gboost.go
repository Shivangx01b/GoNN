package ml

import (
	"math"
)

// GradientBoostingRegressor uses depth-limited regression trees as weak learners.
type GradientBoostingRegressor struct {
	NEstimators int
	LR          float64
	MaxDepth    int
	Seed        int64

	init  float64
	trees []*DecisionTreeRegressor
}

func (m *GradientBoostingRegressor) defaults() {
	if m.NEstimators == 0 {
		m.NEstimators = 100
	}
	if m.LR == 0 {
		m.LR = 0.1
	}
	if m.MaxDepth == 0 {
		m.MaxDepth = 3
	}
}

func (m *GradientBoostingRegressor) Fit(X [][]float64, y []float64) {
	m.defaults()
	// Initial prediction: mean of y.
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))
	m.init = mean
	pred := make([]float64, len(y))
	for i := range pred {
		pred[i] = mean
	}
	m.trees = make([]*DecisionTreeRegressor, 0, m.NEstimators)
	for t := 0; t < m.NEstimators; t++ {
		// Residuals
		res := make([]float64, len(y))
		for i := range y {
			res[i] = y[i] - pred[i]
		}
		tree := &DecisionTreeRegressor{MaxDepth: m.MaxDepth, Seed: m.Seed + int64(t)}
		tree.Fit(X, res)
		upd := tree.Predict(X)
		for i := range pred {
			pred[i] += m.LR * upd[i]
		}
		m.trees = append(m.trees, tree)
	}
}

func (m *GradientBoostingRegressor) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i := range out {
		out[i] = m.init
	}
	for _, tree := range m.trees {
		p := tree.Predict(X)
		for i := range out {
			out[i] += m.LR * p[i]
		}
	}
	return out
}

// GradientBoostingClassifier supports binary and multiclass classification.
// Binary: deviance loss (logistic). Multiclass: softmax with K trees per round.
type GradientBoostingClassifier struct {
	NEstimators int
	LR          float64
	MaxDepth    int
	Seed        int64

	classes []int
	// Binary: single sequence of trees + init logit.
	// Multiclass: trees[K][NEstimators] + init[K] logits.
	binTrees   []*DecisionTreeRegressor
	binInit    float64
	multiTrees [][]*DecisionTreeRegressor
	multiInit  []float64
}

func (m *GradientBoostingClassifier) defaults() {
	if m.NEstimators == 0 {
		m.NEstimators = 100
	}
	if m.LR == 0 {
		m.LR = 0.1
	}
	if m.MaxDepth == 0 {
		m.MaxDepth = 3
	}
}

func (m *GradientBoostingClassifier) Fit(X [][]float64, y []int) {
	m.defaults()
	m.classes = uniqueInts(y)
	n := len(y)
	if len(m.classes) <= 2 {
		// Binary
		pos := m.classes[len(m.classes)-1]
		yb := make([]float64, n)
		for i, v := range y {
			if v == pos {
				yb[i] = 1
			}
		}
		// Initial log-odds
		pPos := 0.0
		for _, v := range yb {
			pPos += v
		}
		pPos /= float64(n)
		pPos = math.Max(math.Min(pPos, 1-1e-9), 1e-9)
		m.binInit = math.Log(pPos / (1 - pPos))
		f := make([]float64, n)
		for i := range f {
			f[i] = m.binInit
		}
		m.binTrees = make([]*DecisionTreeRegressor, 0, m.NEstimators)
		for t := 0; t < m.NEstimators; t++ {
			grad := make([]float64, n)
			for i := range yb {
				p := sigmoid(f[i])
				grad[i] = yb[i] - p // negative gradient of log loss
			}
			tree := &DecisionTreeRegressor{MaxDepth: m.MaxDepth, Seed: m.Seed + int64(t)}
			tree.Fit(X, grad)
			upd := tree.Predict(X)
			for i := range f {
				f[i] += m.LR * upd[i]
			}
			m.binTrees = append(m.binTrees, tree)
		}
		return
	}
	// Multiclass
	K := len(m.classes)
	Y := make([][]float64, K)
	for k := 0; k < K; k++ {
		Y[k] = make([]float64, n)
		for i, v := range y {
			if v == m.classes[k] {
				Y[k][i] = 1
			}
		}
	}
	F := make([][]float64, K)
	m.multiInit = make([]float64, K)
	for k := 0; k < K; k++ {
		pK := 0.0
		for _, v := range Y[k] {
			pK += v
		}
		pK = math.Max(math.Min(pK/float64(n), 1-1e-9), 1e-9)
		m.multiInit[k] = math.Log(pK)
		F[k] = make([]float64, n)
		for i := range F[k] {
			F[k][i] = m.multiInit[k]
		}
	}
	m.multiTrees = make([][]*DecisionTreeRegressor, K)
	for k := 0; k < K; k++ {
		m.multiTrees[k] = make([]*DecisionTreeRegressor, 0, m.NEstimators)
	}
	for t := 0; t < m.NEstimators; t++ {
		// softmax over F columns
		P := make([][]float64, K)
		for k := 0; k < K; k++ {
			P[k] = make([]float64, n)
		}
		for i := 0; i < n; i++ {
			row := make([]float64, K)
			for k := 0; k < K; k++ {
				row[k] = F[k][i]
			}
			sm := softmax(row)
			for k := 0; k < K; k++ {
				P[k][i] = sm[k]
			}
		}
		for k := 0; k < K; k++ {
			grad := make([]float64, n)
			for i := range grad {
				grad[i] = Y[k][i] - P[k][i]
			}
			tree := &DecisionTreeRegressor{MaxDepth: m.MaxDepth, Seed: m.Seed + int64(t*K+k)}
			tree.Fit(X, grad)
			upd := tree.Predict(X)
			for i := range F[k] {
				F[k][i] += m.LR * upd[i]
			}
			m.multiTrees[k] = append(m.multiTrees[k], tree)
		}
	}
}

func (m *GradientBoostingClassifier) PredictProba(X [][]float64) [][]float64 {
	n := len(X)
	if len(m.classes) <= 2 {
		out := make([][]float64, n)
		f := make([]float64, n)
		for i := range f {
			f[i] = m.binInit
		}
		for _, tree := range m.binTrees {
			upd := tree.Predict(X)
			for i := range f {
				f[i] += m.LR * upd[i]
			}
		}
		for i := 0; i < n; i++ {
			p := sigmoid(f[i])
			out[i] = []float64{1 - p, p}
		}
		return out
	}
	K := len(m.classes)
	F := make([][]float64, K)
	for k := 0; k < K; k++ {
		F[k] = make([]float64, n)
		for i := range F[k] {
			F[k][i] = m.multiInit[k]
		}
	}
	for k := 0; k < K; k++ {
		for _, tree := range m.multiTrees[k] {
			upd := tree.Predict(X)
			for i := range F[k] {
				F[k][i] += m.LR * upd[i]
			}
		}
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, K)
		for k := 0; k < K; k++ {
			row[k] = F[k][i]
		}
		out[i] = softmax(row)
	}
	return out
}

func (m *GradientBoostingClassifier) Predict(X [][]float64) []int {
	probs := m.PredictProba(X)
	out := make([]int, len(X))
	for i, row := range probs {
		out[i] = m.classes[argmaxF(row)]
	}
	return out
}
