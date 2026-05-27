package ml

import (
	"sort"
)

// KNNClassifier is a k-nearest-neighbor classifier.
type KNNClassifier struct {
	K       int
	Weights string // "uniform" or "distance"
	Xtrain  [][]float64
	Ytrain  []int
	classes []int
}

func (m *KNNClassifier) Fit(X [][]float64, y []int) {
	if m.K <= 0 {
		m.K = 5
	}
	if m.Weights == "" {
		m.Weights = "uniform"
	}
	m.Xtrain = copyMatrix(X)
	m.Ytrain = append([]int(nil), y...)
	m.classes = uniqueInts(y)
}

type knnNeighbor struct {
	dist float64
	idx  int
}

func (m *KNNClassifier) neighbors(x []float64) []knnNeighbor {
	ds := make([]knnNeighbor, len(m.Xtrain))
	for i, xt := range m.Xtrain {
		ds[i] = knnNeighbor{dist: euclidean(x, xt), idx: i}
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i].dist < ds[j].dist })
	if m.K < len(ds) {
		ds = ds[:m.K]
	}
	return ds
}

func (m *KNNClassifier) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		nbrs := m.neighbors(x)
		votes := map[int]float64{}
		for _, n := range nbrs {
			w := 1.0
			if m.Weights == "distance" {
				if n.dist == 0 {
					votes[m.Ytrain[n.idx]] += 1e9
					continue
				}
				w = 1.0 / n.dist
			}
			votes[m.Ytrain[n.idx]] += w
		}
		var best int
		var bestV float64 = -1
		for k, v := range votes {
			if v > bestV {
				bestV = v
				best = k
			}
		}
		out[i] = best
	}
	return out
}

// KNNRegressor is a k-nearest-neighbor regressor.
type KNNRegressor struct {
	K       int
	Weights string
	Xtrain  [][]float64
	Ytrain  []float64
}

func (m *KNNRegressor) Fit(X [][]float64, y []float64) {
	if m.K <= 0 {
		m.K = 5
	}
	if m.Weights == "" {
		m.Weights = "uniform"
	}
	m.Xtrain = copyMatrix(X)
	m.Ytrain = append([]float64(nil), y...)
}

func (m *KNNRegressor) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		ds := make([]knnNeighbor, len(m.Xtrain))
		for j, xt := range m.Xtrain {
			ds[j] = knnNeighbor{dist: euclidean(x, xt), idx: j}
		}
		sort.Slice(ds, func(i, j int) bool { return ds[i].dist < ds[j].dist })
		k := m.K
		if k > len(ds) {
			k = len(ds)
		}
		ds = ds[:k]
		var sw, swy float64
		for _, n := range ds {
			w := 1.0
			if m.Weights == "distance" {
				if n.dist == 0 {
					sw += 1e9
					swy += 1e9 * m.Ytrain[n.idx]
					continue
				}
				w = 1.0 / n.dist
			}
			sw += w
			swy += w * m.Ytrain[n.idx]
		}
		if sw > 0 {
			out[i] = swy / sw
		}
	}
	return out
}
