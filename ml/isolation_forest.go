package ml

import (
	"math"
	"math/rand"
	"sort"
)

// iTreeNode is a node in an isolation tree.
type iTreeNode struct {
	leaf    bool
	size    int // # samples that reached this leaf
	depth   int // depth at which this leaf was created
	feature int
	thresh  float64
	left    *iTreeNode
	right   *iTreeNode
}

// cFactor is the average path length of unsuccessful BST search with n nodes.
func cFactor(n int) float64 {
	if n <= 1 {
		return 0
	}
	nf := float64(n)
	return 2.0*(math.Log(nf-1)+0.5772156649) - 2.0*(nf-1)/nf
}

// IsolationForest detects anomalies via random partitioning trees.
//
// Predict returns -1 for anomalies and 1 for normal points. The decision
// threshold is determined by ContaminationRate (fraction of training samples
// expected to be anomalies, in [0, 0.5]).
type IsolationForest struct {
	NEstimators       int
	MaxSamples        int
	ContaminationRate float64
	Seed              int64

	trees        []*iTreeNode
	heightLimit  int
	subsample    int
	threshold    float64 // decision threshold on score; score > threshold => anomaly
}

func (m *IsolationForest) Fit(X [][]float64) {
	if m.NEstimators == 0 {
		m.NEstimators = 100
	}
	n, d := shapeOf(X)
	if m.MaxSamples == 0 || m.MaxSamples > n {
		m.MaxSamples = 256
		if m.MaxSamples > n {
			m.MaxSamples = n
		}
	}
	m.subsample = m.MaxSamples
	m.heightLimit = int(math.Ceil(math.Log2(float64(m.subsample))))
	if m.heightLimit < 1 {
		m.heightLimit = 1
	}
	rng := rand.New(rand.NewSource(m.Seed))

	m.trees = make([]*iTreeNode, m.NEstimators)
	for t := 0; t < m.NEstimators; t++ {
		// Pick m.subsample random indices without replacement (Fisher-Yates partial).
		perm := rng.Perm(n)[:m.subsample]
		sub := make([][]float64, m.subsample)
		for i, k := range perm {
			sub[i] = X[k]
		}
		m.trees[t] = m.buildTree(sub, 0, d, rng)
	}

	// Set anomaly threshold from contamination rate on training scores.
	if m.ContaminationRate <= 0 {
		m.ContaminationRate = 0.1
	}
	scores := m.score(X)
	sorted := make([]float64, len(scores))
	copy(sorted, scores)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
	k := int(float64(n) * m.ContaminationRate)
	if k >= n {
		k = n - 1
	}
	if k < 0 {
		k = 0
	}
	m.threshold = sorted[k]
}

func (m *IsolationForest) buildTree(X [][]float64, depth, d int, rng *rand.Rand) *iTreeNode {
	n := len(X)
	if depth >= m.heightLimit || n <= 1 {
		return &iTreeNode{leaf: true, size: n, depth: depth}
	}
	// Try features in random order until we find one with non-zero range.
	feats := rng.Perm(d)
	for _, f := range feats {
		fmin, fmax := math.Inf(1), math.Inf(-1)
		for _, row := range X {
			if row[f] < fmin {
				fmin = row[f]
			}
			if row[f] > fmax {
				fmax = row[f]
			}
		}
		if fmax-fmin < 1e-12 {
			continue
		}
		thr := fmin + rng.Float64()*(fmax-fmin)
		var left, right [][]float64
		for _, row := range X {
			if row[f] < thr {
				left = append(left, row)
			} else {
				right = append(right, row)
			}
		}
		if len(left) == 0 || len(right) == 0 {
			continue
		}
		return &iTreeNode{
			feature: f,
			thresh:  thr,
			left:    m.buildTree(left, depth+1, d, rng),
			right:   m.buildTree(right, depth+1, d, rng),
		}
	}
	return &iTreeNode{leaf: true, size: n, depth: depth}
}

// pathLength returns the depth at which x is isolated, adjusted for unfinished
// trees by adding the average BST path length over the leaf's remaining size.
func pathLength(node *iTreeNode, x []float64, currentDepth int) float64 {
	if node.leaf {
		return float64(currentDepth) + cFactor(node.size)
	}
	if x[node.feature] < node.thresh {
		return pathLength(node.left, x, currentDepth+1)
	}
	return pathLength(node.right, x, currentDepth+1)
}

// score returns the anomaly score per point in [0, 1]; larger == more anomalous.
func (m *IsolationForest) score(X [][]float64) []float64 {
	out := make([]float64, len(X))
	c := cFactor(m.subsample)
	if c <= 0 {
		c = 1
	}
	for i, x := range X {
		avg := 0.0
		for _, tree := range m.trees {
			avg += pathLength(tree, x, 0)
		}
		avg /= float64(len(m.trees))
		out[i] = math.Pow(2.0, -avg/c)
	}
	return out
}

// AnomalyScore returns the raw isolation-forest score for each input row.
func (m *IsolationForest) AnomalyScore(X [][]float64) []float64 {
	return m.score(X)
}

// Predict returns -1 for anomalies and 1 for normal points.
func (m *IsolationForest) Predict(X [][]float64) []int {
	s := m.score(X)
	out := make([]int, len(X))
	for i, v := range s {
		if v >= m.threshold {
			out[i] = -1
		} else {
			out[i] = 1
		}
	}
	return out
}
