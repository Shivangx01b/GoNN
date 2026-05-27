package ml

import (
	"math"
	"math/rand"
	"sort"
)

// treeNode is a CART tree node.
type treeNode struct {
	leaf     bool
	feature  int
	thresh   float64
	left     *treeNode
	right    *treeNode
	// Classifier: class label. Regressor: mean value.
	classVal int
	regVal   float64
	// Classifier: probability distribution at leaf (over training classes).
	classProb []float64
}

// DecisionTreeClassifier implements CART classification with Gini impurity.
type DecisionTreeClassifier struct {
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int // 0 = use all features (no random subsampling)
	Seed            int64

	root    *treeNode
	classes []int
	rng     *rand.Rand
}

func (m *DecisionTreeClassifier) Fit(X [][]float64, y []int) {
	if m.MinSamplesSplit == 0 {
		m.MinSamplesSplit = 2
	}
	if m.MaxDepth == 0 {
		m.MaxDepth = 32
	}
	m.classes = uniqueInts(y)
	m.rng = rand.New(rand.NewSource(m.Seed))
	idx := make([]int, len(X))
	for i := range idx {
		idx[i] = i
	}
	m.root = m.buildCls(X, y, idx, 0)
}

func (m *DecisionTreeClassifier) classDist(y []int, idx []int) []float64 {
	dist := make([]float64, len(m.classes))
	for _, i := range idx {
		for c, lbl := range m.classes {
			if y[i] == lbl {
				dist[c]++
				break
			}
		}
	}
	for c := range dist {
		dist[c] /= float64(len(idx))
	}
	return dist
}

func gini(dist []float64) float64 {
	s := 1.0
	for _, p := range dist {
		s -= p * p
	}
	return s
}

func (m *DecisionTreeClassifier) buildCls(X [][]float64, y []int, idx []int, depth int) *treeNode {
	node := &treeNode{}
	dist := m.classDist(y, idx)
	node.classProb = dist
	// Majority class
	best := 0
	for i, p := range dist {
		if p > dist[best] {
			best = i
		}
	}
	node.classVal = m.classes[best]

	if depth >= m.MaxDepth || len(idx) < m.MinSamplesSplit || gini(dist) == 0 {
		node.leaf = true
		return node
	}
	feat, thr, ok := m.bestSplitCls(X, y, idx)
	if !ok {
		node.leaf = true
		return node
	}
	var leftIdx, rightIdx []int
	for _, i := range idx {
		if X[i][feat] <= thr {
			leftIdx = append(leftIdx, i)
		} else {
			rightIdx = append(rightIdx, i)
		}
	}
	if len(leftIdx) == 0 || len(rightIdx) == 0 {
		node.leaf = true
		return node
	}
	node.feature = feat
	node.thresh = thr
	node.left = m.buildCls(X, y, leftIdx, depth+1)
	node.right = m.buildCls(X, y, rightIdx, depth+1)
	return node
}

// chooseFeatures returns the feature indices to consider (with subsampling).
func chooseFeatures(d, maxFeats int, r *rand.Rand) []int {
	if maxFeats <= 0 || maxFeats >= d {
		out := make([]int, d)
		for i := range out {
			out[i] = i
		}
		return out
	}
	perm := r.Perm(d)
	return perm[:maxFeats]
}

func (m *DecisionTreeClassifier) bestSplitCls(X [][]float64, y []int, idx []int) (int, float64, bool) {
	_, d := shapeOf(X)
	feats := chooseFeatures(d, m.MaxFeatures, m.rng)
	bestGain := -1.0
	bestFeat := -1
	bestThr := 0.0
	parentDist := m.classDist(y, idx)
	parentGini := gini(parentDist)
	for _, f := range feats {
		// Collect (value, class) pairs sorted.
		type pair struct {
			v float64
			c int
		}
		ps := make([]pair, len(idx))
		for k, i := range idx {
			ps[k] = pair{X[i][f], y[i]}
		}
		sort.Slice(ps, func(i, j int) bool { return ps[i].v < ps[j].v })
		nL := 0
		nR := len(ps)
		distL := make([]float64, len(m.classes))
		distR := make([]float64, len(m.classes))
		for _, p := range ps {
			for c, lbl := range m.classes {
				if p.c == lbl {
					distR[c]++
					break
				}
			}
		}
		for i := 0; i < len(ps)-1; i++ {
			// Move ps[i] from right to left.
			var cidx int
			for c, lbl := range m.classes {
				if ps[i].c == lbl {
					cidx = c
					break
				}
			}
			distL[cidx]++
			distR[cidx]--
			nL++
			nR--
			if ps[i].v == ps[i+1].v {
				continue
			}
			pL := make([]float64, len(m.classes))
			pR := make([]float64, len(m.classes))
			for c := range distL {
				pL[c] = distL[c] / float64(nL)
				pR[c] = distR[c] / float64(nR)
			}
			g := parentGini - (float64(nL)/float64(len(ps)))*gini(pL) - (float64(nR)/float64(len(ps)))*gini(pR)
			if g > bestGain {
				bestGain = g
				bestFeat = f
				bestThr = 0.5 * (ps[i].v + ps[i+1].v)
			}
		}
	}
	if bestFeat == -1 || bestGain <= 0 {
		return -1, 0, false
	}
	return bestFeat, bestThr, true
}

func (m *DecisionTreeClassifier) predictOne(x []float64) int {
	n := m.root
	for !n.leaf {
		if x[n.feature] <= n.thresh {
			n = n.left
		} else {
			n = n.right
		}
	}
	return n.classVal
}

// PredictProba returns probabilities for each training class.
func (m *DecisionTreeClassifier) PredictProba(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i, x := range X {
		n := m.root
		for !n.leaf {
			if x[n.feature] <= n.thresh {
				n = n.left
			} else {
				n = n.right
			}
		}
		row := make([]float64, len(n.classProb))
		copy(row, n.classProb)
		out[i] = row
	}
	return out
}

func (m *DecisionTreeClassifier) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		out[i] = m.predictOne(x)
	}
	return out
}

// DecisionTreeRegressor implements CART regression with MSE.
type DecisionTreeRegressor struct {
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Seed            int64

	root *treeNode
	rng  *rand.Rand
}

func (m *DecisionTreeRegressor) Fit(X [][]float64, y []float64) {
	if m.MinSamplesSplit == 0 {
		m.MinSamplesSplit = 2
	}
	if m.MaxDepth == 0 {
		m.MaxDepth = 32
	}
	m.rng = rand.New(rand.NewSource(m.Seed))
	idx := make([]int, len(X))
	for i := range idx {
		idx[i] = i
	}
	m.root = m.buildReg(X, y, idx, 0)
}

func meanIdx(y []float64, idx []int) float64 {
	s := 0.0
	for _, i := range idx {
		s += y[i]
	}
	return s / float64(len(idx))
}

func mseIdx(y []float64, idx []int, mean float64) float64 {
	s := 0.0
	for _, i := range idx {
		d := y[i] - mean
		s += d * d
	}
	return s
}

func (m *DecisionTreeRegressor) buildReg(X [][]float64, y []float64, idx []int, depth int) *treeNode {
	node := &treeNode{}
	mean := meanIdx(y, idx)
	node.regVal = mean
	if depth >= m.MaxDepth || len(idx) < m.MinSamplesSplit {
		node.leaf = true
		return node
	}
	feat, thr, ok := m.bestSplitReg(X, y, idx)
	if !ok {
		node.leaf = true
		return node
	}
	var leftIdx, rightIdx []int
	for _, i := range idx {
		if X[i][feat] <= thr {
			leftIdx = append(leftIdx, i)
		} else {
			rightIdx = append(rightIdx, i)
		}
	}
	if len(leftIdx) == 0 || len(rightIdx) == 0 {
		node.leaf = true
		return node
	}
	node.feature = feat
	node.thresh = thr
	node.left = m.buildReg(X, y, leftIdx, depth+1)
	node.right = m.buildReg(X, y, rightIdx, depth+1)
	return node
}

func (m *DecisionTreeRegressor) bestSplitReg(X [][]float64, y []float64, idx []int) (int, float64, bool) {
	_, d := shapeOf(X)
	feats := chooseFeatures(d, m.MaxFeatures, m.rng)
	parentMean := meanIdx(y, idx)
	parentSSE := mseIdx(y, idx, parentMean)
	bestGain := 0.0
	bestFeat := -1
	bestThr := 0.0
	for _, f := range feats {
		type pair struct {
			v, t float64
		}
		ps := make([]pair, len(idx))
		for k, i := range idx {
			ps[k] = pair{X[i][f], y[i]}
		}
		sort.Slice(ps, func(i, j int) bool { return ps[i].v < ps[j].v })
		var sumL, sumR float64
		for _, p := range ps {
			sumR += p.t
		}
		var sumSqL, sumSqR float64
		for _, p := range ps {
			sumSqR += p.t * p.t
		}
		nL := 0
		nR := len(ps)
		for i := 0; i < len(ps)-1; i++ {
			sumL += ps[i].t
			sumR -= ps[i].t
			sumSqL += ps[i].t * ps[i].t
			sumSqR -= ps[i].t * ps[i].t
			nL++
			nR--
			if ps[i].v == ps[i+1].v {
				continue
			}
			mL := sumL / float64(nL)
			mR := sumR / float64(nR)
			sseL := sumSqL - float64(nL)*mL*mL
			sseR := sumSqR - float64(nR)*mR*mR
			gain := parentSSE - sseL - sseR
			if gain > bestGain {
				bestGain = gain
				bestFeat = f
				bestThr = 0.5 * (ps[i].v + ps[i+1].v)
			}
		}
	}
	if bestFeat == -1 || bestGain <= 1e-12 {
		return -1, 0, false
	}
	return bestFeat, bestThr, true
}

func (m *DecisionTreeRegressor) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		n := m.root
		for !n.leaf {
			if x[n.feature] <= n.thresh {
				n = n.left
			} else {
				n = n.right
			}
		}
		out[i] = n.regVal
	}
	return out
}

var _ = math.Inf
