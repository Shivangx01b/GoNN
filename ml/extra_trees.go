package ml

import (
	"math"
	"math/rand"
)

// extraTreeNode mirrors treeNode but lives separately so we can tag random
// splits and keep ExtraTrees independent from the CART base.
type extraTreeNode struct {
	leaf      bool
	feature   int
	thresh    float64
	left      *extraTreeNode
	right     *extraTreeNode
	classVal  int
	regVal    float64
	classProb []float64
}

// extraTreeClassifier builds a randomized decision tree for classification.
// At each node it picks MaxFeatures features and, for each, a single random
// threshold; it then keeps the split with the highest Gini gain.
type extraTreeClassifier struct {
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Seed            int64

	root    *extraTreeNode
	classes []int
	rng     *rand.Rand
}

func (m *extraTreeClassifier) fit(X [][]float64, y []int) {
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

func (m *extraTreeClassifier) classDist(y []int, idx []int) []float64 {
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

func (m *extraTreeClassifier) buildCls(X [][]float64, y []int, idx []int, depth int) *extraTreeNode {
	node := &extraTreeNode{}
	dist := m.classDist(y, idx)
	node.classProb = dist
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
	feat, thr, ok := m.randSplitCls(X, y, idx)
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

func (m *extraTreeClassifier) randSplitCls(X [][]float64, y []int, idx []int) (int, float64, bool) {
	_, d := shapeOf(X)
	feats := chooseFeatures(d, m.MaxFeatures, m.rng)
	parentDist := m.classDist(y, idx)
	parentGini := gini(parentDist)
	bestGain := -1.0
	bestFeat := -1
	bestThr := 0.0
	for _, f := range feats {
		// Find min/max of feature f in idx.
		fmin, fmax := math.Inf(1), math.Inf(-1)
		for _, i := range idx {
			v := X[i][f]
			if v < fmin {
				fmin = v
			}
			if v > fmax {
				fmax = v
			}
		}
		if fmax-fmin < 1e-12 {
			continue
		}
		thr := fmin + m.rng.Float64()*(fmax-fmin)
		// Compute Gini gain.
		distL := make([]float64, len(m.classes))
		distR := make([]float64, len(m.classes))
		nL, nR := 0, 0
		for _, i := range idx {
			var cidx int
			for c, lbl := range m.classes {
				if y[i] == lbl {
					cidx = c
					break
				}
			}
			if X[i][f] <= thr {
				distL[cidx]++
				nL++
			} else {
				distR[cidx]++
				nR++
			}
		}
		if nL == 0 || nR == 0 {
			continue
		}
		pL := make([]float64, len(m.classes))
		pR := make([]float64, len(m.classes))
		for c := range distL {
			pL[c] = distL[c] / float64(nL)
			pR[c] = distR[c] / float64(nR)
		}
		total := float64(nL + nR)
		gain := parentGini - (float64(nL)/total)*gini(pL) - (float64(nR)/total)*gini(pR)
		if gain > bestGain {
			bestGain = gain
			bestFeat = f
			bestThr = thr
		}
	}
	if bestFeat == -1 || bestGain <= 0 {
		return -1, 0, false
	}
	return bestFeat, bestThr, true
}

func (m *extraTreeClassifier) predictProba(X [][]float64) [][]float64 {
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

// extraTreeRegressor builds a randomized decision tree for regression.
type extraTreeRegressor struct {
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Seed            int64

	root *extraTreeNode
	rng  *rand.Rand
}

func (m *extraTreeRegressor) fit(X [][]float64, y []float64) {
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

func (m *extraTreeRegressor) buildReg(X [][]float64, y []float64, idx []int, depth int) *extraTreeNode {
	node := &extraTreeNode{}
	node.regVal = meanIdx(y, idx)
	if depth >= m.MaxDepth || len(idx) < m.MinSamplesSplit {
		node.leaf = true
		return node
	}
	feat, thr, ok := m.randSplitReg(X, y, idx)
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

func (m *extraTreeRegressor) randSplitReg(X [][]float64, y []float64, idx []int) (int, float64, bool) {
	_, d := shapeOf(X)
	feats := chooseFeatures(d, m.MaxFeatures, m.rng)
	parentMean := meanIdx(y, idx)
	parentSSE := mseIdx(y, idx, parentMean)
	bestGain := 0.0
	bestFeat := -1
	bestThr := 0.0
	for _, f := range feats {
		fmin, fmax := math.Inf(1), math.Inf(-1)
		for _, i := range idx {
			v := X[i][f]
			if v < fmin {
				fmin = v
			}
			if v > fmax {
				fmax = v
			}
		}
		if fmax-fmin < 1e-12 {
			continue
		}
		thr := fmin + m.rng.Float64()*(fmax-fmin)
		var sumL, sumR, sumSqL, sumSqR float64
		nL, nR := 0, 0
		for _, i := range idx {
			if X[i][f] <= thr {
				sumL += y[i]
				sumSqL += y[i] * y[i]
				nL++
			} else {
				sumR += y[i]
				sumSqR += y[i] * y[i]
				nR++
			}
		}
		if nL == 0 || nR == 0 {
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
			bestThr = thr
		}
	}
	if bestFeat == -1 || bestGain <= 1e-12 {
		return -1, 0, false
	}
	return bestFeat, bestThr, true
}

func (m *extraTreeRegressor) predict(X [][]float64) []float64 {
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

// ExtraTreesClassifier is an ensemble of randomized decision trees for
// classification. Unlike RandomForest, splits are drawn at random and trees
// are typically trained on the full sample (no bootstrap by default).
type ExtraTreesClassifier struct {
	NEstimators     int
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Bootstrap       bool
	Seed            int64

	trees   []*extraTreeClassifier
	classes []int
}

func (m *ExtraTreesClassifier) Fit(X [][]float64, y []int) {
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
	m.trees = make([]*extraTreeClassifier, m.NEstimators)
	for t := 0; t < m.NEstimators; t++ {
		Xb, yb := X, y
		if m.Bootstrap {
			Xb = make([][]float64, n)
			yb = make([]int, n)
			for i := 0; i < n; i++ {
				k := r.Intn(n)
				Xb[i] = X[k]
				yb[i] = y[k]
			}
		}
		tree := &extraTreeClassifier{
			MaxDepth:        m.MaxDepth,
			MinSamplesSplit: m.MinSamplesSplit,
			MaxFeatures:     m.MaxFeatures,
			Seed:            r.Int63(),
		}
		tree.fit(Xb, yb)
		m.trees[t] = tree
	}
}

func (m *ExtraTreesClassifier) PredictProba(X [][]float64) [][]float64 {
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
		probs := tree.predictProba(X)
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

func (m *ExtraTreesClassifier) Predict(X [][]float64) []int {
	probs := m.PredictProba(X)
	out := make([]int, len(X))
	for i, row := range probs {
		out[i] = m.classes[argmaxF(row)]
	}
	return out
}

// ExtraTreesRegressor is an ensemble of randomized regression trees.
type ExtraTreesRegressor struct {
	NEstimators     int
	MaxDepth        int
	MinSamplesSplit int
	MaxFeatures     int
	Bootstrap       bool
	Seed            int64

	trees []*extraTreeRegressor
}

func (m *ExtraTreesRegressor) Fit(X [][]float64, y []float64) {
	if m.NEstimators == 0 {
		m.NEstimators = 100
	}
	_, d := shapeOf(X)
	if m.MaxFeatures == 0 {
		m.MaxFeatures = d
		if m.MaxFeatures < 1 {
			m.MaxFeatures = 1
		}
	}
	r := rand.New(rand.NewSource(m.Seed))
	n := len(X)
	m.trees = make([]*extraTreeRegressor, m.NEstimators)
	for t := 0; t < m.NEstimators; t++ {
		Xb, yb := X, y
		if m.Bootstrap {
			Xb = make([][]float64, n)
			yb = make([]float64, n)
			for i := 0; i < n; i++ {
				k := r.Intn(n)
				Xb[i] = X[k]
				yb[i] = y[k]
			}
		}
		tree := &extraTreeRegressor{
			MaxDepth:        m.MaxDepth,
			MinSamplesSplit: m.MinSamplesSplit,
			MaxFeatures:     m.MaxFeatures,
			Seed:            r.Int63(),
		}
		tree.fit(Xb, yb)
		m.trees[t] = tree
	}
}

func (m *ExtraTreesRegressor) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for _, tree := range m.trees {
		preds := tree.predict(X)
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
