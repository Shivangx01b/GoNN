package ml

import (
	"fmt"
	"math/rand"
)

// TrainTestSplit splits X and y (which must be []float64 or []int) into train/test sets.
// Returns Xtr, Xte, ytr, yte; ytr/yte have the same underlying type as y.
func TrainTestSplit(X [][]float64, y interface{}, testSize float64, randomSeed int64) (
	Xtr, Xte [][]float64, ytr, yte interface{},
) {
	n := len(X)
	if testSize <= 0 || testSize >= 1 {
		testSize = 0.25
	}
	nTest := int(float64(n) * testSize)
	r := rand.New(rand.NewSource(randomSeed))
	perm := r.Perm(n)
	testIdx := perm[:nTest]
	trainIdx := perm[nTest:]
	Xtr = make([][]float64, len(trainIdx))
	Xte = make([][]float64, len(testIdx))
	for i, k := range trainIdx {
		Xtr[i] = X[k]
	}
	for i, k := range testIdx {
		Xte[i] = X[k]
	}
	switch yy := y.(type) {
	case []float64:
		ytrS := make([]float64, len(trainIdx))
		yteS := make([]float64, len(testIdx))
		for i, k := range trainIdx {
			ytrS[i] = yy[k]
		}
		for i, k := range testIdx {
			yteS[i] = yy[k]
		}
		ytr = ytrS
		yte = yteS
	case []int:
		ytrS := make([]int, len(trainIdx))
		yteS := make([]int, len(testIdx))
		for i, k := range trainIdx {
			ytrS[i] = yy[k]
		}
		for i, k := range testIdx {
			yteS[i] = yy[k]
		}
		ytr = ytrS
		yte = yteS
	default:
		panic(fmt.Sprintf("TrainTestSplit: unsupported y type %T", y))
	}
	return
}

// KFold splits indices into NSplits folds.
type KFold struct {
	NSplits int
	Shuffle bool
	Seed    int64
}

// Split returns NSplits pairs [trainIdx, testIdx].
func (k KFold) Split(n int) [][][]int {
	if k.NSplits <= 1 {
		k.NSplits = 5
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	if k.Shuffle {
		r := rand.New(rand.NewSource(k.Seed))
		r.Shuffle(n, func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	}
	folds := make([][]int, k.NSplits)
	base := n / k.NSplits
	rem := n % k.NSplits
	pos := 0
	for f := 0; f < k.NSplits; f++ {
		sz := base
		if f < rem {
			sz++
		}
		folds[f] = idx[pos : pos+sz]
		pos += sz
	}
	out := make([][][]int, k.NSplits)
	for f := 0; f < k.NSplits; f++ {
		test := folds[f]
		var train []int
		for g := 0; g < k.NSplits; g++ {
			if g != f {
				train = append(train, folds[g]...)
			}
		}
		out[f] = [][]int{train, test}
	}
	return out
}

// CrossValScore runs k-fold CV using the model. model may be a Classifier or Regressor.
// Returns score per fold (accuracy for classifiers, R2 for regressors).
func CrossValScore(model interface{}, X [][]float64, y interface{}, kSplits int) []float64 {
	if kSplits <= 1 {
		kSplits = 5
	}
	kf := KFold{NSplits: kSplits, Shuffle: true, Seed: 0}
	splits := kf.Split(len(X))
	scores := make([]float64, kSplits)
	for f, sp := range splits {
		train, test := sp[0], sp[1]
		Xtr := make([][]float64, len(train))
		Xte := make([][]float64, len(test))
		for i, k := range train {
			Xtr[i] = X[k]
		}
		for i, k := range test {
			Xte[i] = X[k]
		}
		switch m := model.(type) {
		case Classifier:
			yy := y.([]int)
			ytr := make([]int, len(train))
			yte := make([]int, len(test))
			for i, k := range train {
				ytr[i] = yy[k]
			}
			for i, k := range test {
				yte[i] = yy[k]
			}
			m.Fit(Xtr, ytr)
			scores[f] = Accuracy(yte, m.Predict(Xte))
		case Regressor:
			yy := y.([]float64)
			ytr := make([]float64, len(train))
			yte := make([]float64, len(test))
			for i, k := range train {
				ytr[i] = yy[k]
			}
			for i, k := range test {
				yte[i] = yy[k]
			}
			m.Fit(Xtr, ytr)
			scores[f] = R2Score(yte, m.Predict(Xte))
		default:
			panic(fmt.Sprintf("CrossValScore: model must implement Classifier or Regressor, got %T", model))
		}
	}
	return scores
}
