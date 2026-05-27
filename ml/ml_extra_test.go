package ml

import (
	"math"
	"math/rand"
	"testing"
)

func makeLinearDataset(n int, w []float64, b, noise float64, seed int64) ([][]float64, []float64) {
	r := rand.New(rand.NewSource(seed))
	d := len(w)
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, d)
		s := b
		for j := 0; j < d; j++ {
			X[i][j] = r.NormFloat64()
			s += X[i][j] * w[j]
		}
		y[i] = s + r.NormFloat64()*noise
	}
	return X, y
}

func makeSeparable(n int, seed int64) ([][]float64, []int) {
	r := rand.New(rand.NewSource(seed))
	X := make([][]float64, 0, 2*n)
	y := make([]int, 0, 2*n)
	for i := 0; i < n; i++ {
		X = append(X, []float64{r.NormFloat64() - 2, r.NormFloat64() - 2})
		y = append(y, 0)
		X = append(X, []float64{r.NormFloat64() + 2, r.NormFloat64() + 2})
		y = append(y, 1)
	}
	return X, y
}

func TestLassoRecoversCoefficients(t *testing.T) {
	w := []float64{3.0, -1.5, 0.0, 0.0}
	X, y := makeLinearDataset(200, w, 0.5, 0.01, 1)
	m := &Lasso{Alpha: 0.01, MaxIter: 2000, Tol: 1e-6}
	m.Fit(X, y)
	for i, want := range w {
		if math.Abs(m.Weights[i]-want) > 0.2 {
			t.Fatalf("Lasso w[%d]=%v want %v", i, m.Weights[i], want)
		}
	}
	if math.Abs(m.Bias-0.5) > 0.2 {
		t.Fatalf("Lasso bias=%v want ~0.5", m.Bias)
	}
}

func TestElasticNetRecoversCoefficients(t *testing.T) {
	w := []float64{2.0, -1.0}
	X, y := makeLinearDataset(200, w, 0.0, 0.01, 2)
	m := &ElasticNet{Alpha: 0.01, L1Ratio: 0.5, MaxIter: 2000, Tol: 1e-6}
	m.Fit(X, y)
	for i, want := range w {
		if math.Abs(m.Weights[i]-want) > 0.2 {
			t.Fatalf("ElasticNet w[%d]=%v want %v", i, m.Weights[i], want)
		}
	}
}

func TestAdaBoostClassifierSeparable(t *testing.T) {
	X, y := makeSeparable(100, 3)
	m := &AdaBoostClassifier{NEstimators: 30, MaxDepth: 1, Seed: 1}
	m.Fit(X, y)
	preds := m.Predict(X)
	correct := 0
	for i := range preds {
		if preds[i] == y[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(y))
	if acc < 0.8 {
		t.Fatalf("AdaBoost accuracy %.3f < 0.8", acc)
	}
}

func TestExtraTreesClassifierSeparable(t *testing.T) {
	X, y := makeSeparable(100, 4)
	m := &ExtraTreesClassifier{NEstimators: 30, MaxDepth: 6, Seed: 5}
	m.Fit(X, y)
	preds := m.Predict(X)
	correct := 0
	for i := range preds {
		if preds[i] == y[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(y))
	if acc < 0.8 {
		t.Fatalf("ExtraTreesClassifier accuracy %.3f < 0.8", acc)
	}
}

func TestExtraTreesRegressorFitsLine(t *testing.T) {
	w := []float64{2.5}
	X, y := makeLinearDataset(120, w, 1.0, 0.05, 6)
	m := &ExtraTreesRegressor{NEstimators: 30, MaxDepth: 8, Seed: 6}
	m.Fit(X, y)
	preds := m.Predict(X)
	rss, tss := 0.0, 0.0
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))
	for i, p := range preds {
		df := y[i] - p
		rss += df * df
		dt := y[i] - mean
		tss += dt * dt
	}
	r2 := 1 - rss/tss
	if r2 < 0.8 {
		t.Fatalf("ExtraTreesRegressor R^2=%.3f", r2)
	}
}

func TestIsolationForestDetectsAnomalies(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	var X [][]float64
	for i := 0; i < 200; i++ {
		X = append(X, []float64{r.NormFloat64() * 0.5, r.NormFloat64() * 0.5})
	}
	// Add 10 obvious anomalies far away.
	anomalyStart := len(X)
	for i := 0; i < 10; i++ {
		X = append(X, []float64{10 + r.Float64(), 10 + r.Float64()})
	}
	m := &IsolationForest{NEstimators: 80, MaxSamples: 64, ContaminationRate: 0.05, Seed: 7}
	m.Fit(X)
	preds := m.Predict(X)
	// Anomalies must mostly be flagged.
	flagged := 0
	for i := anomalyStart; i < len(X); i++ {
		if preds[i] == -1 {
			flagged++
		}
	}
	if flagged < 8 {
		t.Fatalf("IsolationForest flagged %d/10 anomalies", flagged)
	}
	// Normal points mostly not flagged.
	falsePos := 0
	for i := 0; i < anomalyStart; i++ {
		if preds[i] == -1 {
			falsePos++
		}
	}
	if falsePos > 30 {
		t.Fatalf("IsolationForest false positives %d/%d", falsePos, anomalyStart)
	}
}

func makeGaussianBlobs(centers [][]float64, perBlob int, scale float64, seed int64) ([][]float64, []int) {
	r := rand.New(rand.NewSource(seed))
	var X [][]float64
	var y []int
	for c, cen := range centers {
		for i := 0; i < perBlob; i++ {
			row := make([]float64, len(cen))
			for j := range cen {
				row[j] = cen[j] + r.NormFloat64()*scale
			}
			X = append(X, row)
			y = append(y, c)
		}
	}
	return X, y
}

func TestLDAClassifiesGaussians(t *testing.T) {
	X, y := makeGaussianBlobs([][]float64{{-3, -3}, {3, 3}}, 60, 0.5, 11)
	m := &LDA{}
	m.Fit(X, y)
	preds := m.Predict(X)
	correct := 0
	for i := range preds {
		if preds[i] == y[i] {
			correct++
		}
	}
	if float64(correct)/float64(len(y)) < 0.95 {
		t.Fatalf("LDA accuracy %.3f", float64(correct)/float64(len(y)))
	}
}

func TestLDATransformReducesDims(t *testing.T) {
	X, y := makeGaussianBlobs([][]float64{{-3, -3, 0}, {3, 3, 0}, {0, 0, 5}}, 30, 0.5, 12)
	m := &LDA{NComponents: 2}
	m.Fit(X, y)
	red := m.Transform(X)
	if len(red) != len(X) || len(red[0]) != 2 {
		t.Fatalf("LDA transform shape %dx%d", len(red), len(red[0]))
	}
}

func TestQDAClassifiesGaussians(t *testing.T) {
	X, y := makeGaussianBlobs([][]float64{{-3, -3}, {3, 3}}, 60, 0.5, 13)
	m := &QDA{}
	m.Fit(X, y)
	preds := m.Predict(X)
	correct := 0
	for i := range preds {
		if preds[i] == y[i] {
			correct++
		}
	}
	if float64(correct)/float64(len(y)) < 0.95 {
		t.Fatalf("QDA accuracy %.3f", float64(correct)/float64(len(y)))
	}
}

func TestGMMRecoversTwoClusters(t *testing.T) {
	X, _ := makeGaussianBlobs([][]float64{{-5, -5}, {5, 5}}, 80, 0.5, 21)
	m := &GaussianMixture{NComponents: 2, MaxIter: 200, Seed: 21}
	m.Fit(X)
	// Means should be close to (-5,-5) and (5,5) in some order.
	a := euclidean(m.Means[0], []float64{-5, -5}) + euclidean(m.Means[1], []float64{5, 5})
	b := euclidean(m.Means[0], []float64{5, 5}) + euclidean(m.Means[1], []float64{-5, -5})
	best := a
	if b < best {
		best = b
	}
	if best > 1.0 {
		t.Fatalf("GMM did not recover means: %v %v", m.Means[0], m.Means[1])
	}
}

func TestMeanShiftFindsClusters(t *testing.T) {
	X, _ := makeGaussianBlobs([][]float64{{-5, -5}, {5, 5}}, 60, 0.3, 31)
	m := &MeanShift{Bandwidth: 2.0, MaxIter: 100}
	m.Fit(X)
	if len(m.Centers) < 2 || len(m.Centers) > 3 {
		t.Fatalf("MeanShift found %d centers, want 2-3", len(m.Centers))
	}
	preds := m.Predict(X)
	if len(preds) != len(X) {
		t.Fatalf("MeanShift Predict returned len %d, want %d", len(preds), len(X))
	}
}

func TestTSNEShape(t *testing.T) {
	X, _ := makeGaussianBlobs([][]float64{{-3, -3, -3}, {3, 3, 3}}, 20, 0.3, 41)
	m := &TSNE{NComponents: 2, Perplexity: 5, NIter: 150, Seed: 41}
	m.Fit(X)
	if len(m.Embedding) != len(X) || len(m.Embedding[0]) != 2 {
		t.Fatalf("TSNE shape %dx%d", len(m.Embedding), len(m.Embedding[0]))
	}
}

func TestKernelPCAShape(t *testing.T) {
	X, _ := makeGaussianBlobs([][]float64{{-3, -3}, {3, 3}}, 30, 0.3, 51)
	m := &KernelPCA{NComponents: 2, Kernel: "rbf", Gamma: 0.5}
	red := m.FitTransform(X)
	if len(red) != len(X) || len(red[0]) != 2 {
		t.Fatalf("KernelPCA shape %dx%d", len(red), len(red[0]))
	}
}

func TestFastICAShape(t *testing.T) {
	r := rand.New(rand.NewSource(61))
	n := 200
	// Two independent sources mixed by a 2x2 matrix.
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		s1 := math.Sin(float64(i) * 0.1)
		s2 := r.Float64()*2 - 1
		X[i] = []float64{0.5*s1 + 0.5*s2, 0.3*s1 - 0.7*s2}
	}
	m := &FastICA{NComponents: 2, MaxIter: 200, Seed: 61}
	red := m.FitTransform(X)
	if len(red) != n || len(red[0]) != 2 {
		t.Fatalf("FastICA shape %dx%d", len(red), len(red[0]))
	}
}

func TestBayesianRidgeFitsLine(t *testing.T) {
	w := []float64{3.0, -2.0}
	X, y := makeLinearDataset(200, w, 1.0, 0.05, 71)
	m := &BayesianRidge{MaxIter: 300}
	m.Fit(X, y)
	for i, want := range w {
		if math.Abs(m.Weights[i]-want) > 0.2 {
			t.Fatalf("BayesianRidge w[%d]=%v want %v", i, m.Weights[i], want)
		}
	}
	if math.Abs(m.Bias-1.0) > 0.2 {
		t.Fatalf("BayesianRidge bias=%v want ~1.0", m.Bias)
	}
	preds, stds := m.PredictWithStd(X)
	if len(preds) != len(X) || len(stds) != len(X) {
		t.Fatalf("PredictWithStd lengths preds=%d stds=%d", len(preds), len(stds))
	}
}
