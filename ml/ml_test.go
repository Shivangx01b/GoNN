package ml

import (
	"math"
	"math/rand"
	"testing"
)

func TestLinearRegressionFitsLine(t *testing.T) {
	rand.Seed(0)
	xs := make([][]float64, 100)
	ys := make([]float64, 100)
	for i := range xs {
		x := rand.Float64() * 10
		xs[i] = []float64{x}
		ys[i] = 3*x + 1 + rand.NormFloat64()*0.05
	}
	m := &LinearRegression{}
	m.Fit(xs, ys)
	if math.Abs(m.Weights[0]-3) > 0.05 || math.Abs(m.Bias-1) > 0.1 {
		t.Fatalf("LR: got w=%v b=%v, want ~3, ~1", m.Weights[0], m.Bias)
	}
}

func TestRidgeFitsLine(t *testing.T) {
	rand.Seed(0)
	xs := make([][]float64, 100)
	ys := make([]float64, 100)
	for i := range xs {
		x := rand.Float64() * 10
		xs[i] = []float64{x}
		ys[i] = 3*x + 1 + rand.NormFloat64()*0.05
	}
	m := &Ridge{Alpha: 0.01}
	m.Fit(xs, ys)
	if math.Abs(m.Weights[0]-3) > 0.5 {
		t.Fatalf("Ridge: got w=%v, want near 3", m.Weights[0])
	}
}

func TestKMeansFindsBlobs(t *testing.T) {
	rand.Seed(0)
	// 3 well-separated blobs
	X := make([][]float64, 0, 150)
	for c := 0; c < 3; c++ {
		cx, cy := float64(c)*10, float64(c)*10
		for i := 0; i < 50; i++ {
			X = append(X, []float64{
				cx + rand.NormFloat64()*0.3,
				cy + rand.NormFloat64()*0.3,
			})
		}
	}
	km := &KMeans{K: 3, MaxIter: 100, Seed: 1}
	km.Fit(X)
	labels := km.Predict(X)
	// Each blob should be ~50 points
	counts := [3]int{}
	for _, l := range labels {
		counts[l]++
	}
	for _, c := range counts {
		if c < 30 || c > 70 {
			t.Fatalf("KMeans clusters unbalanced: %v", counts)
		}
	}
}

func TestPCAReducesDims(t *testing.T) {
	X := [][]float64{
		{1, 2, 3, 4}, {2, 4, 6, 8}, {3, 6, 9, 12}, {4, 8, 12, 16},
	}
	p := &PCA{NComponents: 1}
	p.Fit(X)
	red := p.Transform(X)
	if len(red) != 4 || len(red[0]) != 1 {
		t.Fatalf("PCA shape: got %dx%d want 4x1", len(red), len(red[0]))
	}
}

func TestAccuracyMetric(t *testing.T) {
	a := Accuracy([]int{1, 2, 3, 4}, []int{1, 2, 4, 4})
	if math.Abs(a-0.75) > 1e-9 {
		t.Fatalf("Accuracy: got %v want 0.75", a)
	}
}

func TestStandardScalerRoundTrip(t *testing.T) {
	X := [][]float64{{1, 100}, {2, 200}, {3, 300}, {4, 400}}
	s := &StandardScaler{}
	s.Fit(X)
	Xs := s.Transform(X)
	// Each column should have mean 0 and unit std.
	for j := 0; j < 2; j++ {
		var sum, sumSq float64
		for i := 0; i < 4; i++ {
			sum += Xs[i][j]
			sumSq += Xs[i][j] * Xs[i][j]
		}
		mean := sum / 4
		variance := sumSq/4 - mean*mean
		if math.Abs(mean) > 1e-6 || math.Abs(variance-1) > 1e-6 {
			t.Fatalf("scaler col %d: mean=%v var=%v", j, mean, variance)
		}
	}
}
