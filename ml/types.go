// Package ml provides classical (non-deep-learning) machine learning algorithms
// operating on plain [][]float64 matrices and []float64 / []int labels.
package ml

// Regressor predicts continuous targets.
type Regressor interface {
	Fit(X [][]float64, y []float64)
	Predict(X [][]float64) []float64
}

// Classifier predicts discrete integer labels.
type Classifier interface {
	Fit(X [][]float64, y []int)
	Predict(X [][]float64) []int
}

// Clusterer assigns inputs to clusters.
type Clusterer interface {
	Fit(X [][]float64)
	Predict(X [][]float64) []int
}

// Transformer maps inputs to a transformed feature space.
type Transformer interface {
	Fit(X [][]float64)
	Transform(X [][]float64) [][]float64
}
