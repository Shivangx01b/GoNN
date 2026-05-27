package ml

import (
	"gonum.org/v1/gonum/mat"
)

// PCA performs principal component analysis via SVD.
type PCA struct {
	NComponents       int
	Components        [][]float64 // [NComponents][d]
	Mean              []float64
	ExplainedVariance []float64
}

func (m *PCA) Fit(X [][]float64) {
	n, d := shapeOf(X)
	m.Mean = meanCol(X)
	Xc := make([]float64, n*d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Xc[i*d+j] = X[i][j] - m.Mean[j]
		}
	}
	A := mat.NewDense(n, d, Xc)
	var svd mat.SVD
	if !svd.Factorize(A, mat.SVDThin) {
		panic("PCA: SVD failed")
	}
	var V mat.Dense
	svd.VTo(&V)
	s := svd.Values(nil)
	k := m.NComponents
	if k == 0 || k > d {
		k = d
	}
	if k > len(s) {
		k = len(s)
	}
	m.Components = make([][]float64, k)
	m.ExplainedVariance = make([]float64, k)
	denom := float64(n - 1)
	if denom < 1 {
		denom = 1
	}
	for i := 0; i < k; i++ {
		row := make([]float64, d)
		for j := 0; j < d; j++ {
			row[j] = V.At(j, i)
		}
		m.Components[i] = row
		m.ExplainedVariance[i] = (s[i] * s[i]) / denom
	}
	m.NComponents = k
}

func (m *PCA) Transform(X [][]float64) [][]float64 {
	n := len(X)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, m.NComponents)
		for c := 0; c < m.NComponents; c++ {
			s := 0.0
			for j := range X[i] {
				s += (X[i][j] - m.Mean[j]) * m.Components[c][j]
			}
			out[i][c] = s
		}
	}
	return out
}

// FitTransform fits then transforms.
func (m *PCA) FitTransform(X [][]float64) [][]float64 {
	m.Fit(X)
	return m.Transform(X)
}

// InverseTransform reconstructs the original feature space.
func (m *PCA) InverseTransform(Z [][]float64) [][]float64 {
	n := len(Z)
	d := len(m.Mean)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, d)
		copy(out[i], m.Mean)
		for c := 0; c < m.NComponents; c++ {
			for j := 0; j < d; j++ {
				out[i][j] += Z[i][c] * m.Components[c][j]
			}
		}
	}
	return out
}
